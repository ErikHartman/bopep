"""
Acquisition functions for discrete candidate optimization.

TODO: Investigate proper BoTorch multi-objective acquisition functions 
      (qLogExpectedHypervolumeImprovement) that work with discrete candidates.
      Current EHVI implementation uses a simplified heuristic approach.

Currently only supports q=1 (sequential candidate evaluation).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import Tensor, Size

from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler


# Publicly exposed names
available_acquisition_functions = {
    "expected_improvement", "upper_confidence_bound", "expected_hypervolume_improvement",
    "mean", "standard_deviation" 
}


def _is_multiobjective_predictions(predictions: Dict) -> bool:
    if not predictions:
        raise ValueError("predictions is empty.")
    sample = next(iter(predictions.values()))
    return isinstance(sample, dict)


def _ordered_objective_names_from_predictions(
    predictions: Dict[str, Dict[str, Tuple[float, float]]]
) -> List[str]:
    first = next(iter(predictions.values()))
    if not isinstance(first, dict) or not first:
        raise ValueError("Multiobjective predictions must be {item: {obj: (mean, std), ...}}.")
    return list(first.keys())


def _predictions_to_arrays(
    predictions: Dict[str, Union[Tuple[float, float], Dict[str, Tuple[float, float]]]]
) -> Tuple[List[str], Optional[List[str]], np.ndarray, np.ndarray]:
    """
    Returns:
      items: list of peptide names in a stable order
      obj_names: list of objective names if multiobjective, else None
      mu: [N, M] mean array
      sd: [N, M] std array
    """
    items = list(predictions.keys())
    if _is_multiobjective_predictions(predictions):
        obj_names = _ordered_objective_names_from_predictions(predictions)
        mu = np.array([[predictions[it][o][0] for o in obj_names] for it in items], dtype=float)
        sd = np.array([[predictions[it][o][1] for o in obj_names] for it in items], dtype=float)
        return items, obj_names, mu, sd
    else:
        mu = np.array([[predictions[it][0]] for it in items], dtype=float)
        sd = np.array([[predictions[it][1]] for it in items], dtype=float)
        return items, None, mu, sd


def _objectives_to_tensor_and_indices(
    items: List[str],
    obj_names: Optional[List[str]],
    objectives: Dict[str, Union[float, Dict[str, float]]],
) -> Tuple[Tensor, Tensor]:
    """
    Map observed outcomes to a tensor Y_obs and baseline indices aligned with items.
    Only overlapping peptides are used.
    """
    item_to_idx = {it: i for i, it in enumerate(items)}
    obs_rows = []
    baseline_idx = []
    if obj_names is None:
        for it, val in objectives.items():
            if it in item_to_idx:
                obs_rows.append([float(val)])
                baseline_idx.append([item_to_idx[it]])
    else:
        for it, valdict in objectives.items():
            if it in item_to_idx:
                row = [float(valdict[o]) for o in obj_names]
                obs_rows.append(row)
                baseline_idx.append([item_to_idx[it]])

    if not obs_rows:
        Y = torch.empty(0, 1 if obj_names is None else len(obj_names), dtype=torch.double)
        Xb = torch.empty(0, 1, dtype=torch.long)
    else:
        Y = torch.tensor(np.asarray(obs_rows, dtype=float), dtype=torch.double)
        Xb = torch.tensor(np.asarray(baseline_idx, dtype=int), dtype=torch.long)

    return Y, Xb

class TablePosteriorModel(Model):
    """
    Fixed BoTorch Model over a discrete candidate set.
    Posterior is independent across candidates and across objectives.
    """

    def __init__(self, mu: np.ndarray, sd: np.ndarray, device=None, dtype=torch.double):
        super().__init__()
        assert mu.shape == sd.shape and mu.ndim == 2, "mu and sd must be [N, M]"
        self.N, self.M = mu.shape
        self._mu = torch.as_tensor(mu, dtype=dtype, device=device)
        self._sd = torch.as_tensor(sd, dtype=dtype, device=device).clamp_min(1e-12)

    @property
    def num_outputs(self) -> int:
        return self.M

    def posterior(
        self,
        X: Tensor,
        output_indices=None,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        Return a posterior distribution for the given input indices.
        For X of shape [..., q, 1], returns posterior with proper BoTorch shapes.
        """
        # Handle [1, q, 1] -> [q, 1] conversion for acquisition functions
        if X.dim() == 3 and X.shape[0] == 1:
            X = X.squeeze(0)  # [q, 1]
            
        # X: [q, 1] -> idx: [q]
        idx = X.squeeze(-1).long()
        q = idx.shape[0]
        
        # Look up mean and std for each query point
        mu = self._mu[idx]  # [q, M]
        sd = self._sd[idx]  # [q, M]
        
        # For BoTorch compatibility: create a multivariate normal distribution
        # Flatten mean and create block diagonal covariance
        mu_flat = mu.reshape(-1)  # [q*M]
        var_flat = (sd ** 2).reshape(-1)  # [q*M]
        cov = torch.diag(var_flat)  # [q*M, q*M]
        
        # Create MVN with batch_shape=[] and event_shape=[q*M]
        mvn = MultivariateNormal(mu_flat, covariance_matrix=cov)
        
        # Create a simple posterior wrapper
        class TablePosterior(GPyTorchPosterior):
            def __init__(self, mvn_dist, q_val, m_val, mu_tensor, var_tensor):
                super().__init__(mvn_dist)
                self.q_val = q_val
                self.m_val = m_val
                self._mean = mu_tensor  # Store original [q, M] mean
                self._variance = var_tensor  # Store original [q, M] variance
            
            @property
            def mean(self):
                return self._mean  # [q, M]
            
            @property
            def variance(self):
                return self._variance  # [q, M]
                
            def sample(self, sample_shape: Size = Size(), base_samples: Optional[Tensor] = None) -> Tensor:
                """Sample from the posterior. Returns tensor of shape [..., q, M]."""
                # Sample from the flat MVN using the parent class
                flat_samples = self.mvn.sample(sample_shape)  # [..., q*M]
                # Reshape to [..., q, M]
                new_shape = flat_samples.shape[:-1] + (self.q_val, self.m_val)
                return flat_samples.reshape(new_shape)
        
        return TablePosterior(mvn, q, self.M, mu, sd ** 2)


class AcquisitionFunction:
    """
    Acquisition functions for discrete candidate optimization.
      - expected_improvement: qLogExpectedImprovement for single objective
      - upper_confidence_bound: qUpperConfidenceBound for single objective  
      - expected_hypervolume_improvement: Simplified heuristic for multi-objective
      - mean: Pure exploitation (highest predicted value)
      - standard_deviation: Pure exploration (highest uncertainty)
    """

    def __init__(self):
        pass

    def compute_acquisition(
        self,
        predictions: Dict[str, Union[Tuple[float, float], Dict[str, Tuple[float, float]]]],
        acquisition_function: str,
        *,
        objectives: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        num_mc: int = 256,
        kappa: float = 1.96,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """
        Compute acquisition values on a discrete candidate set.

        Args:
            predictions: Single-objective {name: (mean, std)} or 
                        multi-objective {name: {obj: (mean, std), ...}} predictions
            acquisition_function: One of available_acquisition_functions
            objectives: Observed outcomes dict (required for EI and EHVI)
            num_mc: MC samples for BoTorch acquisitions
            kappa: For UCB, beta = kappa**2  
            device: Torch device for computations
            
        Returns:
            Dict mapping candidate names to acquisition scores
        """
        if acquisition_function not in available_acquisition_functions:
            raise ValueError(f"Unknown acquisition '{acquisition_function}'. Allowed: {available_acquisition_functions}")

        # Handle simple acquisition functions first
        if acquisition_function in ["mean", "standard_deviation"]:
            return self._compute_simple_acquisition(predictions, acquisition_function, objectives, kappa)

        # Handle BoTorch acquisition functions
        items, obj_names, mu_np, sd_np = _predictions_to_arrays(predictions)
        N, M = mu_np.shape
        device = device or torch.device("cpu")

        model = TablePosteriorModel(mu_np, sd_np, device=device, dtype=torch.double)
        # indices tensor for candidates
        idx_tensor = torch.arange(N, dtype=torch.long, device=device).unsqueeze(-1)  # [N, 1]

        # Single-objective
        if acquisition_function in {"expected_improvement", "upper_confidence_bound"}:
            if M != 1:
                raise ValueError(f"'{acquisition_function}' requires single-objective predictions.")
            if acquisition_function == "expected_improvement":
                if not objectives:
                    raise ValueError("objectives is required for expected_improvement to compute best_f.")
                Y_obs, _ = _objectives_to_tensor_and_indices(items, None, objectives)
                if Y_obs.numel() == 0:
                    raise ValueError("No overlap between objectives and predictions. Cannot compute best_f for expected_improvement.")
                best_f = Y_obs.max().item()
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(num_mc)]))
                acq = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
            else:
                beta = float(kappa ** 2)
                acq = qUpperConfidenceBound(model=model, beta=beta)

            scores = {}
            with torch.no_grad():
                for i, name in enumerate(items):
                    x_i = idx_tensor[i : i + 1]          # [1, 1]
                    val = acq(x_i).item()
                    scores[name] = float(val)
            return scores

        # Multiobjective
        if objectives is None:
            raise ValueError(f"'objectives' is required for {acquisition_function}.")
        if obj_names is None:
            raise ValueError(f"'{acquisition_function}' requires multiobjective predictions.")

        Y_obs, X_baseline = _objectives_to_tensor_and_indices(items, obj_names, objectives)
        if Y_obs.numel() == 0:
            raise ValueError("No overlap between objectives and predictions. Provide objectives for some candidates.")

        if acquisition_function == "expected_hypervolume_improvement":
            # Simplified EHVI: Compute hypervolume improvement heuristic for each candidate
            # TODO: Replace with proper BoTorch qLogExpectedHypervolumeImprovement
            
            scores = {}
            with torch.no_grad():
                for i, name in enumerate(items):
                    x_i = idx_tensor[i : i + 1]  # [1, 1]
                    posterior = model.posterior(x_i)
                    pred_mean = posterior.mean.squeeze(0)  # [M] - predicted mean for this candidate
                    pred_point = pred_mean.numpy()
                    current_points = Y_obs.numpy()  # [N_obs, M] - existing Pareto front
                    
                    # Hypervolume improvement heuristic based on dominance relationships
                    dominates = np.all(pred_point >= current_points, axis=1)
                    dominated_by = np.any(pred_point <= current_points, axis=1)
                    
                    if np.any(dominates):
                        # Dominates existing points - highest value
                        hv_improvement = 1.0 + np.sum(dominates)
                    elif not np.any(dominated_by):
                        # Non-dominated - medium value  
                        hv_improvement = 1.0
                    else:
                        # Dominated - lower value based on distance to Pareto front
                        min_distances = np.min(np.abs(current_points - pred_point), axis=0)
                        hv_improvement = 1.0 / (1.0 + np.sum(min_distances))
                    
                    scores[name] = float(hv_improvement)
            
            return scores
        else:
            raise RuntimeError("Internal routing error.")

    def _compute_simple_acquisition(
        self, 
        predictions: Dict[str, Tuple[float, float]], 
        acquisition_function: str,
        objectives: Optional[Dict[str, float]] = None,
        kappa: float = 1.96
    ) -> Dict[str, float]:
        """Compute simple acquisition functions using only predictions."""
        # Only handle single-objective predictions for simple functions
        if not all(isinstance(v, tuple) and len(v) == 2 for v in predictions.values()):
            raise ValueError(f"Simple acquisition function '{acquisition_function}' requires single-objective predictions.")
        
        peptides = list(predictions.keys())
        means = np.array([predictions[p][0] for p in peptides], dtype=float)
        stds = np.array([predictions[p][1] for p in peptides], dtype=float)
        
        if acquisition_function == "standard_deviation":
            return self.standard_deviation(peptides, stds)
        elif acquisition_function == "mean":
            return self.mean(peptides, means)
        else:
            raise ValueError(f"Unknown simple acquisition function: {acquisition_function}")

    def standard_deviation(self, peptides: list, stds: np.ndarray):
        """Standard deviation as an acquisition function. Pure exploration."""
        return {peptide: float(std) for peptide, std in zip(peptides, stds)}

    def mean(self, peptides: list, means: np.ndarray):
        """Mean as an acquisition function (pure exploitation)."""
        return {peptide: float(mean) for peptide, mean in zip(peptides, means)}



if __name__ == "__main__":
    """Demo of acquisition functions."""
    acq = AcquisitionFunction()
    
    # Single-objective test data
    preds = {"p1": (0.2, 0.1), "p2": (0.5, 0.2), "p3": (0.3, 0.05)}
    objs = {"p1": 0.1, "p3": 0.25}
    
    # Multi-objective test data
    preds_mo = {
        "p1": {"objA": (0.2, 0.1), "objB": (0.7, 0.2)},
        "p2": {"objA": (0.6, 0.2), "objB": (0.4, 0.1)},
        "p3": {"objA": (0.3, 0.05), "objB": (0.5, 0.15)},
    }
    objs_mo = {
        "p1": {"objA": 0.10, "objB": 0.65},
        "p3": {"objA": 0.25, "objB": 0.45},
    }
    
    # Test all acquisition functions
    scores_ei = acq.compute_acquisition(preds, "expected_improvement", objectives=objs)
    scores_ucb = acq.compute_acquisition(preds, "upper_confidence_bound", kappa=1.96)
    scores_ehvi = acq.compute_acquisition(preds_mo, "expected_hypervolume_improvement", objectives=objs_mo)
    scores_std = acq.compute_acquisition(preds, "standard_deviation")
    scores_mean = acq.compute_acquisition(preds, "mean")

    print("Expected Improvement:", scores_ei)
    print("Upper Confidence Bound:", scores_ucb)
    print("Expected Hypervolume Improvement:", scores_ehvi)
    print("Standard Deviation:", scores_std)
    print("Mean:", scores_mean)
