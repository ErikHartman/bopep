import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Union

available_acquisition_functions = [
    "expected_improvement",
    "standard_deviation",
    "upper_confidence_bound",
    "probability_of_improvement",
    "mean",
    # multiobjective
    "parego_chebyshev_ei",
    "parego_chebyshev_ucb",
]

def _simplex_random_weights(m, rng):
    w = rng.gamma(1.0, 1.0, size=m)
    w /= np.sum(w)
    # avoid degeneracy
    w = np.maximum(w, 1e-6)
    w /= np.sum(w)
    return w

class AcquisitionFunction:
    def __init__(self, rng_seed: int = 123):
        self.best_so_far_ei = 0.0
        self.best_so_far_pi = 0.0
        self.rng = np.random.default_rng(rng_seed)

        # For ParEGO running reference points
        self._ideal = None   # z*: per-objective ideal
        self._nadir = None   # approximate nadir for normalization

    def compute_acquisition(self, predictions: dict, acquisition_function: str, maximize: bool = True, **kwargs):
        """
        Single-objective predictions: {sequence: (mean, std)}
        Multiobjective predictions: {sequence: {obj: (mean, std), ...}}
        """

        # Detect multiobjective if first value is a dict
        first_val = next(iter(predictions.values()))
        if isinstance(first_val, dict) and acquisition_function.startswith("parego_chebyshev"):
            if acquisition_function == "parego_chebyshev_ei":
                return self.parego_chebyshev_ei(predictions, **kwargs)
            elif acquisition_function == "parego_chebyshev_ucb":
                return self.parego_chebyshev_ucb(predictions, **kwargs)
            else:
                raise ValueError(f"Unknown MOO acquisition {acquisition_function}")

        # Single-objective path
        sequences = list(predictions.keys())
        means = np.array([predictions[p][0] for p in sequences])
        stds  = np.array([predictions[p][1] for p in sequences])
        
        if acquisition_function == "expected_improvement":
            return self.expected_improvement(sequences, means, stds, maximize=maximize)
        elif acquisition_function == "upper_confidence_bound":
            return self.upper_confidence_bound(sequences, means, stds, maximize=maximize, **kwargs)
        elif acquisition_function == "probability_of_improvement":
            return self.probability_of_improvement(sequences, means, stds, maximize=maximize)
        elif acquisition_function == "standard_deviation":
            return self.standard_deviation(sequences, stds)
        elif acquisition_function == "mean":
            return self.mean(sequences, means, maximize=maximize)
        else:
            raise ValueError(f"Acquisition function '{acquisition_function}' not recognized.")

    def expected_improvement(self, sequences: list, means: np.ndarray, stds: np.ndarray, maximize: bool = True):
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        if maximize:
            improvement = means - self.best_so_far_ei
            current_best_pred = np.max(means)
            if current_best_pred > self.best_so_far_ei:
                self.best_so_far_ei = current_best_pred
        else:
            improvement = self.best_so_far_ei - means  # For minimization, improvement is negative of usual
            current_best_pred = np.min(means)
            if current_best_pred < self.best_so_far_ei:
                self.best_so_far_ei = current_best_pred

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.divide(improvement, stds, out=np.zeros_like(improvement), where=(stds > 1e-12))

        ei = improvement * norm.cdf(Z) + stds * norm.pdf(Z)
        ei[stds <= 1e-12] = 0.0

        return {sequence: float(value) for sequence, value in zip(sequences, ei)}

    def upper_confidence_bound(self, sequences: list, means, stds, kappa=1.96, maximize: bool = True):
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        
        if maximize:
            # For maximization: mean + kappa * std (optimistic upper bound)
            ucb = means + kappa * stds
        else:
            # For minimization: mean - kappa * std (optimistic lower bound)
            ucb = means - kappa * stds
            
        return {sequence: float(value) for sequence, value in zip(sequences, ucb)}

    def probability_of_improvement(self, sequences, means, stds, maximize=True):
        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        if maximize:
            improvement = means - self.best_so_far_pi
        else:
            improvement = self.best_so_far_pi - means
            
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.divide(improvement, stds, out=np.zeros_like(improvement), where=(stds > 1e-12))
        pi = norm.cdf(Z)
        pi[stds <= 1e-12] = 0.0

        if maximize:
            current_best_pred = np.max(means)
            if current_best_pred > self.best_so_far_pi:
                self.best_so_far_pi = current_best_pred
        else:
            current_best_pred = np.min(means)
            if current_best_pred < self.best_so_far_pi:
                self.best_so_far_pi = current_best_pred

        return {sequence: float(value) for sequence, value in zip(sequences, pi)}

    def standard_deviation(self, sequences, stds):
        return {sequence: float(std) for sequence, std in zip(sequences, stds)}

    def mean(self, sequences, means, maximize=True):
        return {sequence: float(mean) for sequence, mean in zip(sequences, means)}

    def _extract_arrays(self, predictions_moo: dict, objective_order: Union[list, None]):
        """Return sequences, obj_names, means [N,M], stds [N,M]."""
        sequences = list(predictions_moo.keys())
        if objective_order is None:
            obj_names = list(next(iter(predictions_moo.values())).keys())
        else:
            obj_names = list(objective_order)

        N = len(sequences)
        M = len(obj_names)
        means = np.zeros((N, M), dtype=float)
        stds  = np.zeros((N, M), dtype=float)
        for i, p in enumerate(sequences):
            for j, o in enumerate(obj_names):
                m, s = predictions_moo[p][o]
                means[i, j] = float(m)
                stds[i, j]  = float(s)
        return sequences, obj_names, means, stds

    def _update_ref_points(self, means: np.ndarray, directions: np.ndarray):
        """
        Update running ideal and nadir using current predictive means.
        directions: +1 for maximize, -1 for minimize
        We convert to minimization internally by flipping maximizing objectives.
        """
        # Convert to minimization space
        Y = means.copy()
        Y[:, directions == 1] *= -1.0

        z_star = np.min(Y, axis=0)
        z_nadir = np.max(Y, axis=0)

        if self._ideal is None:
            self._ideal = z_star
            self._nadir = z_nadir
        else:
            self._ideal = np.minimum(self._ideal, z_star)
            self._nadir = np.maximum(self._nadir, z_nadir)

        # Guard against zero range
        span = np.maximum(self._nadir - self._ideal, 1e-12)
        return self._ideal, self._nadir, span

    def _normalize_min_space(self, samples: np.ndarray, ideal: np.ndarray, span: np.ndarray):
        """
        Normalize in minimization space to [0,1] roughly.
        samples shape: [..., M]
        """
        return (samples - ideal) / span

    def _aug_tchebycheff(self, f_norm: np.ndarray, lam: np.ndarray, rho: float = 0.05):
        """
        f_norm shape: [..., M] in minimization space
        lam shape: [M]
        """
        cheb = np.max(lam * f_norm, axis=-1)
        aug = cheb + rho * np.sum(lam * f_norm, axis=-1)
        return aug

    def parego_chebyshev_ei(
        self,
        predictions_moo: dict,
        objective_order: Union[list, None] = None,
        objective_directions: Union[dict, None] = None,
        weights: Union[np.ndarray, None] = None,
        rho: float = 0.05,
        n_mc: int = 256,
        best_scalar_so_far: Union[float, None] = None,
        rng_seed: Union[int, None] = None,
    ):
        """
        ParEGO using augmented Tchebycheff scalarization with EI computed by Monte Carlo.

        predictions_moo: {sequence: {obj: (mean, std), ...}, ...}
        objective_directions: {obj: "max" or "min"}; default = "max" for all
        weights: optional simplex weights; default = random simplex
        best_scalar_so_far: running best value of the scalarized objective in minimization space.
                            If None, it will be estimated from current means.
        """
        rng = self.rng if rng_seed is None else np.random.default_rng(rng_seed)

        sequences, obj_names, means, stds = self._extract_arrays(predictions_moo, objective_order)
        M = len(obj_names)
        N = len(sequences)

        # Directions
        if objective_directions is None:
            directions = np.ones(M, dtype=int)  # maximize by default
        else:
            directions = np.array([1 if objective_directions[o].lower().startswith("max") else -1 for o in obj_names], dtype=int)

        # Update reference points and spans using means
        ideal, nadir, span = self._update_ref_points(means, directions)

        # Choose weights
        lam = _simplex_random_weights(M, rng) if weights is None else np.asarray(weights, dtype=float)
        lam = np.maximum(lam, 1e-9)
        lam /= np.sum(lam)

        # Build Monte Carlo samples of objectives at each candidate
        # Convert to minimization space: if direction is maximize, flip sign
        flip = np.where(directions == 1, -1.0, 1.0)  # +1 maximize -> multiply by -1
        # samples: [N, n_mc, M]
        eps = rng.standard_normal(size=(N, n_mc, M))
        samples = means[:, None, :] + stds[:, None, :] * eps
        samples = samples * flip  # to minimization
        # normalize
        samples_norm = self._normalize_min_space(samples, ideal, span)

        # Scalarize each sample
        g_samples = self._aug_tchebycheff(samples_norm, lam, rho=rho)  # [N, n_mc], lower is better

        # Current scalarized mean estimate to set baseline if needed
        means_min = means * flip
        means_norm = self._normalize_min_space(means_min, ideal, span)
        g_mean = self._aug_tchebycheff(means_norm, lam, rho=rho)  # [N]

        if best_scalar_so_far is None:
            # For EI we need a baseline to improve on. Use the best current mean scalarization.
            best_scalar_so_far = float(np.min(g_mean))

        # Improvement in minimization: I = max(0, best_so_far - g)
        improv = np.maximum(0.0, best_scalar_so_far - g_samples)
        ei = np.mean(improv, axis=1)

        return {sequences[i]: float(ei[i]) for i in range(N)}

    def parego_chebyshev_ucb(
        self,
        predictions_moo: dict,
        objective_order: Union[list, None] = None,
        objective_directions: Union[dict, None] = None,
        weights: Union[np.ndarray, None] = None,
        rho: float = 0.05,
        kappa: float = 1.0,
    ):
        """
        Fast surrogate: UCB in Chebyshev space.
        Build an optimistic point per objective, normalize, then Chebyshev scalarize.
        """
        sequences, obj_names, means, stds = self._extract_arrays(predictions_moo, objective_order)
        M = len(obj_names)
        N = len(sequences)

        # Directions
        if objective_directions is None:
            directions = np.ones(M, dtype=int)
        else:
            directions = np.array([1 if objective_directions[o].lower().startswith("max") else -1 for o in obj_names], dtype=int)

        # Update reference points
        ideal, nadir, span = self._update_ref_points(means, directions)

        # We form optimistic points in minimization space
        flip = np.where(directions == 1, -1.0, 1.0)

        # For minimization, LCB = mean - kappa*std; for maximization, UCB = mean + kappa*std
        # After flipping to minimization space, a single formula works:
        optimistic = means + (directions * kappa) * stds  # maximize -> mean + kappa*std, minimize -> mean - kappa*std
        optimistic_min = optimistic * flip
        optimistic_norm = self._normalize_min_space(optimistic_min, ideal, span)

        lam = _simplex_random_weights(M, self.rng) if weights is None else np.asarray(weights, dtype=float)
        lam = np.maximum(lam, 1e-9)
        lam /= np.sum(lam)

        g_ucb = self._aug_tchebycheff(optimistic_norm, lam, rho=rho)
        # We return negative because higher acquisition is better in your API
        scores = -g_ucb
        return {sequences[i]: float(scores[i]) for i in range(N)}


if __name__ == "__main__":
    # example
    acq = AcquisitionFunction()
    preds = {
        "pep1": {"obj1": (0.5, 0.1), "obj2": (1.0, 0.2)},
        "pep2": {"obj1": (0.6, 0.15), "obj2": (0.9, 0.25)},
        "pep3": {"obj1": (0.4, 0.05), "obj2": (1.1, 0.1)},
    }
    print(acq.parego_chebyshev_ei(preds))
    print(acq.parego_chebyshev_ucb(preds))