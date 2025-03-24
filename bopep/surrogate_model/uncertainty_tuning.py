import numpy as np
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
from scipy.stats import norm

from bopep.surrogate_model.nn_ensemble import NeuralNetworkEnsemble
from bopep.surrogate_model.mc_dropout import MonteCarloDropout
from bopep.surrogate_model.mve import MVE
from bopep.surrogate_model.deep_evidential_regression import DeepEvidentialRegression
from bopep.surrogate_model.helpers import BasePredictionModel


class UncertaintyTuner:
    """
    Tunes the uncertainty-related hyperparameter for various surrogate models.

    Can do coverage-based calibration, reliability-based calibration, or both.
    """

    def __init__(
        self,
        model_type: Literal["mve", "der", "nn_ensemble", "mc_dropout"],
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        coverage_levels: List[float] = [0.5, 0.9],
        calibration_metric: Literal["msce", "coverage", "both"] = "both",
        coverage_weight: float = 1.0,
        msce_weight: float = 1.0,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.network_type = network_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.coverage_levels = coverage_levels
        self.calibration_metric = calibration_metric
        self.coverage_weight = coverage_weight
        self.msce_weight = msce_weight
        self.model_kwargs = model_kwargs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.param_ranges = {
            "mve": np.logspace(-4, 0, 20),
            "der": np.logspace(-4, 0, 20),
            "nn_ensemble": np.arange(2, 16),
            "mc_dropout": np.linspace(0.05, 0.8, 20),
        }

    def _create_model(self, param_value: float) -> BasePredictionModel:
        if self.model_type == "mve":
            return MVE(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                network_type=self.network_type,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                mve_regularization=param_value,
                **self.model_kwargs,
            )
        elif self.model_type == "der":
            return DeepEvidentialRegression(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                network_type=self.network_type,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                evidential_regularization=param_value,
                **self.model_kwargs,
            )
        elif self.model_type == "nn_ensemble":
            return NeuralNetworkEnsemble(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                network_type=self.network_type,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                n_networks=int(param_value),
                **self.model_kwargs,
            )
        elif self.model_type == "mc_dropout":
            return MonteCarloDropout(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                network_type=self.network_type,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                dropout_rate=param_value,
                **self.model_kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @staticmethod
    def negative_log_likelihood(
        means: torch.Tensor, stds: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Calculate the negative log likelihood (NLL) under a normal assumption."""
        stds = torch.clamp(stds, min=1e-6)  # avoid log(0)
        nll = 0.5 * torch.log(2 * np.pi * stds**2) + 0.5 * (
            (targets - means) ** 2
        ) / (stds**2)
        return nll.mean().item()

    @staticmethod
    def reliability_calibration_error(
        means: torch.Tensor, stds: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
    ) -> float:
        """
        Approximate calibration error using a reliability-diagram style approach (MSCE).
        """
        means_np = means.cpu().numpy().flatten()
        stds_np = stds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        residuals = (targets_np - means_np) / stds_np
        expected_props = np.linspace(0, 1, n_bins + 1)[1:]

        observed_props = []
        for q in expected_props:
            # threshold at the q-th percentile (in absolute value) of residual distribution
            threshold = np.sqrt(2) * np.abs(np.percentile(residuals, q * 100))
            observed_props.append(np.mean(np.abs(residuals) < threshold))

        observed_props = np.array(observed_props)
        expected_props = np.array(expected_props)

        msce = np.mean((observed_props - expected_props) ** 2)
        return msce

    def coverage_calibration_error(
        self, means: torch.Tensor, stds: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        Compute how well the predicted intervals match nominal coverage levels.
        For each coverage alpha in self.coverage_levels, we find the predicted alpha% interval
        under N(mean, std) and check the fraction of targets that actually lie within.

        We then compute the average absolute difference from the nominal coverage.
        """
        means_np = means.cpu().numpy().flatten()
        stds_np = stds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        coverage_error_sum = 0.0

        for alpha in self.coverage_levels:
            # E.g. alpha=0.9 => 90% coverage -> z ~ 1.645
            # Two-sided, so we do (1 - alpha)/2 in each tail
            z_value = norm.ppf(0.5 + alpha / 2.0)  # 0.5 + alpha/2 for upper quantile
            lower = means_np - z_value * stds_np
            upper = means_np + z_value * stds_np

            # fraction of points actually inside [lower, upper]
            frac_in_interval = np.mean((targets_np >= lower) & (targets_np <= upper))
            coverage_error_sum += abs(frac_in_interval - alpha)

        # Average over coverage_levels
        coverage_error = coverage_error_sum / len(self.coverage_levels)
        return coverage_error

    def _evaluate_model(
        self,
        model: BasePredictionModel,
        val_dataloader: DataLoader,
        return_components: bool = False,
    ) -> Union[float, Tuple[float, float, float, float, float]]:
        """
        Evaluate the model's predictions on the validation set to get:
          - RMSE (pointwise accuracy)
          - NLL (sharpness + correctness)
          - Reliability-based MSCE
          - Coverage-based error
        Then combine calibration metrics according to self.calibration_metric.

        Returns:
            If return_components=False, return the combined "score" used for hyperparam selection.
            If return_components=True, return (combined_score, rmse, nll, msce, coverage_err).
        """
        model.eval()
        model.to(self.device)

        all_means = []
        all_stds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, lengths in val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                means, stds = model.forward_predict(inputs, lengths=lengths)

                all_means.append(means)
                all_stds.append(stds)
                all_targets.append(targets)

        means = torch.cat(all_means)
        stds = torch.cat(all_stds)
        targets = torch.cat(all_targets)

        nll = self.negative_log_likelihood(means, stds, targets)
        rmse = torch.sqrt(torch.mean((means - targets) ** 2)).item()
        msce = self.reliability_calibration_error(means, stds, targets)
        coverage_err = self.coverage_calibration_error(means, stds, targets)

        if self.calibration_metric == "msce":
            calibration_term = msce * self.msce_weight
        elif self.calibration_metric == "coverage":
            calibration_term = coverage_err * self.coverage_weight
        else:
            calibration_term = (msce * self.msce_weight) + (
                coverage_err * self.coverage_weight
            )

        combined_score = nll + calibration_term

        if return_components:
            return combined_score, rmse, nll, msce, coverage_err
        else:
            return combined_score

    def tune_parameter(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lengths: Optional[List[int]] = None,
        n_splits: int = 3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        param_values: Optional[List[float]] = None,
    ) -> Dict:
        """
        Manual search over param_values with cross-validation.
        (Grid or random search if param_values is large or random.)

        Returns a dictionary with the best param, plus lists of all metrics.
        """
        if param_values is None:
            param_values = self.param_ranges[self.model_type]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        param_scores = []
        param_rmse = []
        param_nll = []
        param_msce = []
        param_cov = []

        for param in param_values:
            print(f"Evaluating {self.model_type} with param value: {param}")
            cv_scores = []
            cv_rmses = []
            cv_nlls = []
            cv_msces = []
            cv_coverrs = []

            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                train_lengths = [lengths[i] for i in train_idx] if lengths else None
                val_lengths = [lengths[i] for i in val_idx] if lengths else None

                model = self._create_model(param)
                model.to(self.device)

                val_dataset = TensorDataset(
                    X_val,
                    y_val,
                    (
                        torch.tensor(val_lengths)
                        if val_lengths
                        else torch.zeros(X_val.shape[0])
                    ),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                # Build the dict needed for fit_dict
                embedding_dict = {}
                scores_dict = {}
                for i, idx in enumerate(train_idx):
                    key = f"sample_{i}"
                    embedding_dict[key] = X_train[i].cpu().numpy()
                    scores_dict[key] = float(y_train[i].item())

                # Train
                model.fit_dict(
                    embedding_dict=embedding_dict,
                    scores_dict=scores_dict,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=self.device,
                    verbose=False,
                )

                # Evaluate
                cscore, rmse, nll, msce, coverr = self._evaluate_model(
                    model, val_loader, return_components=True
                )

                cv_scores.append(cscore)
                cv_rmses.append(rmse)
                cv_nlls.append(nll)
                cv_msces.append(msce)
                cv_coverrs.append(coverr)

            param_scores.append(np.mean(cv_scores))
            param_rmse.append(np.mean(cv_rmses))
            param_nll.append(np.mean(cv_nlls))
            param_msce.append(np.mean(cv_msces))
            param_cov.append(np.mean(cv_coverrs))

            print(
                f"  Avg. Combined Score: {param_scores[-1]:.4f}, "
                f"RMSE: {param_rmse[-1]:.4f}, NLL: {param_nll[-1]:.4f}, "
                f"MSCE: {param_msce[-1]:.4f}, CoverageErr: {param_cov[-1]:.4f}"
            )

        best_idx = np.argmin(param_scores)
        best_param = param_values[best_idx]

        results = {
            "best_param": best_param,
            "best_score": param_scores[best_idx],
            "param_values": param_values,
            "scores": param_scores,
            "rmse": param_rmse,
            "nll": param_nll,
            "msce": param_msce,
            "coverage_err": param_cov,
        }

        return results

    def tune_parameter_optuna(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lengths: Optional[List[int]] = None,
        n_splits: int = 3,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        n_trials: int = 20,
    ) -> Dict:
        """
        Use Optuna to automatically search for the best hyperparameter (e.g., reg coeff).
        Returns a dictionary with the best parameter and a summary of the best score.
        """
        # For RNN-based models, if no lengths are given, assume full length
        if lengths is None and (self.network_type in ["bilstm", "bigru"]):
            lengths = [X.shape[1]] * X.shape[0]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        def objective(trial):
            # Define how we sample the parameter based on model_type
            if self.model_type in ["mve", "der"]:
                # log-uniform from 1e-3 to 1.0 or so
                param_value = trial.suggest_float("param_value", 1e-3, 1.0, log=True)
            elif self.model_type == "mc_dropout":
                # uniform from 0.05 to 0.5
                param_value = trial.suggest_float("param_value", 0.05, 0.5, log=False)
            elif self.model_type == "nn_ensemble":
                # integer from 2 to 15
                param_value = trial.suggest_int("param_value", 2, 15)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            cv_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                train_lengths = [lengths[i] for i in train_idx] if lengths else None
                val_lengths = [lengths[i] for i in val_idx] if lengths else None

                model = self._create_model(param_value)
                model.to(self.device)

                val_dataset = TensorDataset(
                    X_val,
                    y_val,
                    (
                        torch.tensor(val_lengths)
                        if val_lengths
                        else torch.zeros(X_val.shape[0])
                    ),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

                # Build dict for fit_dict
                embedding_dict = {}
                scores_dict = {}
                for j, idx in enumerate(train_idx):
                    key = f"sample_{j}"
                    embedding_dict[key] = X_train[j].cpu().numpy()
                    scores_dict[key] = float(y_train[j].item())

                # Train
                model.fit_dict(
                    embedding_dict=embedding_dict,
                    scores_dict=scores_dict,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=self.device,
                    verbose=False,
                )

                # Evaluate on val
                cscore = self._evaluate_model(
                    model, val_loader, return_components=False
                )
                cv_scores.append(cscore)

            return float(np.mean(cv_scores))

        # Create and run the study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Retrieve results
        best_param_value = study.best_params["param_value"]
        best_score = study.best_value

        results = {
            "best_param": best_param_value,
            "best_score": best_score,
            "study": study,
        }
        return results


def tune_uncertainty_parameter(
    model_type: Literal["mve", "der", "nn_ensemble", "mc_dropout"],
    X: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    lengths: Optional[List[int]] = None,
    hidden_dims: List[int] = [128, 64],
    network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
    num_layers: int = 1,
    hidden_dim: Optional[int] = None,
    epochs: int = 50,
    batch_size: int = 32,
    n_splits: int = 3,
    param_values: Optional[List[float]] = None,
    # New calibration config:
    calibration_metric: Literal["msce", "coverage", "both"] = "msce",
    coverage_levels: List[float] = [0.5, 0.9],
    coverage_weight: float = 1.0,
    msce_weight: float = 1.0,
    **model_kwargs,
) -> float:
    """
    Convenience function to tune an uncertainty hyperparameter (reg coeff, dropout, ensemble size).
    Uses cross-validation over param_values by default.
    If you want Bayesian optimization via Optuna, call tuner.tune_parameter_optuna.
    """
    tuner = UncertaintyTuner(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        network_type=network_type,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        coverage_levels=coverage_levels,
        calibration_metric=calibration_metric,
        coverage_weight=coverage_weight,
        msce_weight=msce_weight,
        **model_kwargs,
    )

    results = tuner.tune_parameter(
        X=X,
        y=y,
        lengths=lengths,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        param_values=param_values,
    )

    return results["best_param"]
