import numpy as np
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import optuna
from scipy.stats import norm

from bopep.surrogate_model.nn_ensemble import NeuralNetworkEnsemble
from bopep.surrogate_model.mc_dropout import MonteCarloDropout
from bopep.surrogate_model.mve import MVE
from bopep.surrogate_model.deep_evidential_regression import DeepEvidentialRegression
from bopep.surrogate_model.helpers import (
    BasePredictionModel,
    VariableLengthDataset,
    variable_length_collate_fn,
)

class HyperparameterTuner:
    """
    Unified tuner that searches:
      1) Network architecture hyperparams (MLP vs BiLSTM vs BiGRU)
      2) Uncertainty hyperparam (MVE/DER reg, dropout rate, or ensemble size)
    Uses a combined metric: RMSE + NLL + MSCE + Coverage Error.
    """

    def __init__(
        self,
        model_type: Literal["mve", "deep_evidential", "nn_ensemble", "mc_dropout"],
        input_dim: int,
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        coverage_levels: List[float] = [0.5, 0.9],
        rmse_weight: float = 1.0,
        msce_weight: float = 5.0,
        coverage_weight: float = 5.0,
        device: Optional[torch.device] = None,
        n_splits: int = 3,
        n_trials: int = 20,
        random_state: int = 42,
        hidden_dim_min: int = 16,
        hidden_dim_max: int = 256,
    ):
        """
        Args:
            model_type: "mve", "deep_evidential", "nn_ensemble", or "mc_dropout"
            input_dim: feature dimension
            network_type: "mlp", "bilstm", or "bigru"
            coverage_levels: coverage alphas for calibration
            rmse_weight, msce_weight, coverage_weight: weighting in combined score
            device: Torch device
            n_splits: K-fold CV splits
            n_trials: Number of Optuna trials
            random_state: for cross-validation reproducibility
            hidden_dim_min, hidden_dim_max: Range for hidden dims in MLP or RNN
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.network_type = network_type
        self.coverage_levels = coverage_levels
        self.rmse_weight = rmse_weight
        self.msce_weight = msce_weight
        self.coverage_weight = coverage_weight
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

        self.hidden_dim_min = hidden_dim_min
        self.hidden_dim_max = hidden_dim_max

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.best_params: Optional[Dict[str, Union[float, List[int]]]] = None
        self.best_score = float("inf")

    def _create_model(
        self,
        param_value: float,
        hidden_dims: Optional[List[int]] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
    ) -> "BasePredictionModel":
        """
        Build the actual model (MVE, DER, nn_ensemble, or mc_dropout)
        for either MLP or RNN style (BiLSTM/BiGRU).
        """
        if self.model_type == "mve":
            return MVE(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                mve_regularization=param_value,
            )
        elif self.model_type == "deep_evidential":
            return DeepEvidentialRegression(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                evidential_regularization=param_value,
            )
        elif self.model_type == "mc_dropout":
            return MonteCarloDropout(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                dropout_rate=param_value,
            )
        elif self.model_type == "nn_ensemble":
            return NeuralNetworkEnsemble(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                n_networks=int(param_value),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _evaluate_model(
        self, model: "BasePredictionModel", val_loader: DataLoader
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns (combined_score, rmse, nll, msce, coverage_err).
        """
        model.eval()
        model.to(self.device)

        all_means, all_stds, all_targets = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, lengths in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).view(-1, 1)

                means, stds = model.forward_predict(batch_x, lengths=lengths)
                all_means.append(means)
                all_stds.append(stds)
                all_targets.append(batch_y)

        means = torch.cat(all_means)
        stds = torch.cat(all_stds)
        targets = torch.cat(all_targets)

        nll = self.negative_log_likelihood(means, stds, targets)
        rmse = torch.sqrt(torch.mean((means - targets) ** 2)).item()
        msce = self.reliability_calibration_error(means, stds, targets)
        coverage_err = self.coverage_calibration_error(means, stds, targets)

        combined_score = (
            nll
            + self.rmse_weight * rmse
            + self.msce_weight * msce
            + self.coverage_weight * coverage_err
        )
        return combined_score, rmse, nll, msce, coverage_err

    @staticmethod
    def negative_log_likelihood(
        means: torch.Tensor, stds: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Standard Gaussian NLL."""
        stds = torch.clamp(stds, min=1e-9)
        nll = 0.5 * torch.log(2 * np.pi * stds**2) + 0.5 * ((targets - means) ** 2) / (stds**2)
        return nll.mean().item()

    @staticmethod
    def reliability_calibration_error(means, stds, targets, quantiles=None):
        """
        We measure how often the absolute normalized residual
        is below each standard Normal z-value for various coverage levels.
        Returns the mean squared difference between the nominal coverage
        and the observed coverage across all quantiles.
        """
        if quantiles is None:
            quantiles = np.linspace(0.05, 0.95, 10)  # example range

        residuals = (targets - means) / stds  # shape (N,)
        residuals = residuals.cpu().numpy().ravel()
        
        errors = []
        for q in quantiles:
            z_q = norm.ppf((1.0 + q) / 2.0)  # e.g. q=0.9 -> z_q=1.645
            observed_frac = np.mean(np.abs(residuals) < z_q)
            errors.append((observed_frac - q)**2)
        return np.mean(errors)

    def coverage_calibration_error(
        self, means: torch.Tensor, stds: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """
        Average absolute difference between nominal and empirical coverage
        for the coverage_levels provided.
        """
        means_np = means.cpu().numpy().flatten()
        stds_np = stds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        coverage_error_sum = 0.0
        for alpha in self.coverage_levels:
            # z-value for two-sided normal coverage
            z_value = norm.ppf(0.5 + alpha / 2.0)
            lower = means_np - z_value * stds_np
            upper = means_np + z_value * stds_np
            frac_in_interval = np.mean((targets_np >= lower) & (targets_np <= upper))
            coverage_error_sum += abs(frac_in_interval - alpha)

        return coverage_error_sum / len(self.coverage_levels)

    def _fit_model(
        self,
        model: "BasePredictionModel",
        train_embed_dict: Dict[str, np.ndarray],
        train_scores_dict: Dict[str, float],
        epochs: int,
        learning_rate: float,
        batch_size: int = 64,
    ):
        """
        Just calls model.fit_dict(...) with your chosen hyperparams.
        You can add early stopping or other logic if you wish.
        """
        model.fit_dict(
            embedding_dict=train_embed_dict,
            scores_dict=train_scores_dict,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=self.device,
            verbose=False,
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Samples architecture (depending on network_type), 
        plus the uncertainty hyperparam.
        Then does cross-validation and returns average combined score.
        """
        if self.network_type == "mlp":
            n_layers = trial.suggest_int("num_layers", 1, 5)
            hidden_dims = []
            for i in range(n_layers):
                hd = trial.suggest_int(
                    f"hidden_dim_{i}", self.hidden_dim_min, self.hidden_dim_max, log=True
                )
                hidden_dims.append(hd)
            chosen_hidden_dim = None
        else:
            n_layers = trial.suggest_int("num_layers", 1, 5)
            chosen_hidden_dim = trial.suggest_int(
                "rnn_hidden_dim", self.hidden_dim_min, self.hidden_dim_max, log=True
            )
            hidden_dims = None  # Not used for RNN

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 50, 200)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        if self.model_type in ["mve", "deep_evidential"]:
            param_value = trial.suggest_float("uncertainty_param", 0, 1)
        elif self.model_type == "mc_dropout":
            param_value = trial.suggest_float("uncertainty_param", 0.01, 0.8)
        elif self.model_type == "nn_ensemble":
            param_value = trial.suggest_int("uncertainty_param", 2, 10)
        else:
            raise ValueError(f"Unknown model_type {self.model_type}")

        keys = list(self.embedding_dict.keys())
        indices = np.arange(len(keys))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        cv_scores = []
        for train_idx, val_idx in kf.split(indices):
            train_embed_dict = {keys[i]: self.embedding_dict[keys[i]] for i in train_idx}
            train_scores_dict = {keys[i]: self.scores_dict[keys[i]] for i in train_idx}

            val_embed_dict = {keys[i]: self.embedding_dict[keys[i]] for i in val_idx}
            val_scores_dict = {keys[i]: self.scores_dict[keys[i]] for i in val_idx}

            model = self._create_model(
                param_value=param_value,
                hidden_dims=hidden_dims,     
                hidden_dim=chosen_hidden_dim, 
                num_layers=n_layers,
            )
            model.to(self.device)

            self._fit_model(
                model,
                train_embed_dict,
                train_scores_dict,
                epochs,
                learning_rate,
                batch_size=batch_size,
            )

            val_dataset = VariableLengthDataset(val_embed_dict, val_scores_dict)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=variable_length_collate_fn,
            )

            combined_score, _, _, _, _ = self._evaluate_model(model, val_loader)
            cv_scores.append(combined_score)

        avg_score = float(np.mean(cv_scores))

        if avg_score < self.best_score:
            self.best_score = avg_score
            self.best_params = {
                "network_type": self.network_type,
                "num_layers": n_layers,
                "hidden_dims": hidden_dims if hidden_dims is not None else None,
                "hidden_dim": chosen_hidden_dim if chosen_hidden_dim is not None else None,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "uncertainty_param": param_value,
                "avg_score": avg_score,
            }

        return avg_score

    def tune(
        self,
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float],
        previous_study: Optional[optuna.study.Study] = None,
    ) -> Tuple[Dict[str, Union[float, List[int], None]], optuna.study.Study]:
        """
        Main entry point: run the Optuna study for cross-validation 
        and store best hyperparams in self.best_params.
        
        Args:
            embedding_dict: Dictionary of embeddings
            scores_dict: Dictionary of scores
            previous_study: Optional previous Optuna study to warm-start from
            
        Returns:
            Tuple of (best_params, study) where study can be reused in future calls
        """
        self.embedding_dict = embedding_dict
        self.scores_dict = scores_dict

        # Create a new study without inheriting old trials
        study = optuna.create_study(direction="minimize")
        
        # If we have a previous study, use it to seed initial configurations
        # but don't keep the old scores - re-evaluate them on new data
        if previous_study is not None and previous_study.trials:
            # Sort trials by their values (best trials first)
            sorted_trials = sorted(
                [t for t in previous_study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                key=lambda t: t.value
            )
            
            # Take the top 5 best trials from previous study
            num_trials_to_seed = min(5, len(sorted_trials))
            for i in range(num_trials_to_seed):
                trial = sorted_trials[i]
                # Enqueue the trial for the new study, but it will be re-evaluated
                study.enqueue_trial(trial.params)
            
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)

        return self.best_params or {}, study


def tune_hyperparams(
    model_type: Literal["mve", "deep_evidential", "nn_ensemble", "mc_dropout"],
    embedding_dict: Dict[str, np.ndarray],
    scores_dict: Dict[str, float],
    network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
    coverage_levels: List[float] = [0.5, 0.9],
    rmse_weight: float = 1.0,
    msce_weight: float = 1.0,
    coverage_weight: float = 1.0,
    n_splits: int = 3,
    n_trials: int = 20,
    random_state: int = 42,
    device: Optional[torch.device] = None,
    hidden_dim_min: int = 16,
    hidden_dim_max: int = 256,
    previous_study: Optional[optuna.study.Study] = None,
) -> Tuple[Dict[str, Union[float, List[int], None]], optuna.study.Study]:
    """
    High-level API for tuning architecture + uncertainty hyperparams with 
    MLP/BiLSTM/BiGRU options, using a combined calibration metric.
    
    Returns:
        Tuple of (best_params, study) where study can be reused in future calls
    """

    sample_embedding = next(iter(embedding_dict.values()))
    if sample_embedding.ndim == 2:
        input_dim = sample_embedding.shape[1]
    else:
        input_dim = sample_embedding.shape[0]
    
    tuner = HyperparameterTuner(
        model_type=model_type,
        input_dim=input_dim,
        network_type=network_type,
        coverage_levels=coverage_levels,
        rmse_weight=rmse_weight,
        msce_weight=msce_weight,
        coverage_weight=coverage_weight,
        device=device,
        n_splits=n_splits,
        n_trials=n_trials,
        random_state=random_state,
        hidden_dim_min=hidden_dim_min,
        hidden_dim_max=hidden_dim_max,
    )

    best_params, study = tuner.tune(embedding_dict, scores_dict, previous_study)
    return best_params, study
