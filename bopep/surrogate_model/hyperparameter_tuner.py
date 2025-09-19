import numpy as np
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import optuna
import math

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
    Uses NLL as the primary metric, supports both single and multi-objective optimization.
    For multi-objective, NLLs are summed across objectives.
    """

    def __init__(
        self,
        model_type: Literal["mve", "deep_evidential", "nn_ensemble", "mc_dropout"],
        input_dim: int,
        n_objectives: int = 1,
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        device: Optional[torch.device] = None,
        n_splits: int = 3,
        n_trials: int = 20,
        random_state: int = 42,
        hidden_dim_min: int = 16,
        hidden_dim_max: int = 256,
        uncertainty_param_min: Optional[Union[float, int]] = None,
        uncertainty_param_max: Optional[Union[float, int]] = None,
    ):
        """
        Args:
            model_type: "mve", "deep_evidential", "nn_ensemble", or "mc_dropout"
            input_dim: feature dimension
            n_objectives: number of objectives (1 for single-objective, >1 for multi-objective)
            network_type: "mlp", "bilstm", or "bigru"
            device: Torch device
            n_splits: K-fold CV splits
            n_trials: Number of Optuna trials
            random_state: for cross-validation reproducibility
            hidden_dim_min, hidden_dim_max: Range for hidden dims in MLP or RNN
            uncertainty_param_min, uncertainty_param_max: Range for uncertainty hyperparam
        """
        # Validate model_type
        valid_model_types = ["mve", "deep_evidential", "nn_ensemble", "mc_dropout"]
        if model_type not in valid_model_types:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of {valid_model_types}")
            
        self.model_type = model_type
        self.input_dim = input_dim
        self.n_objectives = n_objectives
        self.network_type = network_type
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state

        self.hidden_dim_min = hidden_dim_min
        self.hidden_dim_max = hidden_dim_max

        self.uncertainty_param_min = uncertainty_param_min
        self.uncertainty_param_max = uncertainty_param_max

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
                n_objectives=self.n_objectives,
            )
        elif self.model_type == "deep_evidential":
            return DeepEvidentialRegression(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                evidential_regularization=param_value,
                n_objectives=self.n_objectives,
            )
        elif self.model_type == "mc_dropout":
            return MonteCarloDropout(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                dropout_rate=param_value,
                n_objectives=self.n_objectives,
            )
        elif self.model_type == "nn_ensemble":
            return NeuralNetworkEnsemble(
                input_dim=self.input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=self.network_type,
                n_networks=int(param_value),
                n_objectives=self.n_objectives,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _evaluate_model(
        self, model: "BasePredictionModel", val_loader: DataLoader
    ) -> float:
        """
        Returns NLL. For multi-objective, returns sum of NLLs across objectives.
        """
        model.eval()
        model.to(self.device)

        all_means, all_stds, all_targets = [], [], []

        with torch.no_grad():
            for batch_x, batch_y, lengths in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Handle target reshaping based on objectives
                if self.n_objectives == 1:
                    # Single objective: ensure shape [batch_size, 1]
                    if batch_y.dim() == 1:
                        batch_y = batch_y.view(-1, 1)
                else:
                    # Multi-objective: ensure shape [batch_size, n_objectives]
                    if batch_y.dim() == 1:
                        # This shouldn't happen for multi-objective, but handle gracefully
                        batch_y = batch_y.view(-1, 1)

                means, stds = model.forward_predict(batch_x, lengths=lengths)
                all_means.append(means)
                all_stds.append(stds)
                all_targets.append(batch_y)

        means = torch.cat(all_means)
        stds = torch.cat(all_stds)
        targets = torch.cat(all_targets)

        if self.n_objectives == 1:
            return self._nll_gaussian(means, stds, targets)
        else:
            return self._nll_multi_gaussian(means, stds, targets)

    
    @staticmethod
    def _nll_gaussian(means: torch.Tensor,
                    stds:  torch.Tensor,
                    targets: torch.Tensor) -> float:
        """
        Returns the mean negative log‐likelihood under N(means, stds^2).
        NLL = 0.5 * [((y-μ)^2/σ^2) + log(σ^2) + log(2π)].
        """
        stds = torch.clamp(stds, min=1e-6)
        var  = stds**2
        nll = 0.5 * ( (targets - means)**2 / var
                    + torch.log(var)
                    + math.log(2 * math.pi)
                )

        return nll.mean().item()
    
    @staticmethod
    def _nll_multi_gaussian(means: torch.Tensor,
                          stds: torch.Tensor,
                          targets: torch.Tensor) -> float:
        """
        Returns the sum of mean negative log-likelihoods across objectives.
        For multi-objective case where:
        - means: [batch_size, n_objectives] or [batch_size, n_objectives, 1]
        - stds: [batch_size, n_objectives] or [batch_size, n_objectives, 1]  
        - targets: [batch_size, n_objectives]
        """
        # Ensure consistent shapes
        if means.dim() == 3:
            means = means.squeeze(-1)  # [batch_size, n_objectives]
        if stds.dim() == 3:
            stds = stds.squeeze(-1)   # [batch_size, n_objectives]
        if targets.dim() == 3:
            targets = targets.squeeze(-1)  # [batch_size, n_objectives]
        
        stds = torch.clamp(stds, min=1e-6)
        var = stds**2
        nll = 0.5 * ( (targets - means)**2 / var
                    + torch.log(var)
                    + math.log(2 * math.pi)
                )
        
        # Sum NLL across objectives, then take mean across batch
        return nll.sum(dim=-1).mean().item()
    

    def _fit_model(
        self,
        model: "BasePredictionModel",
        train_embed_dict: Dict[str, np.ndarray],
        train_objective_dict: Dict[str, float],
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
            objective_dict=train_objective_dict,
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
            # Ensure parameters are floats
            min_param = 0.0 if self.uncertainty_param_min is None else float(self.uncertainty_param_min)
            max_param = 1.0 if self.uncertainty_param_max is None else float(self.uncertainty_param_max)
            param_value = trial.suggest_float("uncertainty_param", min_param, max_param)
        elif self.model_type == "mc_dropout":
            # Ensure parameters are floats
            min_param = 0.01 if self.uncertainty_param_min is None else float(self.uncertainty_param_min)
            max_param = 0.8 if self.uncertainty_param_max is None else float(self.uncertainty_param_max)
            param_value = trial.suggest_float("uncertainty_param", min_param, max_param)
        elif self.model_type == "nn_ensemble":
            # Ensure parameters are integers
            min_param = 2 if self.uncertainty_param_min is None else int(self.uncertainty_param_min)
            max_param = 10 if self.uncertainty_param_max is None else int(self.uncertainty_param_max)
            param_value = trial.suggest_int("uncertainty_param", min_param, max_param)
        else:
            raise ValueError(f"Unknown model_type {self.model_type}")

        keys = list(self.embedding_dict.keys())
        indices = np.arange(len(keys))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        cv_scores = []
        for train_idx, val_idx in kf.split(indices):
            train_embed_dict = {keys[i]: self.embedding_dict[keys[i]] for i in train_idx}
            train_objective_dict = {keys[i]: self.objective_dict[keys[i]] for i in train_idx}

            val_embed_dict = {keys[i]: self.embedding_dict[keys[i]] for i in val_idx}
            val_objective_dict = {keys[i]: self.objective_dict[keys[i]] for i in val_idx}

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
                train_objective_dict,
                epochs,
                learning_rate,
                batch_size=batch_size,
            )

            val_dataset = VariableLengthDataset(val_embed_dict, val_objective_dict)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=variable_length_collate_fn,
            )

            nlls = self._evaluate_model(model, val_loader)
            cv_scores.append(nlls)

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
        objective_dict: Dict[str, Union[float, List[float], np.ndarray]],
        previous_study: Optional[optuna.study.Study] = None,
    ) -> Tuple[Dict[str, Union[float, List[int], None]], optuna.study.Study]:
        """
        Main entry point: run the Optuna study for cross-validation 
        and store best hyperparams in self.best_params.
        
        Args:
            embedding_dict: Dictionary of embeddings
            objective_dict: Dictionary of scores
            previous_study: Optional previous Optuna study to warm-start from
            
        Returns:
            Tuple of (best_params, study) where study can be reused in future calls
        """
        self.embedding_dict = embedding_dict
        self.objective_dict = objective_dict

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
    objective_dict: Dict[str, Union[float, List[float], np.ndarray]],
    network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
    n_splits: int = 3,
    n_trials: int = 20,
    random_state: int = 42,
    device: Optional[torch.device] = None,
    hidden_dim_min: int = 16,
    hidden_dim_max: int = 256,
    previous_study: Optional[optuna.study.Study] = None,
    uncertainty_param_min: Optional[Union[float, int]] = None,
    uncertainty_param_max: Optional[Union[float, int]] = None,
) -> Tuple[Dict[str, Union[float, List[int], None]], optuna.study.Study]:
    """
    High-level API for tuning architecture + uncertainty hyperparams with 
    MLP/BiLSTM/BiGRU options, supporting both single and multi-objective optimization.
    
    Args:
        objective_dict: Can contain single values (float) for single-objective,
                       or lists/arrays for multi-objective optimization
    
    Returns:
        Tuple of (best_params, study) where study can be reused in future calls
    """

    sample_embedding = next(iter(embedding_dict.values()))
    if sample_embedding.ndim == 2:
        input_dim = sample_embedding.shape[1]
    else:
        input_dim = sample_embedding.shape[0]
    
    # Auto-detect number of objectives
    sample_objective = next(iter(objective_dict.values()))
    if isinstance(sample_objective, (list, np.ndarray)):
        n_objectives = len(sample_objective)
    else:
        n_objectives = 1
    
    tuner = HyperparameterTuner(
        model_type=model_type,
        input_dim=input_dim,
        n_objectives=n_objectives,
        network_type=network_type,
        device=device,
        n_splits=n_splits,
        n_trials=n_trials,
        random_state=random_state,
        hidden_dim_min=hidden_dim_min,
        hidden_dim_max=hidden_dim_max,
        uncertainty_param_min = uncertainty_param_min,
        uncertainty_param_max = uncertainty_param_max,
    )

    best_params, study = tuner.tune(embedding_dict, objective_dict, previous_study)
    return best_params, study

