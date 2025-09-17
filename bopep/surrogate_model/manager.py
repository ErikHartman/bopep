import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging
from sklearn.metrics import r2_score, mean_absolute_error
from bopep.surrogate_model import (
    tune_hyperparams,
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    MVE
)

class SurrogateModelManager:
    """
    Manages surrogate model operations including hyperparameter optimization,
    model initialization, training, and prediction. Used by both BoPep and BoGA.
    """
    
    def __init__(self, surrogate_model_kwargs: Dict[str, Any], device: Optional[str] = None):
        """
        Initialize the surrogate model manager.
        """
        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.best_hyperparams = None
        self.previous_study = None
        
    def optimize_hyperparameters(
        self, 
        embeddings: Dict[str, Any], 
        objectives: Dict[str, float],
        n_trials: int = 20,
        n_splits: int = 3,
        random_state: int = 42,
        iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Returns the best hyperparameters found.
        """
        model_type = self.surrogate_model_kwargs['model_type']
        network_type = self.surrogate_model_kwargs['network_type']
        
        if iteration is not None:
            logging.info(f"Starting hyperparameter optimization for {network_type} {model_type} model...")
        
        if self.previous_study is not None:
            logging.info(f"Using previous study with {len(self.previous_study.trials)} trials")
        
        self.best_hyperparams, self.previous_study = tune_hyperparams(
            model_type=model_type,
            embedding_dict=embeddings,
            objective_dict=objectives,
            network_type=network_type,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            previous_study=self.previous_study,
            device=self.device
        )
        
        if iteration is not None:
            logging.info(f"Hyperparameter optimization complete. Best parameters: {self.best_hyperparams}")
        
        return self.best_hyperparams
    
    def initialize_model(self, hyperparams: Optional[Dict[str, Any]] = None, embeddings: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the surrogate model with given hyperparameters.

        Sets self.model to the initialized model instance.
        """
        if hyperparams is None:
            hyperparams = self.best_hyperparams
            if hyperparams is None:
                raise ValueError("No hyperparameters available. Run optimize_hyperparameters first.")
        
        # Clean up previous model
        self.cleanup_model()
        
        model_type = self.surrogate_model_kwargs['model_type']
        network_type = self.surrogate_model_kwargs['network_type']
        
        # Determine input_dim
        if 'input_dim' in self.surrogate_model_kwargs:
            input_dim = self.surrogate_model_kwargs['input_dim']
        elif embeddings:
            sample_embedding = next(iter(embeddings.values()))
            if hasattr(sample_embedding, 'ndim'):
                if sample_embedding.ndim == 2:
                    input_dim = sample_embedding.shape[1]
                else:
                    input_dim = sample_embedding.shape[0]
            else:
                input_dim = len(sample_embedding)
        else:
            raise ValueError("Cannot determine input_dim. Provide embeddings or set in surrogate_model_kwargs.")
        
        # Extract hyperparameters
        hidden_dims = hyperparams.get('hidden_dims')
        hidden_dim = hyperparams.get('hidden_dim')
        num_layers = hyperparams.get('num_layers', 2)
        uncertainty_param = hyperparams.get('uncertainty_param')
        
        # Initialize model based on type
        if model_type == 'mve':
            self.model = MVE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                mve_regularization=uncertainty_param
            )
        elif model_type == 'deep_evidential':
            self.model = DeepEvidentialRegression(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                evidential_regularization=uncertainty_param
            )
        elif model_type == 'mc_dropout':
            self.model = MonteCarloDropout(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                dropout_rate=uncertainty_param
            )
        elif model_type == 'nn_ensemble':
            self.model = NeuralNetworkEnsemble(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                n_networks=int(uncertainty_param)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        logging.info(f"Initialized {model_type} model with {network_type} network architecture")
    
    def train(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> None:
        """
        Train the model on all provided data.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")
        
        self.model.fit_dict(embeddings, objectives, device=self.device)
    
    def predict(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions on the provided embeddings.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model first.")
        
        return self.model.predict_dict(embeddings, device=self.device)
    
    def cleanup_model(self) -> None:
        """Clean up the current model and free memory."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.cpu()
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Couldn't clean up previous model: {e}")
            finally:
                self.model = None
    
    
    def train_with_validation_split(
        self, 
        embeddings: dict, 
        objectives: dict, 
        validation_size: int = 5,
        min_training_samples: int = 10,
        min_validation_samples: int = 3
    ) -> Tuple[float, dict]:
        """
        Train model with automatic train/validation split.
        """
        if not self.model:
            raise RuntimeError("Model must be initialized before training")
            
        total_samples = len(embeddings)
        
        # Determine if we should split
        num_validate = self._compute_split_indices(
            total_samples, validation_size, min_training_samples, min_validation_samples
        )
        
        if num_validate is None:
            # Train on all data
            return self._train_on_all(embeddings, objectives) # returns loss, metrics
        else:
            # Split and train with validation
            train_emb, train_obj, val_emb, val_obj = self._split_train_validation(
                embeddings, objectives, num_validate
            )
            return self._train_and_validate(train_emb, train_obj, val_emb, val_obj) # returns loss, metrics

    def _train_and_validate(self, train_emb: dict, train_obj: dict, val_emb: dict, val_obj: dict) -> Tuple[float, dict]:
        """Train on train set, evaluate on both splits."""
        if not self.best_hyperparams:
            raise RuntimeError("Hyperparameters must be optimized before training")
            
        loss = self.model.fit_dict(
            embedding_dict=train_emb,
            objective_dict=train_obj,
            val_embedding_dict=val_emb,
            val_objective_dict=val_obj,
            epochs=self.best_hyperparams.get("epochs", 100),
            learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
            batch_size=self.best_hyperparams.get("batch_size", 16),
            device=self.device,
        )
        
        train_pred = self.model.predict_dict(train_emb, device=self.device)
        val_pred = self.model.predict_dict(val_emb, device=self.device)
        train_m = self._compute_model_metrics(train_pred, train_obj)
        val_m = self._compute_model_metrics(val_pred, val_obj)

        metrics = {
            "train_r2": train_m["r2"],
            "train_mae": train_m["mae"],
            "val_r2": val_m["r2"],
            "val_mae": val_m["mae"],
        }
        
        logging.info(
            f"Loss {loss:.4f}, train R2 {train_m['r2']:.4f}, "
            f"val R2 {val_m['r2']:.4f} "
            f"(N_train={len(train_emb)}, N_val={len(val_emb)})"
        )
        return loss, metrics

    def _train_on_all(self, embeddings: dict, objectives: dict) -> Tuple[float, dict]:
        """Train on the entire dataset (no validation)."""
        if not self.best_hyperparams:
            raise RuntimeError("Hyperparameters must be optimized before training")
            
        loss = self.model.fit_dict(
            embedding_dict=embeddings,
            objective_dict=objectives,
            epochs=self.best_hyperparams.get("epochs", 100),
            learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
            batch_size=self.best_hyperparams.get("batch_size", 16),
            device=self.device,
        )
        
        preds = self.model.predict_dict(embeddings, device=self.device)
        m = self._compute_model_metrics(preds, objectives)
        metrics = {"r2": m["r2"], "mae": m["mae"]}
        
        logging.info(f"Loss {loss:.4f}, R2 {m['r2']:.4f}, N={len(embeddings)}")
        return loss, metrics

    def _calculate_validation_metrics(
        self, 
        predictions: Dict[str, Any], 
        actual: Dict[str, float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate validation metrics comparing predictions to actual values.
        """
        if not predictions or not actual:
            return 0.0, {}
        
        # Find common sequences
        common_seqs = set(predictions.keys()) & set(actual.keys())
        if not common_seqs:
            return 0.0, {}
        
        # Extract values
        pred_values = []
        actual_values = []
        
        for seq in common_seqs:
            pred = predictions[seq]
            # Handle uncertainty models that return (mean, std) tuples
            if isinstance(pred, tuple):
                pred_values.append(pred[0])
            else:
                pred_values.append(pred)
            actual_values.append(actual[seq])
        
        pred_values = np.array(pred_values)
        actual_values = np.array(actual_values)
        
        # Calculate metrics
        mse = np.mean((pred_values - actual_values) ** 2)
        mae = np.mean(np.abs(pred_values - actual_values))
        
        # Correlation coefficient (if variance exists)
        if np.var(pred_values) > 0 and np.var(actual_values) > 0:
            correlation = np.corrcoef(pred_values, actual_values)[0, 1]
        else:
            correlation = 0.0
        
        # R² score
        ss_res = np.sum((actual_values - pred_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'correlation': float(correlation),
            'r2': float(r2),
            'n_samples': len(common_seqs)
        }
        
        return float(mse), metrics
    
    def _compute_model_metrics(self, predictions_dict: dict, objectives: dict) -> dict:
        """Compute R2 and MAE metrics from predictions and objectives."""
        peptides = list(predictions_dict.keys())
        actual = np.array([objectives[p] for p in peptides])
        predicted = np.array([predictions_dict[p][0] for p in peptides])
        
        
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        return {"r2": r2, "mae": mae}

    def _compute_split_indices(
        self, total_samples: int, n_validate: Optional[Union[int, float]], min_training_samples: int = 10, 
        min_validation_samples: int = 3
    ) -> Optional[int]:
        """Return num_validate or None if split is infeasible."""
        if n_validate is None:
            return None
            
        if isinstance(n_validate, float) and n_validate < 1:
            num = int(total_samples * n_validate)
        else:
            num = n_validate

        if (
            num < min_validation_samples
            or (total_samples - num) < min_training_samples
        ):
            logging.warning(
                f"Cannot split {total_samples} samples into "
                f"{min_training_samples} train + "
                f"{min_validation_samples} val; training on all."
            )
            return None

        return num

    def _split_train_validation(
        self, embeddings: dict, objectives: dict, num_validate: int
    ) -> Tuple[dict, dict, dict, dict]:
        """Split the available data into training and validation sets."""
        peptides = list(objectives.keys())
        val_indices = np.random.choice(len(peptides), num_validate, replace=False)
        val_peptides = [peptides[i] for i in val_indices]
        train_peptides = [p for p in peptides if p not in val_peptides]
        
        train_embeddings = {p: embeddings[p] for p in train_peptides}
        train_objectives = {p: objectives[p] for p in train_peptides}
        val_embeddings = {p: embeddings[p] for p in val_peptides}
        val_objectives = {p: objectives[p] for p in val_peptides}

        logging.info(
            f"Split data into {len(train_peptides)} training and {len(val_peptides)} validation samples"
        )

        return train_embeddings, train_objectives, val_embeddings, val_objectives
