import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging
from sklearn.metrics import r2_score
from bopep.surrogate_model import (
    tune_hyperparams,
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    MVE
)
from bopep.surrogate_model.multi_model import MultiModelWrapper
from bopep.surrogate_model.helpers import ObjectiveMixin

class SurrogateModelManager(ObjectiveMixin):
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
        objectives: Dict[str, Union[float, Dict[str, float]]],
        n_trials: int = 20,
        n_splits: int = 3,
        random_state: int = 42,
        iteration: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        model_type = self.surrogate_model_kwargs['model_type']
        network_type = self.surrogate_model_kwargs['network_type']
        
        if iteration is not None:
            logging.info(f"Starting hyperparameter optimization for {network_type} {model_type} model...")
        
        if self.previous_study is not None:
            logging.info(f"Using previous study with {len(self.previous_study.trials)} trials")
        
        max_seq_len = self.surrogate_model_kwargs.get('max_seq_len', 150)
        
        self.best_hyperparams, self.previous_study = tune_hyperparams(
            model_type=model_type,
            embedding_dict=embeddings,
            objective_dict=objectives,
            network_type=network_type,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            previous_study=self.previous_study,
            device=self.device,
            max_seq_len=max_seq_len
        )
        
        if iteration is not None:
            logging.info(f"Hyperparameter optimization complete. Best parameters: {self.best_hyperparams}")
        
        return self.best_hyperparams


    def initialize_model(self, hyperparams: Optional[Dict[str, Any]] = None, embeddings: Optional[Dict[str, Any]] = None, objectives: Optional[Dict[str, Union[float, Dict[str, float]]]] = None) -> None:
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
        
        # Auto-detect or get n_objectives parameter
        if objectives:
            self._setup_objectives(objectives)
            n_objectives = self._n_objectives
        else:
            n_objectives = self.surrogate_model_kwargs.get('n_objectives', 1)

        # Extract hyperparameters
        hidden_dims = hyperparams.get('hidden_dims')
        hidden_dim = hyperparams.get('hidden_dim')
        num_layers = hyperparams.get('num_layers', 2)
        uncertainty_param = hyperparams.get('uncertainty_param')
        
        # Check if multi-model approach is requested
        use_multi_model = self.surrogate_model_kwargs.get('multi_model', False)
        
        # Map model type to class
        model_classes = {
            'mve': MVE,
            'deep_evidential': DeepEvidentialRegression,
            'mc_dropout': MonteCarloDropout,
            'nn_ensemble': NeuralNetworkEnsemble
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = model_classes[model_type]
        
        # Prepare model kwargs based on model type
        model_kwargs = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'network_type': network_type,
            'n_objectives': n_objectives,
            'max_seq_len': self.surrogate_model_kwargs.get('max_seq_len', 150)
        }
        
        # Add model-specific parameters
        if model_type == 'mve':
            model_kwargs['mve_regularization'] = uncertainty_param
        elif model_type == 'deep_evidential':
            model_kwargs['evidential_regularization'] = uncertainty_param
        elif model_type == 'mc_dropout':
            model_kwargs['dropout_rate'] = uncertainty_param
        elif model_type == 'nn_ensemble':
            model_kwargs['n_networks'] = int(uncertainty_param)
        
        # Initialize model - either wrapped or direct
        if use_multi_model and n_objectives > 1:
            # Use MultiModelWrapper for multi-objective optimization
            self.model = MultiModelWrapper(
                model_class=model_class,
                **model_kwargs
            )
            wrapper_info = " (multi-model wrapper)"
        else:
            # Use single model directly
            self.model = model_class(**model_kwargs)
            wrapper_info = ""
        
        self.model.to(self.device)
        obj_info = f" ({n_objectives} objectives)" if n_objectives > 1 else ""
        logging.info(f"Initialized {model_type} model with {network_type} network architecture{obj_info}{wrapper_info}")
    
    def train(self, embeddings: Dict[str, Any], objectives: Dict[str, Union[float, Dict[str, float]]]) -> None:
        """Train the model on all provided data."""
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
    ) -> Dict[str, Any]:
        """Train model with automatic train/validation split."""
        if not self.model:
            raise RuntimeError("Model must be initialized before training")
        
        # Basic validation
        self._validate_multi_objective_consistency(objectives)
            
        total_samples = len(embeddings)
        
        # Determine if we should split
        num_validate = self._compute_split_indices(
            total_samples, validation_size, min_training_samples, min_validation_samples
        )
        
        if num_validate is None:
            # Train on all data
            return self._train_on_all(embeddings, objectives)
        else:
            # Split and train with validation
            train_emb, train_obj, val_emb, val_obj = self._split_train_validation(
                embeddings, objectives, num_validate
            )
            return self._train_and_validate(train_emb, train_obj, val_emb, val_obj)

    def _train_and_validate(self, train_emb: dict, train_obj: dict, val_emb: dict, val_obj: dict) -> Dict[str, Any]:
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
        
        # Compute metrics for both train and validation
        train_metrics = self._compute_metrics(train_pred, train_obj)
        val_metrics = self._compute_metrics(val_pred, val_obj)

        # Combine into unified metrics dict with train_ and val_ prefixes
        metrics = {}
        
        # Add training metrics with train_ prefix
        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value
            
        # Add validation metrics with val_ prefix
        for key, value in val_metrics.items():
            metrics[f"val_{key}"] = value
        
        # Enhanced logging for named objectives
        log_msg = f"Loss {loss:.4f}, train R2 {train_metrics['r2']:.4f}, val R2 {val_metrics['r2']:.4f}"
        
        # Add named objective details if available
        named_objectives = [k for k in train_metrics.keys() if k.startswith('r2_') and k != 'r2']
        if named_objectives:
            obj_names = [k[3:] for k in named_objectives]  # Remove 'r2_' prefix
            train_details = [f"{name}:{train_metrics[f'r2_{name}']:.3f}" for name in obj_names]
            val_details = [f"{name}:{val_metrics[f'r2_{name}']:.3f}" for name in obj_names]
            log_msg += f" (train: {', '.join(train_details)}; val: {', '.join(val_details)})"
        
        log_msg += f" (N_train={len(train_emb)}, N_val={len(val_emb)})"
        logging.info(log_msg)
        
        return metrics

    def _train_on_all(self, embeddings: dict, objectives: dict) -> Dict[str, Any]:
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
        train_metrics = self._compute_metrics(preds, objectives)
        
        # Create unified metrics dict with train_ prefix
        metrics = {}
        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value
        
        # Add validation fields as None to indicate no validation was performed
        metrics["val_r2"] = None
        metrics["val_mae"] = None
        metrics["val_mse"] = None
        
        # Enhanced logging for named objectives
        log_msg = f"Loss {loss:.4f}, R2 {train_metrics['r2']:.4f}"

        named_objectives = [k for k in train_metrics.keys() if k.startswith('r2_') and k != 'r2']
        if named_objectives:
            obj_names = [k[3:] for k in named_objectives]  # Remove 'r2_' prefix
            obj_details = [f"{name}:{train_metrics[f'r2_{name}']:.3f}" for name in obj_names]
            log_msg += f" ({', '.join(obj_details)})"
        
        log_msg += f", N={len(embeddings)}"
        logging.info(log_msg)
        
        return metrics

    def _compute_metrics(
        self, 
        predictions: Dict[str, Any], 
        actual: Dict[str, Union[float, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """Unified method to compute all metrics using ObjectiveMixin logic."""
        if not predictions or not actual:
            return {}
        
        common_seqs = set(predictions.keys()) & set(actual.keys())
        if not common_seqs:
            return {}
        
        # Setup objectives to get structure info
        self._setup_objectives(actual)
        
        metrics = {}
        
        if self._objective_names:
            # Multi-objective case
            all_mses = []
            all_maes = []
            all_r2s = []
            
            for obj_name in self._objective_names:
                pred_values = []
                actual_values = []
                
                for seq in common_seqs:
                    # Extract actual value (mean from tuple if needed)
                    actual_obj = actual[seq][obj_name]
                    if isinstance(actual_obj, tuple):
                        actual_values.append(actual_obj[0])  # mean
                    else:
                        actual_values.append(actual_obj)
                    
                    # Extract predicted value (mean from tuple)
                    pred = predictions[seq]
                    if isinstance(pred, dict):
                        pred_obj = pred[obj_name]
                        if isinstance(pred_obj, tuple):
                            pred_values.append(pred_obj[0])  # mean
                        else:
                            pred_values.append(pred_obj)
                    else:
                        raise ValueError("Prediction format doesn't match named objective format")
                
                pred_array = np.array(pred_values)
                actual_array = np.array(actual_values)
                
                # Calculate metrics for this objective
                mse = np.mean((pred_array - actual_array) ** 2)
                mae = np.mean(np.abs(pred_array - actual_array))
                r2 = r2_score(actual_array, pred_array)
                
                metrics[f'mse_{obj_name}'] = float(mse)
                metrics[f'mae_{obj_name}'] = float(mae)
                metrics[f'r2_{obj_name}'] = float(r2)
                
                all_mses.append(mse)
                all_maes.append(mae)
                all_r2s.append(r2)
            
            # Aggregate metrics
            metrics.update({
                'mse': float(np.mean(all_mses)),
                'mae': float(np.mean(all_maes)),
                'r2': float(np.mean(all_r2s))
            })
            
        else:
            # Single objective case
            pred_values = []
            actual_values = []
            
            for seq in common_seqs:
                # Extract actual value (mean from tuple if needed)
                actual_val = actual[seq]
                if isinstance(actual_val, tuple):
                    actual_values.append(actual_val[0])  # mean
                else:
                    actual_values.append(actual_val)
                
                # Extract predicted value (mean from tuple if needed)
                pred = predictions[seq]
                if isinstance(pred, tuple):
                    pred_values.append(pred[0])  # mean
                else:
                    pred_values.append(pred)
            
            pred_array = np.array(pred_values)
            actual_array = np.array(actual_values)
            
            mse = np.mean((pred_array - actual_array) ** 2)
            mae = np.mean(np.abs(pred_array - actual_array))
            r2 = r2_score(actual_array, pred_array)
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
        
        return metrics
    
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
        sequences = list(objectives.keys())
        val_indices = np.random.choice(len(sequences), num_validate, replace=False)
        val_sequences = [sequences[i] for i in val_indices]
        train_sequences = [p for p in sequences if p not in val_sequences]
        
        train_embeddings = {p: embeddings[p] for p in train_sequences}
        train_objectives = {p: objectives[p] for p in train_sequences}
        val_embeddings = {p: embeddings[p] for p in val_sequences}
        val_objectives = {p: objectives[p] for p in val_sequences}

        logging.info(
            f"Split data into {len(train_sequences)} training and {len(val_sequences)} validation samples"
        )

        return train_embeddings, train_objectives, val_embeddings, val_objectives

    def _validate_multi_objective_consistency(self, objectives: Dict[str, Union[float, Dict[str, float]]]) -> None:
        """Basic validation for multi-objective consistency."""
        if not objectives:
            return
        
        sample_obj = next(iter(objectives.values()))
        is_named_multi_objective = isinstance(sample_obj, dict)
        
        if is_named_multi_objective:
            # Named multi-objective: check all have same objective names
            expected_names = set(sample_obj.keys())
            for sequence, obj_dict in objectives.items():
                if not isinstance(obj_dict, dict):
                    raise ValueError(f"Expected dict of named objectives for sequence '{sequence}', got {type(obj_dict)}")
                if set(obj_dict.keys()) != expected_names:
                    raise ValueError(f"Inconsistent objective names for sequence '{sequence}'")
        else:
            # Single objective: check all are single values or tuples
            for sequence, obj in objectives.items():
                if not isinstance(obj, (int, float, tuple)):
                    raise ValueError(f"Expected single objective (float or tuple) for sequence '{sequence}', got {type(obj)}")
