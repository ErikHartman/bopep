"""
Multi-objective wrapper for surrogate models.

This module provides a wrapper that manages multiple single-objective models
for multi-objective optimization, integrating seamlessly with the existing
model factory system.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging

from bopep.surrogate_model.helpers import BasePredictionModel


class MultiObjectiveWrapper(BasePredictionModel):
    """
    A wrapper that manages multiple single-objective models for multi-objective optimization.
    
    This class acts like a single model but internally trains separate models for each
    objective, which can help avoid gradient conflicts in multi-objective scenarios.
    """
    
    def __init__(
        self,
        model_class: type,
        input_dim: int,
        n_objectives: int,
        network_type: str = 'mlp',
        hidden_dims: Optional[List[int]] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        **model_kwargs
    ):
        """
        Initialize the multi-objective wrapper.
        
        Args:
            model_class: The model class to instantiate for each objective
            input_dim: Dimensionality of input embeddings
            n_objectives: Number of objectives
            network_type: Network architecture type
            hidden_dims: List of hidden dimensions for each layer
            hidden_dim: Single hidden dimension (if hidden_dims not provided)
            num_layers: Number of hidden layers
            **model_kwargs: Additional arguments passed to each model
        """
        super().__init__()
        
        self.model_class = model_class
        self.input_dim = input_dim
        self.n_objectives = n_objectives
        self.network_type = network_type
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_kwargs = model_kwargs
        
        # Create individual models for each objective (each with n_objectives=1)
        self.models = torch.nn.ModuleList([
            model_class(
                input_dim=input_dim,
                n_objectives=1,  # Each model handles one objective
                network_type=network_type,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                **model_kwargs
            )
            for _ in range(n_objectives)
        ])
        
        # Store objective names when we see them during training
        self.objective_names = None
        
        logging.info(f"Initialized MultiObjectiveWrapper with {n_objectives} {model_class.__name__} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all individual models."""
        # Get predictions from each model and concatenate
        outputs = []
        for model in self.models:
            output = model(x)  # Each model outputs for 1 objective
            outputs.append(output)
        
        # Concatenate along the objective dimension
        return torch.cat(outputs, dim=-1)
    
    def fit_dict(
        self,
        embedding_dict: Dict[str, np.ndarray],
        objective_dict: Dict[str, Union[Dict[str, float], float]],
        val_embedding_dict: Optional[Dict[str, np.ndarray]] = None,
        val_objective_dict: Optional[Dict[str, Union[Dict[str, float], float]]] = None,
        **kwargs
    ) -> float:
        """
        Train all individual models.
        
        Args:
            embedding_dict: {peptide: embedding_array}
            objective_dict: {peptide: {obj_name: value}} or {peptide: value} for single objective
            val_embedding_dict: Optional validation embeddings
            val_objective_dict: Optional validation objectives
            **kwargs: Training arguments
            
        Returns:
            Average training loss across all models
        """
        # Handle both multi-objective and single-objective formats
        if isinstance(next(iter(objective_dict.values())), dict):
            # Multi-objective format: {peptide: {obj_name: value}}
            objective_names = sorted(set().union(*[obj_dict.keys() for obj_dict in objective_dict.values()]))
            
            if len(objective_names) != self.n_objectives:
                raise ValueError(f"Expected {self.n_objectives} objectives, got {len(objective_names)}")
                
            # Store objective names for later use in prediction
            self.objective_names = objective_names
        else:
            # Single objective format: {peptide: value} - shouldn't happen with this wrapper
            raise ValueError("MultiObjectiveWrapper expects multi-objective format: {peptide: {obj_name: value}}")
        
        losses = []
        
        # Train each model on its corresponding objective
        for i, (obj_name, model) in enumerate(zip(objective_names, self.models)):
            # Extract single-objective data
            single_obj_dict = {
                peptide: obj_values[obj_name] 
                for peptide, obj_values in objective_dict.items()
                if obj_name in obj_values
            }
            
            # Filter embeddings to match
            single_emb_dict = {
                peptide: embedding_dict[peptide] 
                for peptide in single_obj_dict.keys()
            }
            
            # Prepare validation data if provided
            val_single_emb_dict = None
            val_single_obj_dict = None
            if val_embedding_dict is not None and val_objective_dict is not None:
                val_single_obj_dict = {
                    peptide: obj_values[obj_name] 
                    for peptide, obj_values in val_objective_dict.items()
                    if obj_name in obj_values
                }
                val_single_emb_dict = {
                    peptide: val_embedding_dict[peptide] 
                    for peptide in val_single_obj_dict.keys()
                }
            
            # Train the individual model
            loss = model.fit_dict(
                embedding_dict=single_emb_dict,
                objective_dict=single_obj_dict,
                val_embedding_dict=val_single_emb_dict,
                val_objective_dict=val_single_obj_dict,
                **kwargs
            )
            
            losses.append(loss)
            logging.info(f"Training complete for objective {i+1}/{self.n_objectives} ('{obj_name}') with loss: {loss:.4f}")
        
        # Return average loss
        avg_loss = sum(losses) / len(losses)
        return avg_loss
    
    def predict_dict(
        self, 
        embedding_dict: Dict[str, np.ndarray],
        **kwargs
    ) -> Dict[str, Union[tuple, Dict[str, tuple]]]:
        """
        Make predictions using all individual models.
        
        Args:
            embedding_dict: {peptide: embedding_array}
            **kwargs: Additional prediction arguments
            
        Returns:
            For multi-objective: {peptide: {obj_name: (mean, std)}}
        """
        # Use stored objective names if available, otherwise use generic names
        if self.objective_names:
            objective_names = self.objective_names
        else:
            objective_names = [f"obj{i+1}" for i in range(self.n_objectives)]
        
        results = {}
        
        # Get predictions from each model
        model_predictions = []
        for model in self.models:
            pred = model.predict_dict(embedding_dict, **kwargs)
            model_predictions.append(pred)
        
        # Combine predictions into multi-objective format
        for peptide in embedding_dict.keys():
            results[peptide] = {}
            for i, obj_name in enumerate(objective_names):
                if peptide in model_predictions[i]:
                    results[peptide][obj_name] = model_predictions[i][peptide]
                else:
                    # Fallback
                    results[peptide][obj_name] = (0.0, 1.0)
        
        return results
    
    def to(self, device: Union[str, torch.device]) -> 'MultiObjectiveWrapper':
        """Move all models to the specified device."""
        super().to(device)
        for model in self.models:
            model.to(device)
        return self
    
    def train(self, mode: bool = True) -> 'MultiObjectiveWrapper':
        """Set training mode for all models."""
        super().train(mode)
        for model in self.models:
            model.train(mode)
        return self
    
    def eval(self) -> 'MultiObjectiveWrapper':
        """Set evaluation mode for all models."""
        super().eval()
        for model in self.models:
            model.eval()
        return self
    
    def __repr__(self) -> str:
        return (f"MultiObjectiveWrapper("
                f"model_class={self.model_class.__name__}, "
                f"n_objectives={self.n_objectives}, "
                f"n_models={len(self.models)})")