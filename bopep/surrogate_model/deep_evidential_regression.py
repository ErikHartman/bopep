import torch
import numpy as np
from typing import Dict, List, Tuple, Literal, Optional
from torch.utils.data import DataLoader

from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.network_factory import NetworkFactory
from bopep.surrogate_model.helpers import VariableLengthDataset, variable_length_collate_fn


class DeepEvidentialRegression(BasePredictionModel):
    """
    Implementation of Deep Evidential Regression.

    Based on the paper: "Deep Evidential Regression" by Amini et al.
    https://arxiv.org/abs/1910.02600

    The model outputs 4 parameters of a Normal-Inverse-Gamma distribution:
    - mu: predicted mean
    - v: observation noise precision (lambda)
    - alpha: shape parameter for the Inverse-Gamma prior
    - beta: scale parameter for the Inverse-Gamma prior

    The uncertainty is decomposed into:
    - Aleatoric uncertainty: beta / (alpha - 1)
    - Epistemic uncertainty: beta / (v * (alpha - 1))
    - Total uncertainty: beta / (v * (alpha - 1)) + beta / (alpha - 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        evidential_regularization: float = 0.1,
        uncertainty_type: Literal["aleatoric", "epistemic", "total"] = "total",
        n_objectives: int = 1,  # Support for multi-objective outputs
        **kwargs
    ):
        super().__init__()
        self.evidential_regularization = evidential_regularization
        self.uncertainty_type = uncertainty_type
        self.n_objectives = n_objectives

        # We need 4 outputs per objective (mu, v, alpha, beta)
        output_dim = 4

        self.network = NetworkFactory.get_network(
            network_type=network_type,
            input_dim=input_dim,
            output_dim=output_dim,
            n_objectives=n_objectives,
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )

    def forward_once(
        self, 
        x: torch.Tensor, 
        lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.network(x, lengths=lengths)
        
        if self.n_objectives == 1:
            # Original behavior: outputs shape is (N, 4)
            mu = outputs[:, 0:1]
            v = torch.nn.functional.softplus(outputs[:, 1:2]) + 1e-3  
            alpha = torch.nn.functional.softplus(outputs[:, 2:3]) + 1.0 
            beta = torch.nn.functional.softplus(outputs[:, 3:4]) + 1e-3 
        else:
            # Multi-objective: outputs shape is (N, n_objectives, 4)
            mu = outputs[:, :, 0:1]  # (N, n_objectives, 1)
            v = torch.nn.functional.softplus(outputs[:, :, 1:2]) + 1e-3  
            alpha = torch.nn.functional.softplus(outputs[:, :, 2:3]) + 1.0 
            beta = torch.nn.functional.softplus(outputs[:, :, 3:4]) + 1e-3 
            
        return mu, v, alpha, beta

    def forward_predict(
        self, 
        x: torch.Tensor, 
        lengths: Optional[List[int]] = None,
        uncertainty_mode: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictive mean and standard deviation.
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths for variable-length inputs
            uncertainty_mode: Which uncertainty component to use ("aleatoric", "epistemic", or "total")
                              If None, uses the default set in __init__
                              
        Returns:
        - For single objective (n_objectives=1): mean (N, 1), std (N, 1)
        - For multi-objective (n_objectives>1): mean (N, n_objectives), std (N, n_objectives)
        """
        if uncertainty_mode is None:
            uncertainty_mode = self.uncertainty_type
            
        mu, v, alpha, beta = self.forward_once(x, lengths=lengths)
        
        # Calculate different uncertainty components
        aleatoric_variance = beta / (alpha - 1.0)
        epistemic_variance = beta / (v * (alpha - 1.0))
        
        if uncertainty_mode == "aleatoric":
            variance = aleatoric_variance
        elif uncertainty_mode == "epistemic":
            variance = epistemic_variance
        else:
            variance = aleatoric_variance + epistemic_variance
            
        std = torch.sqrt(variance)
        
        # Squeeze the last dimension for consistency with other models
        if self.n_objectives == 1:
            mu = mu.squeeze(-1)  # (N, 1) -> (N,) -> (N, 1) after unsqueeze
            std = std.squeeze(-1)
            mu = mu.unsqueeze(-1)
            std = std.unsqueeze(-1)
        else:
            mu = mu.squeeze(-1)  # (N, n_objectives, 1) -> (N, n_objectives)
            std = std.squeeze(-1)
            
        return mu, std

    def get_uncertainty_components(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty into aleatoric and epistemic components.
        Returns dict with keys:
            - 'mean': predicted mean
            - 'aleatoric': aleatoric uncertainty (data uncertainty)
            - 'epistemic': epistemic uncertainty (model uncertainty)
            - 'total': total uncertainty
        """
        mu, v, alpha, beta = self.forward_once(x, lengths=lengths)
        aleatoric_uncertainty = beta / (alpha - 1.0)
        epistemic_uncertainty = beta / (v * (alpha - 1.0))
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return {
            "mean": mu,
            "aleatoric": aleatoric_uncertainty,
            "epistemic": epistemic_uncertainty,
            "total": total_uncertainty,
            "v": v,
            "alpha": alpha,
            "beta": beta,
        }

    def predict_dict_with_components(
        self, 
        embedding_dict: Dict[str, np.ndarray],
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict with uncertainty decomposition for a dictionary of embeddings.
        Returns {sequence: {'mean': val, 'aleatoric': val, 'epistemic': val, 'total': val}}
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Process in batches to handle large dictionaries
        sequences = list(embedding_dict.keys())
        
        # Build a dataset without scores
        dummy_scores = {p: 0.0 for p in sequences}
        dataset = VariableLengthDataset(embedding_dict, dummy_scores)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=variable_length_collate_fn,
        )
        
        self.eval()
        components_list = []
        
        with torch.no_grad():
            for batch_x, _, lengths in dataloader:
                # Move batch to device
                batch_x = batch_x.to(device)
                
                # Get components
                batch_components = self.get_uncertainty_components(batch_x, lengths)
                
                # Move back to CPU
                batch_components = {k: v.cpu() for k, v in batch_components.items()}
                components_list.append(batch_components)
        
        # Combine results from all batches
        components_combined = {}
        for key in components_list[0].keys():
            components_combined[key] = torch.cat([batch[key] for batch in components_list])
        
        # Convert to numpy and reshape
        components_np = {k: v.numpy().reshape(-1) for k, v in components_combined.items()}
        
        # Format results
        result = {}
        for i, p in enumerate(sequences):
            result[p] = {k: float(v[i]) for k, v in components_np.items()}
            
        return result

    def evidential_loss(
        self,
        mu: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the evidential regression loss with regularization as described in the paper.
        
        The loss consists of two terms:
        1. NLL term: Negative log likelihood of the data under the predictive distribution
        2. Regularization term: KL divergence between the predicted and prior distribution
        
        Args:
            mu: Predicted mean - shape (N, n_objectives, 1) or (N, 1)
            v: Predicted precision parameter - same shape as mu
            alpha: Predicted shape parameter - same shape as mu  
            beta: Predicted scale parameter - same shape as mu
            targets: Ground truth targets - shape (N, n_objectives) or (N, 1)
        """
        # Handle both single and multi-objective cases
        if self.n_objectives > 1:
            # Multi-objective: mu has shape (N, n_objectives, 1), targets (N, n_objectives)
            # Squeeze the last dimension from predictions
            mu = mu.squeeze(-1)  # (N, n_objectives)
            v = v.squeeze(-1)
            alpha = alpha.squeeze(-1)
            beta = beta.squeeze(-1)
            
            # Ensure targets has the right shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)  # (N,) -> (N, 1)
            if targets.shape[1] == 1 and self.n_objectives > 1:
                # Broadcast single target to all objectives (assumes same target for all)
                targets = targets.expand(-1, self.n_objectives)  # (N, 1) -> (N, n_objectives)
        else:
            # Single objective: ensure consistent shapes
            mu = mu.squeeze(-1)  # (N, 1) -> (N,)
            v = v.squeeze(-1)
            alpha = alpha.squeeze(-1) 
            beta = beta.squeeze(-1)
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets
            
            # Add dimension back for computation
            mu = mu.unsqueeze(-1)  # (N,) -> (N, 1)
            v = v.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            targets = targets.unsqueeze(-1)
        
        twoBlambda = 2.0 * beta * (1.0 + v)

        nll = (
            0.5 * torch.log(np.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        error = (targets - mu)
        reg = torch.abs(error) * (2.0 * v + alpha)

        loss = nll + self.evidential_regularization * reg

        # For multi-objective, average losses across objectives, then average across batch
        if self.n_objectives > 1:
            loss = loss.mean(dim=-1)  # Mean across objectives
        
        return loss.mean()  # Average across batch

    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        mu, v, alpha, beta = self.forward_once(batch_x, lengths)
        return self.evidential_loss(mu, v, alpha, beta, batch_y)
    
    def _get_default_criterion(self):
        return None
