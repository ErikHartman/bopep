import torch
import numpy as np
from typing import Dict, List, Tuple, Literal, Optional
from torch.utils.data import DataLoader

from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.network_factory import NetworkFactory
from bopep.surrogate_model.helpers import VariableLengthDataset, variable_length_collate_fn


class DeepEvidentialRegression(BasePredictionModel):
    """
    Implementation of Deep Evidential Regression for aleatoric and epistemic uncertainty estimation.

    Based on the paper:
    "Deep Evidential Regression" by Amini et al. (NeurIPS 2020)
    https://arxiv.org/abs/1910.02600

    The model outputs 4 parameters of a Normal-Inverse-Gamma distribution:
    - mu: predicted mean
    - v: observation noise precision (lambda)
    - alpha: shape parameter for the Inverse-Gamma prior
    - beta: scale parameter for the Inverse-Gamma prior

    The uncertainty is decomposed into:
    - Aleatoric uncertainty: beta / (alpha - 1) for alpha > 1
    - Epistemic uncertainty: beta / (v * (alpha - 1)) for alpha > 1
    - Total uncertainty: beta / (v * (alpha - 1)) + beta / (alpha - 1) for alpha > 1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        evidential_regularization: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.evidential_regularization = evidential_regularization

        # We need 4 outputs (mu, v, alpha, beta)
        output_dim = 4

        self.network = NetworkFactory.get_network(
            network_type=network_type,
            input_dim=input_dim,
            output_dim=output_dim,
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
        # Now pass lengths into the network
        outputs = self.network(x, lengths=lengths)

        # Modified parameter constraints to match paper better
        mu = outputs[:, 0:1]
        v = torch.nn.functional.softplus(outputs[:, 1:2]) + 1e-3  
        alpha = torch.nn.functional.softplus(outputs[:, 2:3]) + 1.0 
        beta = torch.nn.functional.softplus(outputs[:, 3:4]) + 1e-3 
        return mu, v, alpha, beta

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictive mean and standard deviation.
        For evidential regression, mean = mu and variance = beta/(alpha-1) * (1 + 1/v)
        """
        mu, v, alpha, beta = self.forward_once(x, lengths=lengths)
        variance = (beta / (alpha - 1.0)) * (1.0 + 1.0 / v)
        std = torch.sqrt(variance)
        return mu, std

    def get_uncertainty_components(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty into aleatoric and epistemic components.

        Args:
            x: Input tensor of shape (N, D) for MLP or (N, L, D) for RNN
            lengths: Optional sequence lengths for variable-length inputs

        Returns:
            Dict with keys:
                - 'mean': predicted mean
                - 'aleatoric': aleatoric uncertainty (data uncertainty)
                - 'epistemic': epistemic uncertainty (model uncertainty)
                - 'total': total uncertainty
        """
        mu, v, alpha, beta = self.forward_once(x, lengths=lengths)

        # Compute uncertainty components
        # Only valid for alpha > 1, which we enforce in forward_once
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
        Returns {peptide: {'mean': val, 'aleatoric': val, 'epistemic': val, 'total': val}}
        
        Args:
            embedding_dict: Dictionary of embeddings
            batch_size: Batch size for prediction
            device: Device to use for computation (defaults to model's device)
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Process in batches to handle large dictionaries
        peptides = list(embedding_dict.keys())
        
        # Build a dataset without scores
        dummy_scores = {p: 0.0 for p in peptides}
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
        for i, p in enumerate(peptides):
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
            mu: Predicted mean
            v: Predicted precision parameter
            alpha: Predicted shape parameter
            beta: Predicted scale parameter
            targets: Ground truth targets
            
        Returns:
            Total loss combining NLL and regularization
        """
        # NLL loss term
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

        # Combine losses with regularization weight
        loss = nll + self.evidential_regularization * reg

        return loss.mean()

    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        Override the default loss calculation to use evidential loss.
        """
        mu, v, alpha, beta = self.forward_once(batch_x, lengths)
        return self.evidential_loss(mu, v, alpha, beta, batch_y)
    
    def _get_default_criterion(self):
        """
        The DeepEvidentialRegression model uses its own loss function, 
        so we return None here to indicate no external criterion is needed.
        """
        return None
