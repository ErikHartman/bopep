import torch
import numpy as np
from typing import Dict, List, Tuple, Literal, Optional
from torch.nn import functional as F

from bopep.surrogate_model.base_models import (
    BasePredictionModel,
    BiLSTMNetwork,
    MLPNetwork,
)


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
        network_type: Literal["mlp", "bilstm"] = "mlp",
        lstm_layers: int = 1,
        lstm_hidden_dim: Optional[int] = None,
        evidential_regularization: float = 0.1,
    ):
        super().__init__()
        self.evidential_regularization = evidential_regularization

        # We need 4 outputs for evidential regression (mu, v, alpha, beta)
        output_dim = 4

        if network_type == "mlp":
            self.network = MLPNetwork(
                input_dim, hidden_dims, output_dim, dropout_rate=0
            )
        elif network_type == "bilstm":
            lstm_hidden = lstm_hidden_dim or hidden_dims[0]
            self.network = BiLSTMNetwork(
                input_dim, lstm_hidden, lstm_layers, output_dim, dropout_rate=0
            )
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")

    def forward_once(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning the 4 evidential parameters."""
        outputs = self.network(x)

        # Split the outputs into the 4 evidential parameters
        mu = outputs[:, 0:1]

        # Apply softplus to ensure v, alpha, beta are positive
        # Adding small epsilon for numerical stability
        v = F.softplus(outputs[:, 1:2]) + 1e-6
        alpha = F.softplus(outputs[:, 2:3]) + 1.0  # alpha > 1 ensures finite variance
        beta = F.softplus(outputs[:, 3:4]) + 1e-6

        return mu, v, alpha, beta

    def forward_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictive mean and standard deviation.
        For evidential regression, mean = mu and variance = beta/(alpha-1) * (1 + 1/v)
        """
        mu, v, alpha, beta = self.forward_once(x)

        # Compute the total predictive uncertainty (epistemic + aleatoric)
        # Formula: variance = beta/(alpha-1) * (1 + 1/v)
        variance = (beta / (alpha - 1.0)) * (1.0 + 1.0 / v)
        std = torch.sqrt(variance)

        return mu, std

    def get_uncertainty_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty into aleatoric and epistemic components.

        Returns:
            Dict with keys:
                - 'mean': predicted mean
                - 'aleatoric': aleatoric uncertainty (data uncertainty)
                - 'epistemic': epistemic uncertainty (model uncertainty)
                - 'total': total uncertainty
        """
        mu, v, alpha, beta = self.forward_once(x)

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
        self, embedding_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict with uncertainty decomposition for a dictionary of embeddings.
        Returns {peptide: {'mean': val, 'aleatoric': val, 'epistemic': val, 'total': val}}
        """
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        X_torch = torch.tensor(X, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            components = self.get_uncertainty_components(X_torch)

        # Convert tensors to numpy arrays
        components_np = {k: v.cpu().numpy().reshape(-1) for k, v in components.items()}

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
        Compute the evidential regression loss with optional regularization.

        The loss consists of two terms:
        1. NLL term: Negative log likelihood of the data under the predictive distribution
        2. Regularization term: Penalizes high confidence predictions that are far from targets

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

        # Regularization term - penalize overconfident incorrect predictions
        error_term = v * (targets - mu) ** 2
        reg = error_term * (2.0 * alpha + v)

        # Combine losses with regularization weight
        loss = nll + self.evidential_regularization * reg

        return loss.mean()

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> float:
        """Train using evidential regression loss."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # Get evidential parameters
                mu, v, alpha, beta = self.forward_once(batch_x)

                # Compute evidential loss
                loss = self.evidential_loss(mu, v, alpha, beta, batch_y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            final_loss = epoch_loss

        return final_loss
