import torch
from typing import List, Tuple, Literal, Optional

from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.network_factory import NetworkFactory


class MVE(BasePredictionModel):
    """
    A model that uses the MVE method and outputs:
      - mu
      - log_var (log of predictive variance, log(sigma^2))

    Then we minimize the negative log-likelihood.

    https://arxiv.org/abs/2302.08875
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        mve_regularization: float = 0.01,
        n_objectives: int = 1,  # Support for multi-objective outputs
        **kwargs,
    ):
        super().__init__()

        # We need 2 outputs per objective: mu and log_var
        output_dim = 2
        self.mve_regularization = mve_regularization
        self.n_objectives = n_objectives

        self.network = NetworkFactory.get_network(
            network_type=network_type,
            input_dim=input_dim,
            output_dim=output_dim,
            n_objectives=n_objectives,
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )

    def forward_once(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns mu and log_var for each sample.
        """
        outputs = self.network(x, lengths=lengths)
        
        if self.n_objectives == 1:
            # Original behavior: outputs shape is (N, 2)
            mu = outputs[:, 0:1]
            log_var = outputs[:, 1:2]
        else:
            # Multi-objective: outputs shape is (N, n_objectives, 2)
            mu = outputs[:, :, 0:1]  # (N, n_objectives, 1)
            log_var = outputs[:, :, 1:2]  # (N, n_objectives, 1)

        return mu, log_var

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward_once, but we exponentiate log_var and sqrt to get sigma.
        
        Returns:
        - For single objective (n_objectives=1): mean (N, 1), std (N, 1)
        - For multi-objective (n_objectives>1): mean (N, n_objectives), std (N, n_objectives)
        """
        mu, log_var = self.forward_once(x, lengths=lengths)
        sigma = torch.exp(0.5 * log_var)
        
        # Squeeze the last dimension for consistency with other models
        if self.n_objectives == 1:
            mu = mu.squeeze(-1).unsqueeze(-1)  # Keep (N, 1) shape
            sigma = sigma.squeeze(-1).unsqueeze(-1)
        else:
            mu = mu.squeeze(-1)  # (N, n_objectives, 1) -> (N, n_objectives)
            sigma = sigma.squeeze(-1)
            
        return mu, sigma

    def negative_log_likelihood(
        self, mu: torch.Tensor, log_var: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        NLL = 0.5 * [ log_var + (targets - mu)^2 / exp(log_var) ]
        
        Args:
            mu: Predicted mean - shape (N, n_objectives, 1) or (N, 1)
            log_var: Predicted log variance - same shape as mu
            targets: Ground truth targets - shape (N, n_objectives) or (N, 1)
        """
        # Handle both single and multi-objective cases
        if self.n_objectives > 1:
            # Multi-objective: mu has shape (N, n_objectives, 1), targets (N, n_objectives)
            # Squeeze the last dimension from predictions
            mu = mu.squeeze(-1)  # (N, n_objectives)
            log_var = log_var.squeeze(-1)
            
            # Ensure targets has the right shape
            if targets.dim() == 1:
                targets = targets.unsqueeze(-1)  # (N,) -> (N, 1)
            if targets.shape[1] == 1 and self.n_objectives > 1:
                # Broadcast single target to all objectives (assumes same target for all)
                targets = targets.expand(-1, self.n_objectives)  # (N, 1) -> (N, n_objectives)
        else:
            # Single objective: ensure consistent shapes
            mu = mu.squeeze(-1)  # (N, 1) -> (N,)
            log_var = log_var.squeeze(-1)
            targets = targets.squeeze(-1) if targets.dim() > 1 else targets

        nll = 0.5 * (
            log_var
            + ((targets - mu) ** 2) * torch.exp(-log_var)
            + torch.log(2 * torch.tensor(torch.pi))
        )

        # For multi-objective, sum losses across objectives, then average across batch
        if self.n_objectives > 1:
            nll = nll.sum(dim=-1)  # Sum across objectives
        
        return nll.mean()  # Average across batch

    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        Override the default loss calculation to use negative log likelihood.
        """
        mu, log_var = self.forward_once(batch_x, lengths)
        nll_loss = self.negative_log_likelihood(mu, log_var, batch_y)

        # Regularization term: encourage reasonable variance estimates
        # For multi-objective, sum regularization across objectives
        if self.n_objectives > 1:
            reg_term = torch.mean(-log_var.squeeze(-1).sum(dim=-1))  # Sum across objectives, mean across batch
        else:
            reg_term = torch.mean(-log_var)
            
        loss = nll_loss + self.mve_regularization * reg_term

        return loss

    def _get_default_criterion(self):
        """
        Default criterion is None, as we use negative log likelihood (used in helpers.py).
        """
        return None
