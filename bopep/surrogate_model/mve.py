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
        **kwargs
    ):
        super().__init__()

        # We only need 2 outputs: mu and log_var
        output_dim = 2
        self.mve_regularization = mve_regularization
        
        # Add clipping bounds for numerical stability
        self.log_var_min = -10.0  # Lower bound for log variance
        self.log_var_max = 5.0    # Upper bound for log variance

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
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns mu and log_var for each sample.
        """
        outputs = self.network(x, lengths=lengths)
        mu = outputs[:, 0:1]
        
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(outputs[:, 1:2], min=self.log_var_min, max=self.log_var_max)
        return mu, log_var

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward_once, but we exponentiate log_var and sqrt to get sigma.
        """
        mu, log_var = self.forward_once(x, lengths=lengths)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma

    def negative_log_likelihood(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        NLL = 0.5 * [ log_var + (targets - mu)^2 / exp(log_var) ]
        
        With improved numerical stability.
        """
        # Safer implementation of NLL
        squared_error = (targets - mu) ** 2
        precision = torch.exp(-log_var)  # 1/variance is more numerically stable
        
        nll = 0.5 * (log_var + squared_error * precision + torch.log(2 * torch.tensor(torch.pi)))
        return nll.mean() 

    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        Override the default loss calculation to use negative log likelihood.
        """
        mu, log_var = self.forward_once(batch_x, lengths)
        nll_loss = self.negative_log_likelihood(mu, log_var, batch_y)
        loss = nll_loss + self.mve_regularization * torch.mean(log_var)
        return loss
    
    def _get_default_criterion(self):
        """
        Default criterion is None, as we use negative log likelihood (used in helpers.py).
        """
        return None