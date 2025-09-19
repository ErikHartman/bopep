from typing import List, Literal, Optional, Tuple

import torch
from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.network_factory import NetworkFactory


class MonteCarloDropout(BasePredictionModel):
    """Monte Carlo Dropout model for prediction with uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.1, # dropout so that networks don't converge on the same solution
        mc_samples: int = 20,
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        n_objectives: int = 1,  # Support for multi-objective outputs
        **kwargs,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        self.n_objectives = n_objectives

        # Use NetworkFactory to create the appropriate network
        self.network = NetworkFactory.get_network(
            network_type=network_type,
            input_dim=input_dim,
            output_dim=1,
            n_objectives=n_objectives,
            hidden_dims=hidden_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )

    def forward_once(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Single forward pass with dropout activated.
        For MLP => shape (N, D), lengths ignored.
        For BiLSTM => shape (N, L, D), lengths used for packing or slicing if not None.
        """
        return self.network(x, lengths=lengths)

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout: Run forward pass multiple times in train mode,
        then compute mean & std.

        x shape can be:
          - (N, D) for MLP
          - (N, L, D) plus optional `lengths` for BiLSTM
          
        Returns:
        - For single objective (n_objectives=1): mean (N, 1), std (N, 1)
        - For multi-objective (n_objectives>1): mean (N, n_objectives), std (N, n_objectives)
        """
        prev_mode = self.training
        self.train()  # ensure dropout is active

        preds = []
        for _ in range(self.mc_samples):
            # Now we pass lengths into forward_once:
            y_hat = self.forward_once(x, lengths=lengths)  # shape varies based on n_objectives
            preds.append(y_hat.unsqueeze(0))  # add sample dimension

        all_preds = torch.cat(preds, dim=0)  # (mc_samples, N, ...) where ... depends on n_objectives

        if not prev_mode:
            self.eval()  # return to eval mode if we were in eval

        if self.n_objectives == 1:
            # Original behavior: all_preds shape is (mc_samples, N, 1)
            mean = all_preds.mean(dim=0)  # (N, 1)
            std = all_preds.std(dim=0)  # (N, 1)
        else:
            # Multi-objective: all_preds shape is (mc_samples, N, n_objectives, 1)
            # Squeeze the last dimension since output_dim=1
            all_preds = all_preds.squeeze(-1)  # (mc_samples, N, n_objectives)
            mean = all_preds.mean(dim=0)  # (N, n_objectives)
            std = all_preds.std(dim=0)  # (N, n_objectives)
            
        return mean, std

    def _get_default_criterion(self):
        """
        Monte Carlo Dropout uses standard MSE loss during training.
        """
        return torch.nn.MSELoss()

    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        Standard loss calculation for MC Dropout.
        During training, we make a single forward pass rather than multiple MC samples.
        """
        # During training, we just do a single forward pass with dropout active
        self.train()  # Ensure dropout is active
        mean_pred = self.forward_once(batch_x, lengths)
        
        # Handle shape mismatch between predictions and targets
        if self.n_objectives == 1:
            # Single objective: model outputs (N, 1, 1), targets are (N,)
            # Squeeze both dimensions to get (N,)
            mean_pred = mean_pred.squeeze()
        else:
            # Multi-objective: model outputs (N, n_objectives, 1), targets are (N, n_objectives)
            # Squeeze the last dimension only
            mean_pred = mean_pred.squeeze(-1)
            
        return criterion(mean_pred, batch_y)
