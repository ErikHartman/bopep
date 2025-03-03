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
        dropout_rate: float = 0.1,
        mc_samples: int = 20,
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate

        # Use NetworkFactory to create the appropriate network
        self.network = NetworkFactory.get_network(
            network_type=network_type,
            input_dim=input_dim,
            output_dim=1,
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
        """
        prev_mode = self.training
        self.train()  # ensure dropout is active

        preds = []
        for _ in range(self.mc_samples):
            # Now we pass lengths into forward_once:
            y_hat = self.forward_once(x, lengths=lengths)  # shape (N, 1)
            preds.append(y_hat.unsqueeze(0))  # shape (1, N, 1)

        all_preds = torch.cat(preds, dim=0)  # (mc_samples, N, 1)

        if not prev_mode:
            self.eval()  # return to eval mode if we were in eval

        mean = all_preds.mean(dim=0)  # (N, 1)
        std = all_preds.std(dim=0)  # (N, 1)
        return mean, std
