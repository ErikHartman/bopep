from typing import List, Literal, Optional, Tuple

import torch
from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.network_factory import NetworkFactory


class NeuralNetworkEnsemble(BasePredictionModel):
    """Ensemble of neural networks for prediction with uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        n_networks: int = 5,
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_networks = n_networks

        # Create network ensemble using NetworkFactory
        self.networks = torch.nn.ModuleList(
            [
                NetworkFactory.get_network(
                    network_type=network_type,
                    input_dim=input_dim,
                    output_dim=1,
                    hidden_dims=hidden_dims,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    **kwargs,
                )
                for _ in range(n_networks)
            ]
        )

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x shape:
        - (N, D) for MLP
        - (N, L, D) for BiLSTM/BiGRU
        lengths = None or List[int] for variable-length sequences
        """
        # Forward pass through each network in the ensemble, passing lengths
        predictions = torch.stack(
            [net(x, lengths=lengths) for net in self.networks], dim=0
        )  # shape => (n_networks, N, 1)

        mean = torch.mean(predictions, dim=0)  # (N, 1)
        std = torch.std(predictions, dim=0)  # (N, 1)
        return mean, std
