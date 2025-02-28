
from typing import List, Literal, Optional, Tuple

import torch
from bopep.surrogate_model.base_models import BiLSTMNetwork, MLPNetwork
from bopep.surrogate_model.helpers import BasePredictionModel

class NeuralNetworkEnsemble(BasePredictionModel):
    """Ensemble of neural networks for prediction with uncertainty estimation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        n_networks: int = 5,
        network_type: Literal["mlp", "bilstm"] = "mlp",
        lstm_layers: int = 1,
        lstm_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.n_networks = n_networks
        
        if network_type == "mlp":
            self.networks = torch.nn.ModuleList([
                MLPNetwork(input_dim, hidden_dims, dropout_rate=0)
                for _ in range(n_networks)
            ])
        elif network_type == "bilstm":
            lstm_hidden = lstm_hidden_dim or hidden_dims[0]
            self.networks = torch.nn.ModuleList([
                BiLSTMNetwork(input_dim, lstm_hidden, lstm_layers, dropout_rate=0)
                for _ in range(n_networks)
            ])
        elif network_type == "bigru":
            raise NotImplemented("BiGRUNetwork not implemented yet")
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
    
    def forward_predict(
        self, 
        x: torch.Tensor, 
        lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x shape:
        - (N, D) for MLP
        - (N, L, D) for BiLSTM
        lengths = None or List[int] for variable-length sequences
        """
        # Forward pass through each network in the ensemble, passing lengths
        predictions = torch.stack(
            [net(x, lengths=lengths) for net in self.networks], 
            dim=0
        )  # shape => (n_networks, N, 1)
        
        mean = torch.mean(predictions, dim=0)  # (N, 1)
        std = torch.std(predictions, dim=0)    # (N, 1)
        return mean, std