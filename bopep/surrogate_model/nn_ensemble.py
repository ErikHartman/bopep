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
        dropout_rate: float = 0.05, # dropout so that networks don't converge on the same solution
        network_type: Literal["mlp", "bilstm", "bigru"] = "mlp",
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        n_objectives: int = 1,  # Support for multi-objective outputs
        **kwargs,
    ):
        super().__init__()
        self.n_networks = n_networks
        self.n_objectives = n_objectives

        self.networks = torch.nn.ModuleList(
            [
                NetworkFactory.get_network(
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
        
        Returns:
        - For single objective (n_objectives=1): mean (N, 1), std (N, 1)
        - For multi-objective (n_objectives>1): mean (N, n_objectives), std (N, n_objectives)
        """
        # Forward pass through each network in the ensemble, passing lengths
        predictions = torch.stack(
            [net(x, lengths=lengths) for net in self.networks], dim=0
        )  # shape => (n_networks, N, ...) where ... depends on n_objectives
        
        if self.n_objectives == 1:
            # Original behavior: predictions shape is (n_networks, N, 1)
            mean = torch.mean(predictions, dim=0)  # (N, 1)
            std = torch.std(predictions, dim=0)  # (N, 1)
        else:
            # Multi-objective: predictions shape is (n_networks, N, n_objectives, 1)
            # Squeeze the last dimension since output_dim=1
            predictions = predictions.squeeze(-1)  # (n_networks, N, n_objectives)
            mean = torch.mean(predictions, dim=0)  # (N, n_objectives)
            std = torch.std(predictions, dim=0)  # (N, n_objectives)
        
        return mean, std

    def _get_default_criterion(self):
        """
        Neural Network Ensemble uses standard MSE loss during training.
        Each model in the ensemble is trained to minimize MSE.
        """
        return torch.nn.MSELoss()
    
    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        For NN Ensemble, we calculate the loss as the average loss across all networks.
        """
        total_loss = 0.0
        for network in self.networks:
            pred = network(batch_x, lengths=lengths)
            
            # Handle shape mismatch between predictions and targets
            if self.n_objectives == 1:
                # Single objective: model outputs (N, 1, 1), targets are (N,)
                # Squeeze both dimensions to get (N,)
                pred = pred.squeeze()
            else:
                # Multi-objective: model outputs (N, n_objectives, 1), targets are (N, n_objectives)
                # Squeeze the last dimension only
                pred = pred.squeeze(-1)
                
            total_loss += criterion(pred, batch_y)
        return total_loss / len(self.networks)
