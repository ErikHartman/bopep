from typing import List, Literal, Optional, Tuple

import torch
from bopep.surrogate_model.base_models import BasePredictionModel, BiLSTMNetwork, MLPNetwork

class MonteCarloDropout(BasePredictionModel):
    """Monte Carlo Dropout model for prediction with uncertainty estimation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.1,
        mc_samples: int = 20,
        network_type: Literal["mlp", "bilstm"] = "mlp",
        lstm_layers: int = 1,
        lstm_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout_rate = dropout_rate
        
        if network_type == "mlp":
            self.network = MLPNetwork(input_dim, hidden_dims, dropout_rate)
        elif network_type == "bilstm":
            lstm_hidden = lstm_hidden_dim or hidden_dims[0]
            self.network = BiLSTMNetwork(input_dim, lstm_hidden, lstm_layers, dropout_rate)
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
    
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def forward_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Activate dropout by setting the model to train mode temporarily
        prev_mode = self.training
        self.train()
        preds = []
        for _ in range(self.mc_samples):
            y_hat = self.forward_once(x)
            preds.append(y_hat.unsqueeze(0))
        all_preds = torch.cat(preds, dim=0)
        if not prev_mode:
            self.eval()
        mean = all_preds.mean(dim=0)
        std = all_preds.std(dim=0)
        return mean, std