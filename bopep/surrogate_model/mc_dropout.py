from typing import List, Literal, Optional, Tuple

import torch
from bopep.surrogate_model.base_models import BiLSTMNetwork, MLPNetwork
from bopep.surrogate_model.helpers import BasePredictionModel

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
        elif network_type == "bigru":
            raise NotImplementedError("BiGRUNetwork not implemented yet")
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
    
    def forward_once(
        self,
        x: torch.Tensor,
        lengths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Single forward pass with dropout activated.
        For MLP => shape (N, D), lengths ignored.
        For BiLSTM => shape (N, L, D), lengths used for packing or slicing if not None.
        """
        return self.network(x, lengths=lengths)
    
    def forward_predict(
        self, 
        x: torch.Tensor, 
        lengths: Optional[List[int]] = None
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
            preds.append(y_hat.unsqueeze(0))               # shape (1, N, 1)
        
        all_preds = torch.cat(preds, dim=0)  # (mc_samples, N, 1)
        
        if not prev_mode:
            self.eval()  # return to eval mode if we were in eval
        
        mean = all_preds.mean(dim=0)  # (N, 1)
        std = all_preds.std(dim=0)    # (N, 1)
        return mean, std