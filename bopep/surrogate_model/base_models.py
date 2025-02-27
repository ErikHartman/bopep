import torch
import numpy as np
from typing import Dict, List, Tuple

class BaseNetwork(torch.nn.Module):
    """Base class for all network architectures."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        raise NotImplementedError("Subclasses must implement forward.")


class MLPNetwork(BaseNetwork):
    """Simple feed-forward neural network."""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 

        dropout_rate: float = 0.0
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BiLSTMNetwork(BaseNetwork):
    """Bidirectional LSTM network for sequence data."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        # BiLSTM output dimension is 2*hidden_dim (forward + backward)
        self.fc = torch.nn.Linear(2 * hidden_dim, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_length, input_dim]
        lstm_out, _ = self.lstm(x)
        # Use the final hidden state from both directions
        final_hidden = lstm_out[:, -1, :]
        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)
        return output


class DictHandler:
    """Utility class for handling dictionary inputs and outputs."""
    @staticmethod
    def prepare_data_from_dict(
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Convert dictionary data to tensors for training."""
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        Y = np.array([scores_dict[p] for p in peptides], dtype=np.float32).reshape(-1, 1)

        X_torch = torch.tensor(X, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)
        
        return peptides, X_torch, Y_torch
    
    @staticmethod
    def prepare_predictions_dict(
        peptides: List[str],
        mean_pred: np.ndarray,
        std_pred: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Convert prediction tensors back to dictionary format."""
        result = {}
        for i, p in enumerate(peptides):
            result[p] = (float(mean_pred[i]), float(std_pred[i]))
        return result


class BasePredictionModel(torch.nn.Module):
    """Base class for all prediction models with uncertainty."""
    def __init__(self):
        super().__init__()
        self.dict_handler = DictHandler()
    
    def fit_dict(
        self,
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> float:
        """Train using dictionaries of embeddings and scores."""
        _, X_torch, Y_torch = self.dict_handler.prepare_data_from_dict(
            embedding_dict, scores_dict
        )
        return self.fit(X_torch, Y_torch, epochs, batch_size, learning_rate)
    
    def predict_dict(
        self, embedding_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """Predict for a dictionary of embeddings."""
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        X_torch = torch.tensor(X, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            mean_pred, std_pred = self.forward_predict(X_torch)
        
        mean_pred = mean_pred.cpu().numpy().reshape(-1)
        std_pred = std_pred.cpu().numpy().reshape(-1)
        
        return self.dict_handler.prepare_predictions_dict(peptides, mean_pred, std_pred)
    
    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> float:
        """Train on torch Tensors. Returns the final epoch's average MSE."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                mean_pred, _ = self.forward_predict(batch_x)
                loss = criterion(mean_pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(dataloader)
            final_loss = epoch_loss
        return final_loss
    
    def forward_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Abstract method. Subclasses must implement this to return (mean, std)."""
        raise NotImplementedError("Subclasses must implement forward_predict.")
