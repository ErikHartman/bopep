import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

class NNEnsemble(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        n_networks: int = 5,
        dropout_rate: float = 0.1,
    ):
        """
        Neural Network Ensemble using PyTorch.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            n_networks: Number of networks in ensemble
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.n_networks = n_networks

        self.networks = nn.ModuleList(
            [
                self._create_network(input_dim, hidden_dims, output_dim, dropout_rate)
                for _ in range(n_networks)
            ]
        )
        
        # Scalers for X and Y
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self._scalers_fitted = False  # Will be set to True after the first fit

    def _create_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ensemble.
        Returns (mean, std) of predictions across all networks.
        
        predictions shape: (n_networks, batch_size, output_dim)
        """
        predictions = torch.stack([net(x) for net in self.networks], dim=0)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        return mean, std

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> float:
        """
        Train on raw torch Tensors. Returns only the final epoch's average MSE loss.
        
        Args:
            train_x: Torch tensor of shape (N, input_dim).
            train_y: Torch tensor of shape (N, output_dim).
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Initial learning rate for optimizer.

        Returns:
            final_loss (float): The *average* MSE over the final epoch.
        """
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        final_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass for each network
                preds = [net(batch_x) for net in self.networks]
                # Compute MSE for each network, then average
                loss_per_network = [criterion(pred, batch_y) for pred in preds]
                loss = sum(loss_per_network) / self.n_networks

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss over all batches for this epoch
            epoch_loss /= len(dataloader)
            final_loss = epoch_loss

        return final_loss
    
    def fit_dict(
        self,
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> float:
        """
        Fit the ensemble using a dictionary of embeddings and scores.
        
        Args:
            embedding_dict: {peptide: embedding array}
            scores_dict: {peptide: score value}
            epochs, batch_size, learning_rate: training hyperparameters
        
        Returns:
            final_loss (float): The *average* MSE over the final training epoch.
        """
        # Collect data from the dictionaries
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        Y = np.array([scores_dict[p] for p in peptides], dtype=np.float32).reshape(-1, 1)

        # Fit or update scalers (if needed)
        self.x_scaler.fit(X)
        self.y_scaler.fit(Y)
        self._scalers_fitted = True

        # Scale X, Y
        X_scaled = self.x_scaler.transform(X)
        Y_scaled = self.y_scaler.transform(Y)

        # Convert to torch tensors
        X_torch = torch.tensor(X_scaled, dtype=torch.float32)
        Y_torch = torch.tensor(Y_scaled, dtype=torch.float32)

        # Train using the standard Tensor-based fit
        final_loss = self.fit(
            train_x=X_torch,
            train_y=Y_torch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        return final_loss

    def predict_dict(
        self,
        embedding_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Predict using the ensemble for a dictionary of embeddings.
        Returns {peptide: (mean, std)} in *original* scale of Y (if fitted).

        Args:
            embedding_dict: {peptide: embedding array}

        Returns:
            A dict {peptide: (mean, std)} in the unscaled space.
        """
        self.eval()

        # Gather in consistent order
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)

        # Scale inputs if scalers are fitted
        if self._scalers_fitted:
            X_scaled = self.x_scaler.transform(X)
        else:
            X_scaled = X

        X_torch = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            mean_pred, std_pred = self.forward(X_torch)
        
        mean_pred = mean_pred.cpu().numpy().reshape(-1)
        std_pred = std_pred.cpu().numpy().reshape(-1)

        # Inverse scale predictions if scalers are fitted
        if self._scalers_fitted:
            y_scale = self.y_scaler.scale_[0]
            y_mean = self.y_scaler.mean_[0]
            
            mean_pred = mean_pred * y_scale + y_mean
            std_pred = std_pred * y_scale
        
        # Build output dictionary
        result = {}
        for i, p in enumerate(peptides):
            result[p] = (float(mean_pred[i]), float(std_pred[i]))

        return result
