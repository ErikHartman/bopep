import torch
import numpy as np
from typing import List, Tuple
import torch.nn as nn

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
        Forward pass through ensemble.
        Returns mean and standard deviation of predictions.
        """
        # predictions shape: (n_networks, batch_size, output_dim)
        predictions = torch.stack([net(x) for net in self.networks], dim=0)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        return mean, std

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the ensemble. Returns (mean, std) as numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            mean, std = self.forward(x)

        return mean.cpu().numpy(), std.cpu().numpy()

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> List[float]:
        """
        TODO: make this operate on a dictionary of training data.

        A custom training loop to fit the ensemble.
        Each network in the ensemble is trained to minimize MSE against the target.
        """
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        losses = []

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                preds = [net(batch_x) for net in self.networks]

                loss_per_network = [criterion(pred, batch_y) for pred in preds]
                loss = sum(loss_per_network) / self.n_networks

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            losses.append(epoch_loss)


        return losses
