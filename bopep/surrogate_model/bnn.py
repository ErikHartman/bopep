import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn as nn

class BayesianNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        mc_samples: int = 20
    ):
        """
        Bayesian Neural Network using Monte Carlo Dropout for uncertainty estimates.

        Args:
            input_dim:  Dimension of input features.
            hidden_dims: List of hidden-layer sizes, e.g. [64, 64].
            output_dim: Dimension of the output. (Often 1 for regression.)
            dropout_rate: Probability of dropout, used both in training and MC sampling.
            mc_samples: Number of forward passes to draw when predicting (for mean & std).
        """
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        One forward pass through the network (for training or single-sample inference).
        This returns shape (batch_size, output_dim).
        """
        return self.network(x)

    def forward_mcdropout(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Monte Carlo Dropout sampling. Does 'mc_samples' forward passes (with dropout on),
        then returns the mean and std across those samples.

        Returns:
            mean: shape (batch_size, output_dim)
            std:  shape (batch_size, output_dim)
        """
        prev_mode = self.training
        self.train()  # ensure dropout is active

        preds = []
        for _ in range(self.mc_samples):
            y_hat = self.forward_once(x)  # shape (batch_size, output_dim)
            preds.append(y_hat.unsqueeze(0))  # shape (1, batch_size, output_dim)

        # shape: (mc_samples, batch_size, output_dim)
        all_preds = torch.cat(preds, dim=0)

        # restore mode
        if not prev_mode:
            self.eval()

        mean = all_preds.mean(dim=0)
        std = all_preds.std(dim=0)
        return mean, std

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3
    ) -> float:
        """
        Train the BNN on raw torch Tensors (already scaled).
        Returns only the final epoch's average MSE.

        Args:
            train_x: shape (N, input_dim)
            train_y: shape (N, output_dim)
            epochs: number of training epochs
            batch_size: training batch size
            learning_rate: learning rate for optimizer
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

                # Standard single-pass forward for training
                preds = self.forward_once(batch_x)
                loss = criterion(preds, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

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
        Fit the BNN using dictionary inputs. 
        We apply scaling to X and Y.

        Args:
            embedding_dict: {peptide: embedding array of shape (..., input_dim)}
            scores_dict: {peptide: float} 
            epochs, batch_size, learning_rate: training hyperparams

        Returns:
            final_loss (float): final epoch's MSE in the scaled domain
        """

        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        Y = np.array([scores_dict[p] for p in peptides], dtype=np.float32).reshape(-1, 1)

        # to torch
        X_torch = torch.tensor(X, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)

        # train
        final_loss = self.fit(
            train_x=X_torch,
            train_y=Y_torch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        return final_loss

    def predict_dict(self, embedding_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """
        Predict using Monte Carlo Dropout for a dictionary of embeddings.
        Returns {peptide: (mean, std)} in the *original* scale of Y.

        Args:
            embedding_dict: {peptide: embedding array (input_dim,)}
        """
        # gather data
        peptides = list(embedding_dict.keys())
        X = np.array([embedding_dict[p] for p in peptides], dtype=np.float32)
        X_torch = torch.tensor(X, dtype=torch.float32)

        # MC dropout forward
        with torch.no_grad():
            mean_pred, std_pred = self.forward_mcdropout(X_torch)

        mean_pred = mean_pred.cpu().numpy()
        std_pred = std_pred.cpu().numpy()

        output = {}
        for i, p in enumerate(peptides):
            output[p] = (float(mean_pred[i]), float(std_pred[i]))

        return output
