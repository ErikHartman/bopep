import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class DictHandler:
    """Utility class for handling dictionary inputs and outputs."""

    def prepare_data_from_dict(
        self, embedding_dict: Dict[str, np.ndarray], scores_dict: Dict[str, float]
    ):
        """
        Return (list_of_keys, X_torch, Y_torch) where:
          - X_torch is either (N, D) for MLP or (N, L, D) for LSTM
          - Y_torch is (N,) or (N, 1) for regression
        """
        peptides = list(embedding_dict.keys())
        # Stack up embeddings
        X_list = []
        Y_list = []
        for p in peptides:
            if p in scores_dict:
                X_list.append(embedding_dict[p])
                Y_list.append(scores_dict[p])
        # Convert to numpy and then to torch
        # If all embeddings have shape (D,) -> final shape (N, D)
        # If all embeddings have shape (L, D) -> final shape (N, L, D)
        # (This requires that each embedding has the same shape except for the batch dimension.)
        X = np.stack(X_list).astype(np.float32)
        Y = np.array(Y_list).astype(np.float32)

        X_torch = torch.tensor(X, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32).view(-1, 1)  # shape (N, 1)

        return peptides, X_torch, Y_torch

    def prepare_predictions_dict(
        self, peptides: list, mean_pred: np.ndarray, std_pred: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Convert arrays of mean and std back to a dictionary:
            { peptide: (mean, std), ... }
        """
        return {
            pep: (float(m), float(s))
            for pep, m, s in zip(peptides, mean_pred, std_pred)
        }


class VariableLengthDataset(Dataset):
    """
    Stores (embedding, score) pairs.
    Each embedding can have shape (D,) for MLP or (L, D) for a variable-length embedding.
    """

    def __init__(
        self, embedding_dict: Dict[str, np.ndarray], scores_dict: Dict[str, float]
    ):
        super().__init__()
        self.peptides = list(embedding_dict.keys())
        self.embeddings = [
            torch.tensor(embedding_dict[p], dtype=torch.float32) for p in self.peptides
        ]
        self.scores = [
            torch.tensor(scores_dict[p], dtype=torch.float32) for p in self.peptides
        ]

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        # Return (embedding_tensor, score)
        return self.embeddings[idx], self.scores[idx]


def variable_length_collate_fn(batch):
    """
    batch: List of (embedding_tensor, score_tensor)
    """
    embeddings = [item[0] for item in batch]  # List[Tensor]
    scores = [item[1] for item in batch]  # List[Tensor]

    # Check shape:
    # If embeddings[0] is 1D -> we assume MLP style (all must have the same dimension)
    # If embeddings[0] is 2D -> we assume variable length => pad_sequence
    if embeddings[0].dim() == 1:
        # MLP shape => (D,)
        # Just stack them directly => final (N, D)
        x_padded = torch.stack(embeddings, dim=0)  # shape (N, D)
        lengths = None
    else:
        # LSTM shape => (L, D)
        # We must pad them => shape (N, L_max, D)
        lengths = [
            emb.size(0) for emb in embeddings
        ]  # keep track of each sequence length
        x_padded = pad_sequence(embeddings, batch_first=True)

    # Stack scores => shape (N, 1) for regression
    y_stacked = torch.stack(scores).unsqueeze(-1)  # shape (N, 1)

    return x_padded, y_stacked, lengths


class BasePredictionModel(torch.nn.Module):
    """
    Base class for all prediction models with uncertainty.
    Supports both MLP and BiLSTM. Now handles variable-length embeddings
    via padding and optional packing in the forward pass.

    Subclasses must implement the forward_predict(x, lengths) method.
    """

    def __init__(self):
        super().__init__()
        # If you still have a DictHandler for other tasks, you could keep it.
        # self.dict_handler = DictHandler()

    def fit_dict(
        self,
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None
    ) -> float:
        """Train using dictionaries of embeddings and scores."""
        # Use the model's device if none specified
        if device is None:
            device = next(self.parameters()).device
            
        dataset = VariableLengthDataset(embedding_dict, scores_dict)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=variable_length_collate_fn,
        )
        return self._fit_from_dataloader(dataloader, epochs, learning_rate, device)

    def predict_dict(
        self, 
        embedding_dict: Dict[str, np.ndarray], 
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Predict for a dictionary of embeddings, returning {pep: (mean, std), ...}."""
        # Use the model's device if none specified
        if device is None:
            device = next(self.parameters()).device
            
        # We'll gather peptides in the same order as the dataset
        peptides = list(embedding_dict.keys())

        # Build a dataset without scores (we can just pass a dummy or 0.0)
        dummy_scores = {p: 0.0 for p in peptides}
        dataset = VariableLengthDataset(embedding_dict, dummy_scores)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=variable_length_collate_fn,
        )

        self.eval()
        mean_list, std_list = [], []

        with torch.no_grad():
            for batch_x, _, lengths in dataloader:
                # Move batch to the correct device
                batch_x = batch_x.to(device)
                
                # forward_predict should handle (N, L, D) or (N, D)
                mean_pred, std_pred = self.forward_predict(batch_x, lengths)
                mean_list.append(mean_pred.cpu())  # Move back to CPU for numpy
                std_list.append(std_pred.cpu())    # Move back to CPU for numpy

        # Concatenate all predictions
        mean_array = torch.cat(mean_list).view(-1).numpy()
        std_array = torch.cat(std_list).view(-1).numpy()

        # Build dictionary of {peptide: (mean, std)}
        predictions_dict = {}
        for i, pep in enumerate(peptides):
            predictions_dict[pep] = (float(mean_array[i]), float(std_array[i]))

        return predictions_dict

    def _fit_from_dataloader(
        self,
        dataloader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        patience: int = 10,  # Early stopping patience
        min_delta: float = 0.0001,  # Minimum improvement threshold
    ) -> float:
        """Internal method to train on a DataLoader. Returns the final epoch's average MSE."""
        # Use the model's device if none specified
        if device is None:
            device = next(self.parameters()).device
            
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        criterion = torch.nn.MSELoss()

        # Early stopping variables
        best_loss = float('inf')
        best_model_state = None
        counter = 0  # Count epochs without improvement

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y, lengths in dataloader:
                # Move batch data to the correct device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                # Pass lengths to forward_predict for sequence handling
                mean_pred, _ = self.forward_predict(batch_x, lengths)
                loss = criterion(mean_pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            scheduler.step(epoch_loss)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
            
            # Check if this is the best loss so far
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                best_model_state = self.state_dict().copy()
                counter = 0
            else:
                counter += 1
            
            # Early stopping check
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                return best_loss

        # Restore best model if training completed without early stopping
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return best_loss

    def forward_predict(
        self, x: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method: must return (mean, std).

        For MLP: x has shape (N, D), lengths=None (ignored).
        For BiLSTM: x has shape (N, L_max, D), plus a list of lengths.
        E.g. you might call pack_padded_sequence(x, lengths, batch_first=True).
        """
        raise NotImplementedError("Subclasses must implement forward_predict.")
