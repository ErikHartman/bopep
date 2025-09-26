import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class ObjectiveMixin:
    """
    Mixin to handle objective structure detection and formatting.
    Centralizes all objective-related logic in one place.
    """
    
    def _setup_objectives(self, objective_dict: Dict[str, Union[float, Dict[str, float]]]):
        """Setup objective structure from training data."""
        sample_objective = next(iter(objective_dict.values()))
        
        if isinstance(sample_objective, dict):
            # Multi-objective (always named now)
            self._objective_names = sorted(sample_objective.keys())
            self._n_objectives = len(self._objective_names)
        else:
            # Single objective
            self._objective_names = None
            self._n_objectives = 1
    
    def _format_predictions(
        self, 
        peptides: List[str], 
        mean_array: np.ndarray, 
        std_array: np.ndarray
    ) -> Dict[str, Union[Tuple[float, float], Dict[str, Tuple[float, float]]]]:
        """Format predictions according to objective structure."""
        predictions = {}
        
        if self._objective_names:
            # Multi-objective: {peptide: {obj_name: (mean, std)}}
            for i, pep in enumerate(peptides):
                obj_predictions = {}
                for j, obj_name in enumerate(self._objective_names):
                    obj_predictions[obj_name] = (float(mean_array[i, j]), float(std_array[i, j]))
                predictions[pep] = obj_predictions
        else:
            # Single objective: {peptide: (mean, std)}
            mean_flat = mean_array.flatten() if mean_array.ndim > 1 else mean_array
            std_flat = std_array.flatten() if std_array.ndim > 1 else std_array
            
            for i, pep in enumerate(peptides):
                predictions[pep] = (float(mean_flat[i]), float(std_flat[i]))
        
        return predictions


class DictHandler:
    """Utility class for handling dictionary inputs and outputs."""

    def prepare_data_from_dict(
        self, embedding_dict: Dict[str, np.ndarray], objective_dict: Dict[str, float]
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
            if p in objective_dict:
                X_list.append(embedding_dict[p])
                Y_list.append(objective_dict[p])
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
    
    Supported objective formats:
    - Single objective: Dict[str, float] -> {peptide: score, peptide: score}
    - Multi objective: Dict[str, Dict[str, float]] -> {peptide: {name: score, name: score}}
    """

    def __init__(
        self, embedding_dict: Dict[str, np.ndarray], 
        objective_dict: Dict[str, Union[float, Dict[str, float]]]
    ):
        super().__init__()
        self.peptides = list(embedding_dict.keys())
        self.embeddings = [
            torch.tensor(embedding_dict[p], dtype=torch.float32) for p in self.peptides
        ]
        
        # Handle different objective formats
        self.scores = []
        for p in self.peptides:
            score = objective_dict[p]
            
            if isinstance(score, dict):
                # Multi-objective: extract values from dict {obj_name: score}
                score_values = []
                for obj_name in sorted(score.keys()):  # Sort for consistent ordering
                    obj_val = score[obj_name]
                    score_values.append(obj_val)
                self.scores.append(torch.tensor(score_values, dtype=torch.float32))
                
            else:
                # Single objective: convert float to tensor
                self.scores.append(torch.tensor(score, dtype=torch.float32))

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        # Return (embedding_tensor, score_tensor)
        return self.embeddings[idx], self.scores[idx]


def variable_length_collate_fn(batch):
    """
    batch: List of (embedding_tensor, score_tensor)
    Handles both single-objective (score is scalar) and multi-objective (score is vector) cases.
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
        # 2D shape => (L, D)
        # We must pad them => shape (N, L_max, D)
        lengths = [
            emb.size(0) for emb in embeddings
        ]  # keep track of each sequence length
        x_padded = pad_sequence(embeddings, batch_first=True) # pads with 0s

    # Handle scores - they can be scalars or vectors
    if scores[0].dim() == 0:
        # Single objective: scores are scalars
        y_stacked = torch.stack(scores)  # shape (N,) - keep it simple for single objective
    else:
        # Multi-objective: scores are vectors
        y_stacked = torch.stack(scores)  # shape (N, n_objectives)

    return x_padded, y_stacked, lengths


class BasePredictionModel(ObjectiveMixin, torch.nn.Module):
    """
    Base class for all prediction models with optional validation-based early stopping.
    """
    def __init__(self):
        super().__init__()
        # Initialize objective-related attributes
        self._objective_names = None
        self._n_objectives = 1

    def fit_dict(
        self,
        embedding_dict: Dict[str, np.ndarray],
        objective_dict: Dict[str, Union[float, Dict[str, float]]],
        val_embedding_dict: Optional[Dict[str, np.ndarray]] = None,
        val_objective_dict: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        criterion=None,
        clip_grad_norm: float = 1.0,
    ) -> float:
        """
        Train using dictionaries of embeddings and scores, with optional validation set.

        Args:
            embedding_dict: {peptide: embedding_array}
            objective_dict: {peptide: score} for single-objective, 
                          {peptide: [score1, score2, ...]} for multi-objective, or
                          {peptide: {"obj1": (mean, std), "obj2": (mean, std)}} for named multi-objective
            val_embedding_dict: Optional validation embeddings
            val_objective_dict: Optional validation scores (same format as objective_dict)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on
            verbose: Whether to print training progress
            criterion: Loss function (uses model default if None)
            clip_grad_norm: Gradient clipping threshold

        Returns:
            Final training or validation loss

        If a validation set is provided (val_embedding_dict & val_objective_dict),
        early stopping and LR scheduling are based on val_loss. Otherwise,
        training-only early stopping is used on training loss.
        """
        # Detect and store objective format using mixin method
        self._setup_objectives(objective_dict)
            
        if device is None:
            device = next(self.parameters()).device
        if criterion is None:
            criterion = self._get_default_criterion()

        # Build train loader
        train_ds = VariableLengthDataset(embedding_dict, objective_dict)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=variable_length_collate_fn,
        )

        # Build optional val loader
        if val_embedding_dict is not None and val_objective_dict is not None:
            val_ds = VariableLengthDataset(val_embedding_dict, val_objective_dict)
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=variable_length_collate_fn,
            )
        else:
            val_loader = None

        return self._fit_with_optional_validation(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            verbose=verbose,
            criterion=criterion,
            clip_grad_norm=clip_grad_norm,
        )

    def _fit_with_optional_validation(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int = 100,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        criterion=None,
        clip_grad_norm: Optional[float] = None,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> float:
        if device is None:
            device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min"
        )

        best_metric = float('inf')
        best_state = None
        counter = 0

        for epoch in range(1, epochs + 1):
            # --- Training ---
            self.train()
            train_loss = 0.0
            for x, y, lengths in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = self._calculate_loss(x, y, lengths, criterion)
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- Validation (optional) ---
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y, lengths in val_loader:
                        x, y = x.to(device), y.to(device)
                        loss = self._calculate_loss(x, y, lengths, criterion)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                metric = val_loss
            else:
                metric = train_loss


            scheduler.step(metric)
            temp_lr = scheduler.get_last_lr()[0]
            if verbose:
                msg = f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} | lr: {temp_lr:.6f}"
                if val_loader:
                    msg += f" | val_loss: {val_loss:.4f}"
                print(msg)

            # --- Early stopping ---
            if metric < best_metric - min_delta:
                best_metric = metric
                best_state = {k: v.cpu() for k, v in self.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return best_metric
        
    def _get_default_criterion(self):
        """
        Returns the appropriate loss function for the model type.
        Can be overridden by subclasses to provide specialized loss functions.
        """
        # Default to MSE for the base class
        return torch.nn.MSELoss()

    def predict_dict(
        self, 
        embedding_dict: Dict[str, np.ndarray], 
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        uncertainty_mode: Optional[str] = None
    ) -> Dict[str, Tuple]:
        """Predict for a dictionary of embeddings.
        
        Args:
            embedding_dict: Dictionary mapping peptides to their embeddings
            batch_size: Batch size for prediction
            device: Device to use for prediction
            uncertainty_mode: Which uncertainty component to use (for DeepEvidentialRegression)
            
        Returns:
            For single objective: {pep: (mean, std), ...}
            For named multi-objective: {pep: {obj_name: (mean, std), ...}, ...}
            For legacy multi-objective: {pep: ([mean1, mean2, ...], [std1, std2, ...]), ...}
        """

        if device is None:
            device = next(self.parameters()).device
            
        # We'll gather peptides in the same order as the dataset
        peptides = list(embedding_dict.keys())

        # Build a dataset without scores
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
                
                # Pass uncertainty_mode to forward_predict if the method accepts it
                # Check if the method has the uncertainty_mode parameter
                if hasattr(self, "forward_predict") and "uncertainty_mode" in self.forward_predict.__code__.co_varnames:
                    mean_pred, std_pred = self.forward_predict(batch_x, lengths, uncertainty_mode=uncertainty_mode)
                else:
                    # Default behavior for models that don't support uncertainty_mode
                    mean_pred, std_pred = self.forward_predict(batch_x, lengths)
                
                mean_list.append(mean_pred.cpu())  # Move back to CPU for numpy
                std_list.append(std_pred.cpu())    # Move back to CPU for numpy

        # Concatenate all predictions
        all_means = torch.cat(mean_list)  # Shape depends on n_objectives
        all_stds = torch.cat(std_list)
        
        # Convert to numpy and use mixin formatting
        mean_array = all_means.numpy()
        std_array = all_stds.numpy()
        
        return self._format_predictions(peptides, mean_array, std_array)
    
    def _calculate_loss(self, batch_x, batch_y, lengths, criterion):
        """
        Default loss calculation method. Can be overridden by subclasses.
        """
        # Default implementation for standard regression models
        mean_pred, _ = self.forward_predict(batch_x, lengths)
        return criterion(mean_pred, batch_y)

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
