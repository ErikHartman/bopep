import torch
import numpy as np
import optuna
from typing import Dict, Tuple, Type
from sklearn.model_selection import train_test_split

from bopep.surrogate_model.base_models import BasePredictionModel, BiLSTMNetwork, DictHandler, MLPNetwork

class OptunaOptimizer:
    """Hyperparameter optimizer using Optuna for neural network models."""

    
    def __init__(
        self,
        model_class: Type[BasePredictionModel],
        embedding_dict: Dict[str, np.ndarray],
        scores_dict: Dict[str, float],
        n_trials: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
        early_stopping_rounds: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_class: The model class to optimize (MLPNetwork-based or BiLSTMNetwork-based)
            embedding_dict: Dictionary of embeddings {peptide_id: embedding_array}
            scores_dict: Dictionary of scores {peptide_id: score}
            n_trials: Number of Optuna trials to run
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            early_stopping_rounds: Number of epochs with no improvement after which to stop
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_class = model_class
        self.embedding_dict = embedding_dict
        self.scores_dict = scores_dict
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.device = device
        self.dict_handler = DictHandler()
        
        # Prepare data and split into train/validation sets
        peptides, _, _ = self.dict_handler.prepare_data_from_dict(embedding_dict, scores_dict)
        
        # Get indices for train/validation split
        indices = np.arange(len(peptides))
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # Create train/validation datasets
        self.train_peptides = [peptides[i] for i in train_indices]
        self.val_peptides = [peptides[i] for i in val_indices]
        
        self.train_embedding_dict = {p: embedding_dict[p] for p in self.train_peptides}
        self.train_scores_dict = {p: scores_dict[p] for p in self.train_peptides}
        
        self.val_embedding_dict = {p: embedding_dict[p] for p in self.val_peptides}
        self.val_scores_dict = {p: scores_dict[p] for p in self.val_peptides}
        
        self.input_dim = next(iter(embedding_dict.values())).shape[0]
        self.best_params = None
        self.best_val_loss = float('inf')
        self.best_trial = None
    
    def _get_mlp_params(self, trial: optuna.Trial) -> dict:
        """Define the hyperparameter search space for MLP networks."""
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 16, 256, log=True))
        
        params = {
            "input_dim": self.input_dim,
            "hidden_dims": hidden_dims,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 50, 200),
        }
        
        return params
    
    def _get_bilstm_params(self, trial: optuna.Trial) -> dict:
        """Define the hyperparameter search space for BiLSTM networks."""
        params = {
            "input_dim": self.input_dim,
            "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, log=True),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 50, 200),
        }
        
        return params
    
    def _evaluate_model(self, model: BasePredictionModel) -> float:
        """Evaluate the model on validation data and return the MSE."""
        model.eval()
        
        # Prepare validation data
        _, X_val, y_val = self.dict_handler.prepare_data_from_dict(
            self.val_embedding_dict, self.val_scores_dict
        )
        
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        with torch.no_grad():
            mean_pred, _ = model.forward_predict(X_val)
            mse = torch.nn.functional.mse_loss(mean_pred, y_val).item()
        
        return mse
    
    def _train_with_early_stopping(
        self, 
        model: BasePredictionModel, 
        params: dict,
        trial: optuna.Trial
    ) -> float:
        """Train the model with early stopping and return the best validation loss."""
        # Setup the model for training
        model.train()
        model = model.to(self.device)
        
        # Extract training parameters
        learning_rate = params.pop("learning_rate", 1e-3)
        batch_size = params.pop("batch_size", 32)
        epochs = params.pop("epochs", 100)
        
        # Prepare training data
        _, X_train, y_train = self.dict_handler.prepare_data_from_dict(
            self.train_embedding_dict, self.train_scores_dict
        )
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Track best validation performance
        best_val_loss = float('inf')
        no_improve_count = 0
        
        for epoch in range(epochs):
            # Training loop
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                mean_pred, _ = model.forward_predict(batch_x)
                loss = criterion(mean_pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            
            # Evaluate on validation set
            model.eval()
            val_loss = self._evaluate_model(model)
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.early_stopping_rounds:
                break
                
        return best_val_loss
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function that creates and evaluates a model."""
        # Determine model type and get appropriate hyperparameters
        if issubclass(self.model_class, MLPNetwork):
            params = self._get_mlp_params(trial)
            network = MLPNetwork(
                input_dim=params["input_dim"],
                hidden_dims=params["hidden_dims"],
            )
        elif issubclass(self.model_class, BiLSTMNetwork):
            params = self._get_bilstm_params(trial)
            network = BiLSTMNetwork(
                input_dim=params["input_dim"],
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
            )
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")
        
        # Create a custom prediction model using the network that returns fixed std
        model = CustomPredictionModel(network)
        
        # Train the model with early stopping
        val_loss = self._train_with_early_stopping(model, params, trial)
        
        # Update best params if this trial is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_trial = trial
            
            # Store hyperparameters
            self.best_params = {
                "val_loss": val_loss,
                **{k: v for k, v in params.items() if k not in ["input_dim"]}
            }
            
            # If using MLPNetwork, add hidden_dims
            if issubclass(self.model_class, MLPNetwork):
                self.best_params["hidden_dims"] = params["hidden_dims"]
                
            # If using BiLSTMNetwork, add specific params
            if issubclass(self.model_class, BiLSTMNetwork):
                self.best_params["hidden_dim"] = params["hidden_dim"]
                self.best_params["num_layers"] = params["num_layers"]
        
        return val_loss
    
    def optimize(self) -> Dict:
        """Run the hyperparameter optimization and return the best parameters."""
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10, interval_steps=1
            )
        )
        
        print(f"Starting optimization with {self.n_trials} trials")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print("Optimization completed")

        print(f"Best hyperparameters: {study.best_params}")
        
        return self.best_params

class CustomPredictionModel(BasePredictionModel):
    def __init__(self, network):
        super().__init__()
        self.network = network
        
    def forward_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # For simplicity, we're returning a fixed uncertainty (std)
        mean = self.network(x)
        std = torch.ones_like(mean) * 0.1  # Fixed uncertainty
        return mean, std