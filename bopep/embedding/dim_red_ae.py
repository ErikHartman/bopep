import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=[]):
        """
        Flexible autoencoder for dimensionality reduction
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Target dimension for reduced representation
            hidden_layers: List of hidden layer dimensions for the encoder 
                           (decoder will be symmetric)
        """
        super(Autoencoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        last_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(last_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = hidden_dim
            
        # Add final encoder layer to latent dimension
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (symmetric to encoder)
        decoder_layers = []
        last_dim = latent_dim
        
        # Add hidden layers in reverse order
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(last_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = hidden_dim
            
        # Add final decoder layer back to input dimension
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(data, latent_dim, hidden_layers=[], batch_size=64, 
                     max_epochs=100, patience=10, learning_rate=1e-3, 
                     weight_decay=1e-5, verbose=True):
    """
    Train an autoencoder for dimensionality reduction
    
    Args:
        data: Input data as numpy array or torch tensor
        latent_dim: Target dimension for reduced representation
        hidden_layers: List of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        verbose: Whether to print progress
        
    Returns:
        Trained autoencoder model and scaler
    """
    # Convert input to tensor if it's numpy array
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = data.shape[-1]
    model = Autoencoder(input_dim, latent_dim, hidden_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    # Training loop with early stopping
    best_loss = float('inf')
    best_model = None
    counter = 0
    
    progress_bar = tqdm(range(max_epochs)) if verbose else range(max_epochs)
    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(dataloader.dataset)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        if verbose:
            progress_bar.set_description(f"Epoch {epoch+1}/{max_epochs}, Loss: {train_loss:.6f}")
        
        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            best_model = copy.deepcopy(model)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model is None:
        best_model = model
        
    return best_model.to('cpu')


def reduce_dimension_ae(data, latent_dim, is_sequence_data=False, **train_kwargs):
    """
    Reduce dimension of data using autoencoder
    
    Args:
        data: Input data, can be dictionary of embeddings or numpy array
        latent_dim: Target dimension for the reduced embeddings
        is_sequence_data: Whether the data is sequential (3D) or not (2D)
        train_kwargs: Additional arguments for autoencoder training
        
    Returns:
        Reduced data in the same format as input
    """
    # Determine type and prepare data
    is_dict_input = isinstance(data, dict)
    
    if is_dict_input:
        input_keys = list(data.keys())
        
        # Extract numpy arrays from dictionary
        if is_sequence_data:
            # For sequence data, we flatten all sequences
            all_vectors = []
            seq_lengths = {}
            
            for key, emb in data.items():
                seq_lengths[key] = len(emb)  # Store original sequence length
                all_vectors.extend(emb)      # Add each position embedding
                
            train_data = np.array(all_vectors)
        else:
            # For averaged data
            train_data = np.array(list(data.values()))
    else:
        # Input is already a numpy array
        train_data = data
    
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data)
    scaled_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    
    # Train autoencoder
    model = train_autoencoder(scaled_tensor, latent_dim, **train_kwargs)
    
    # Reduce dimension
    with torch.no_grad():
        reduced_data = model.encode(scaled_tensor).numpy()
    
    # Reconstruct output in the same format as input
    if is_dict_input:
        if is_sequence_data:
            # Reconstruct sequences
            result = {}
            start_idx = 0
            
            for key in input_keys:
                seq_len = seq_lengths[key]
                result[key] = reduced_data[start_idx:start_idx+seq_len]
                start_idx += seq_len
        else:
            # Reconstruct dictionary with averaged embeddings
            result = {input_keys[i]: reduced_data[i] for i in range(len(input_keys))}
    else:
        # Return as numpy array
        result = reduced_data
        
    return result
