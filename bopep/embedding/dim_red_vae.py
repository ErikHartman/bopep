import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
from tqdm import tqdm

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=[]):
        """
        Variational autoencoder for dimensionality reduction
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Target dimension for reduced representation
            hidden_layers: List of hidden layer dimensions for the encoder 
                          (decoder will be symmetric)
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Build encoder layers
        encoder_layers = []
        last_dim = input_dim
        
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(last_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = hidden_dim
        
        # Create mean and log variance layers for the latent space
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)
        
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
        
    def encode(self, x):
        """Encode input to latent mean and log variance"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from N(mu, var) using N(0, 1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent representation back to original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode, reparameterize, decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function: reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for the KL divergence term (beta-VAE)
        
    Returns:
        Total loss, reconstruction loss, KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


def train_vae(data, latent_dim, hidden_layers=[], batch_size=64, 
              max_epochs=100, patience=10, learning_rate=1e-3,
              weight_decay=1e-5, beta=1.0, verbose=True):
    """
    Train a variational autoencoder for dimensionality reduction
    
    Args:
        data: Input data as numpy array or torch tensor
        latent_dim: Target dimension for reduced representation
        hidden_layers: List of hidden layer dimensions
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        beta: Weight for the KL divergence term in VAE loss
        verbose: Whether to print progress
        
    Returns:
        Trained VAE model
    """
    # Convert input to tensor if it's numpy array
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VAE on {device}")
    
    # Create DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = data.shape[-1]
    model = VariationalAutoencoder(input_dim, latent_dim, hidden_layers).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    # Training loop with early stopping
    best_loss = float('inf')
    best_model = None
    counter = 0
    
    progress_bar = tqdm(range(max_epochs)) if verbose else range(max_epochs)
    for epoch in progress_bar:
        model.train()
        train_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            # Forward pass
            reconstructed, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = vae_loss_function(reconstructed, inputs, mu, logvar, beta=beta)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            
        train_loss /= len(dataloader.dataset)
        recon_loss_avg = recon_loss_sum / len(dataloader.dataset)
        kl_loss_avg = kl_loss_sum / len(dataloader.dataset)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        if verbose:
            progress_bar.set_description(
                f"Epoch {epoch+1}/{max_epochs}, Loss: {train_loss:.4f}, "
                f"Recon: {recon_loss_avg:.4f}, KL: {kl_loss_avg:.4f}"
            )
        
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


def reduce_dimension_vae(data, latent_dim, is_sequence_data=False, **train_kwargs):
    """
    Reduce dimension of data using variational autoencoder
    
    Args:
        data: Input data, can be dictionary of embeddings or numpy array
        latent_dim: Target dimension for the reduced embeddings
        is_sequence_data: Whether the data is sequential (3D) or not (2D)
        train_kwargs: Additional arguments for VAE training
        
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
    
    # Train VAE
    model = train_vae(scaled_tensor, latent_dim, **train_kwargs)
    
    # Reduce dimension - get the latent mean vectors
    with torch.no_grad():
        mu, _ = model.encode(scaled_tensor)
        reduced_data = mu.numpy()
    
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
