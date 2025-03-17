from bopep.embedding.embed_esm import embed_esm
from bopep.embedding.embed_aaindex import embed_aaindex
from bopep.embedding.utils import filter_peptides
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from bopep.embedding.dim_red_vae import reduce_dimension_vae
import os


class Embedder:
    """
    TODO: 

    Maybe we should always reduce embeddings?

    """
    def __init__(self):
        pass

    def embed_esm(self, peptides: list, average: bool, model_path: str = None, batch_size:int = 128) -> dict:
        peptides = filter_peptides(peptides)
        embeddings = embed_esm(peptides, model_path, average, batch_size)
        return embeddings

    def embed_aaindex(self, peptides: list, average: bool) -> dict:
        peptides = filter_peptides(peptides)
        embeddings = embed_aaindex(peptides, average)
        return embeddings
    
    def scale_embeddings(self, embeddings: dict) -> dict:
        """
        Scales the embeddings using StandardScaler.
        Handles both 1D (vector form) and 2D (matrix form) embeddings.
        """
        # Initialize scaler
        scaler = StandardScaler()
        
        # Prepare data for scaling - flatten all embeddings to 2D
        all_embeddings = []
        for emb in embeddings.values():
            if len(emb.shape) == 2:  # Matrix form (length x embedding)
                all_embeddings.extend(emb)
            else:  # Vector form
                all_embeddings.append(emb)
                
        # Fit scaler on the embedding dimension
        scaler.fit(np.array(all_embeddings))
        
        # Transform embeddings while preserving original shapes
        scaled_embeddings = {}
        for peptide, emb in embeddings.items():
            if len(emb.shape) == 2:  # Matrix form
                # Scale each position's embedding separately
                scaled = scaler.transform(emb)
            else:  # Vector form
                scaled = scaler.transform(emb.reshape(1, -1))[0]
            scaled_embeddings[peptide] = scaled
            
        return scaled_embeddings


    def reduce_embeddings_pca(
        self, embeddings: dict, explained_variance_ratio: float = 0.95, n_components : int = None
    ):
        """
        Reduces the dimensionality of the embeddings using PCA.
        Works with both averaged embeddings (1D per peptide) and 
        sequence embeddings (2D per peptide, with varying lengths).
        
        Args:
            embeddings: Dictionary mapping peptide sequences to their embeddings
            explained_variance_ratio: Target explained variance ratio for PCA
            
        Returns:
            Dictionary with reduced embeddings maintaining the original structure
        """
        # Check if we have sequence embeddings (2D) or averaged embeddings (1D)
        first_emb = next(iter(embeddings.values()))
        is_sequence_embedding = len(first_emb.shape) > 1
        
        if is_sequence_embedding:
            # For sequence embeddings (average=False case)
            # Collect all position embeddings across all sequences
            all_position_embeddings = []
            for emb in embeddings.values():
                all_position_embeddings.extend(emb)  # Add each position's embedding
            
            all_position_embeddings = np.array(all_position_embeddings)
            
            # Apply PCA to all position embeddings
            if n_components is not None:
                pca = PCA(n_components=n_components)
            else:
                pca = PCA(n_components=explained_variance_ratio, svd_solver="full")
            pca.fit(all_position_embeddings)
            
            # Transform each peptide's sequence of embeddings
            reduced_embeddings = {}
            for peptide, emb in embeddings.items():
                reduced_embeddings[peptide] = pca.transform(emb)
                
            print(f"Original embedding dimension: {first_emb.shape[1]}")
            print(f"Reduced embedding dimension: {reduced_embeddings[next(iter(reduced_embeddings))].shape[1]}")
            
        else:
            # For averaged embeddings (average=True case) - original implementation
            embedding_array = np.array(list(embeddings.values()))
            peptide_sequences = list(embeddings.keys())

            pca = PCA(n_components=explained_variance_ratio, svd_solver="full")
            reduced_embeddings_array = pca.fit_transform(embedding_array)

            print("The reduced embeddings are of shape: ", reduced_embeddings_array.shape)

            reduced_embeddings = {
                peptide_sequences[i]: reduced_embeddings_array[i]
                for i in range(len(peptide_sequences))
            }

        return reduced_embeddings

    def reduce_embeddings_autoencoder(
        self, embeddings: dict, latent_dim: int, 
        hidden_layers=None, batch_size=64, max_epochs=100, 
        learning_rate=1e-3, patience=10, verbose=True, beta=1.0
    ):
        """
        Reduces the dimensionality of the embeddings using an autoencoder
        or variational autoencoder.
        
        Args:
            embeddings: Dictionary mapping peptide sequences to their embeddings
            latent_dim: Target dimension for the reduced representation
            hidden_layers: List of dimensions for hidden layers (default: [2*latent_dim])
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            learning_rate: Initial learning rate
            patience: Patience for early stopping
            verbose: Whether to print training progress
            autoencoder_type: Type of autoencoder to use ("standard" or "variational")
            
        Returns:
            Dictionary with reduced embeddings maintaining the original structure
        """
        # Check if we have sequence embeddings (2D) or averaged embeddings (1D)
        first_emb = next(iter(embeddings.values()))
        is_sequence_embedding = len(first_emb.shape) > 1
        
        # Set default hidden layers if not provided
        if hidden_layers is None:
            input_dim = first_emb.shape[-1] if is_sequence_embedding else first_emb.shape[0]
            hidden_layers = [min(input_dim, latent_dim * 2)]
        
        # Use the VAE to reduce dimensions
        reduced_embeddings = reduce_dimension_vae(
            data=embeddings,
            latent_dim=latent_dim,
            is_sequence_data=is_sequence_embedding,
            hidden_layers=hidden_layers,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            patience=patience,
            beta=beta,
            verbose=verbose
        )
        
        # Print information about the reduction
        if is_sequence_embedding:
            print(f"VAE reduction:")
            print(f"Original embedding dimension: {first_emb.shape[1]}")
            print(f"Reduced embedding dimension: {reduced_embeddings[next(iter(reduced_embeddings))].shape[1]}")
        else:
            print(f"VAE reduction:")
            print(f"Original embedding dimension: {first_emb.shape[0]}")
            print(f"Reduced embedding dimension: {reduced_embeddings[next(iter(reduced_embeddings))].shape[0]}")
            
        return reduced_embeddings
    
    def save_embeddings(self, embeddings: dict, save_path: str):
        """
        Saves the embeddings to a file.
        """
        if not save_path.endswith(".npy"):
            raise ValueError("File path must end with '.npy' extension.")
        if os.path.exists(save_path):
            print("File already exists. Will not overwrite.")
            return None
        np.save(save_path, embeddings)
    
    def load_embeddings(self, file_path: str) -> dict:
        """
        Loads embeddings from a file.
        """
        return np.load(file_path, allow_pickle=True).item()
