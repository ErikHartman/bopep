import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from bopep import Embedder
import pandas as pd
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_embeddings(embeddings: np.ndarray, method_name: str, reduction_type: str) -> None:
    """
    Visualize embeddings using UMAP dimensionality reduction.
    
    Args:
        embeddings: Numpy array of embeddings (n_samples x n_features)
        method_name: Name of the embedding method (e.g., "ESM", "AAIndex")
        reduction_type: Type of reduction applied (e.g., "Original", "Autoencoder")
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Check and handle empty/nan values
        if np.isnan(embeddings).any():
            embeddings = np.nan_to_num(embeddings)
            logger.warning(f"NaNs found in {method_name} {reduction_type} embeddings - replaced with zeros")

        # Reduce dimensionality if needed
        if embeddings.shape[1] > 2:
            logger.info(f"Reducing {method_name}-{reduction_type} embeddings with UMAP")
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            reduced = reducer.fit_transform(embeddings)
        else:
            reduced = embeddings[:, :2]  # Use first two dimensions if <= 2D

        # Create plot
        plt.scatter(reduced[:, 0], reduced[:, 1], 
                   alpha=0.7, 
                   edgecolor='w', 
                   linewidth=0.3,
                   cmap='viridis')
        
        plt.title(f"{method_name} Embeddings ({reduction_type})\n"
                  f"Shape: {embeddings.shape} → UMAP: {reduced.shape}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        # Save and close
        filename = f"{method_name.lower()}_{reduction_type.lower()}_embedding.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {filename}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot {method_name} {reduction_type}: {str(e)}")
        plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Load data
        peptides = pd.read_csv("./data/test_data.csv", usecols=["peptide"])["peptide"].tolist()
        logger.info(f"Loaded {len(peptides)} peptides")
        
        # Initialize embedder
        embedder = Embedder()
        
        # Generate embeddings
        logger.info("Generating ESM embeddings")
        esm_embeddings = {
            "2D": embedder.embed_esm(peptides, average=True),
            "3D": embedder.embed_esm(peptides, average=False)
        }
        
        logger.info("Generating AAIndex embeddings")
        aaindex_embeddings = {
            "2D": embedder.embed_aaindex(peptides, average=True),
            "3D": embedder.embed_aaindex(peptides, average=False)
        }

        # Process and visualize embeddings
        for method_name, embeddings in [("ESM", esm_embeddings), ("AAIndex", aaindex_embeddings)]:
            # Original embeddings
            plot_embeddings(np.array(list(embeddings["2D"].values())), 
                           method_name, "Original")

            # Autoencoder processing
            logger.info(f"Processing {method_name} with Autoencoder")
            scaled = embedder.scale_embeddings(embeddings["3D"])
            autoencoder_reduced = embedder.reduce_embeddings_autoencoder(
                scaled, hidden_dim=256, latent_dim=128
            )
            # Average over peptide lengths before plotting
            averaged_embeddings = np.array([embedding.mean(axis=0) for embedding in autoencoder_reduced.values()])
            plot_embeddings(averaged_embeddings,
                           method_name, "Autoencoder")

            # PCA processing
            logger.info(f"Processing {method_name} with PCA")
            scaled_2d = embedder.scale_embeddings(embeddings["2D"])
            pca_reduced = embedder.reduce_embeddings_pca(scaled_2d)
            plot_embeddings(np.array(list(pca_reduced.values())),
                           method_name, "PCA")

    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()