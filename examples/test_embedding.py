import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from bopep import Embedder
import pandas as pd
import logging
import torch
from env_config import ESM_PATH
from matplotlib.colors import Normalize
from matplotlib import cm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reduce_with_umap(embeddings, n_neighbors=15, min_dist=0.1):
    """Apply UMAP dimensionality reduction to embeddings."""
    if embeddings.shape[1] > 2:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=min_dist)
        try:
            return reducer.fit_transform(embeddings)
        except Exception as e:
            logger.error(f"UMAP reduction failed: {str(e)}")
            return embeddings[:, :2] if embeddings.shape[1] > 2 else embeddings
    return embeddings


def plot_embedding_grid(results, peptides, method_name, output_filename=None):
    """
    Create a grid of plots showing different dimensionality reduction methods,
    with points colored by peptide length.
    
    Args:
        results: Dictionary mapping reduction method names to reduced embeddings
        peptides: List of peptide sequences for length calculation
        method_name: Name of the embedding method (e.g., "ESM", "AAIndex")
        output_filename: Filename to save the plot (optional)
    """
    # Get peptide lengths for coloring
    peptide_lengths = np.array([len(p) for p in peptides])
    
    # Create color normalization
    norm = Normalize(vmin=min(peptide_lengths), vmax=max(peptide_lengths))
    cmap = cm.viridis
    
    # Determine grid size
    n_methods = len(results)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each method
    for i, (method, embeddings) in enumerate(results.items()):
        ax = axes[i]
        
        # Check if we have any NaN values
        if np.isnan(embeddings).any():
            embeddings = np.nan_to_num(embeddings)
            logger.warning(f"NaNs found in {method} embeddings - replaced with zeros")
        
        # Apply UMAP if needed
        if embeddings.shape[1] > 2:
            logger.info(f"Reducing {method} embeddings with UMAP")
            reduced = reduce_with_umap(embeddings)
        else:
            reduced = embeddings
        
        # Scatter plot with points colored by peptide length
        scatter = ax.scatter(
            reduced[:, 0], 
            reduced[:, 1],
            c=peptide_lengths, 
            cmap=cmap,
            alpha=0.7,
            edgecolor='w',
            linewidth=0.3,
            s=40
        )
        
        # Add title and labels
        ax.set_title(f"{method}\n({embeddings.shape[1]} dims")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=0.2)
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    # Add a colorbar for peptide length
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Peptide Length')
    
    # Add overall title
    fig.suptitle(f"{method_name} Embeddings Comparison", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Make room for colorbar and title
    
    # Save or show
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {output_filename}")
    else:
        plt.show()
    
    plt.close()
    return fig


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    peptides = pd.read_csv("./data/test_data.csv", usecols=["peptide"])[
        "peptide"
    ].tolist()
    logger.info(f"Loaded {len(peptides)} peptides")

    # Initialize embedder
    embedder = Embedder()

    # Generate embeddings
    logger.info("Generating ESM embeddings")
    esm_embeddings = {
        "2D": embedder.embed_esm(peptides, average=True, model_path=ESM_PATH),
        "3D": embedder.embed_esm(peptides, average=False, model_path=ESM_PATH),
    }

    logger.info("Generating AAIndex embeddings")
    aaindex_embeddings = {
        "2D": embedder.embed_aaindex(peptides, average=True),
        "3D": embedder.embed_aaindex(peptides, average=False),
    }

    # Process and visualize embeddings for each method
    for method_name, embeddings in [
        ("ESM", esm_embeddings),
        ("AAIndex", aaindex_embeddings),
    ]:
        # Initial scaled embeddings we'll use for reduction
        scaled_2d = embedder.scale_embeddings(embeddings["2D"])
        scaled_3d = embedder.scale_embeddings(embeddings["3D"])
        
        # Collect results from all methods in a dictionary
        # for the grid visualization
        results = {}
        
        # Original embeddings
        results["Original"] = np.array(list(embeddings["2D"].values()))
        
        # PCA 2D
        logger.info(f"Processing {method_name} with PCA_2D")
        pca_2d_reduced = embedder.reduce_embeddings_pca(scaled_2d)
        results["PCA (2D)"] = np.array(list(pca_2d_reduced.values()))
        
        # PCA 3D
        logger.info(f"Processing {method_name} with PCA_3D")
        pca_3d_reduced = embedder.reduce_embeddings_pca(scaled_3d)
        # Average over peptide lengths
        pca_3d_avg = np.array([emb.mean(axis=0) for emb in pca_3d_reduced.values()])
        results["PCA (3D→Avg)"] = pca_3d_avg
        
        # Standard autoencoder 2D
        logger.info(f"Processing {method_name} with AE_2D")
        first_emb_2d = next(iter(scaled_2d.values()))
        input_dim_2d = first_emb_2d.shape[0]
        latent_dim_2d = min(128, input_dim_2d // 2)
        
        ae_2d_reduced = embedder.reduce_embeddings_autoencoder(
            scaled_2d,
            latent_dim=latent_dim_2d,
            hidden_layers=[input_dim_2d, input_dim_2d // 2],
            max_epochs=100,
            batch_size=32
        )
        results["Autoencoder (2D)"] = np.array(list(ae_2d_reduced.values()))
        
        # Standard autoencoder 3D
        logger.info(f"Processing {method_name} with AE_3D")
        first_emb_3d = next(iter(scaled_3d.values()))
        input_dim_3d = first_emb_3d.shape[1]
        latent_dim_3d = min(128, input_dim_3d // 2)
        
        ae_3d_reduced = embedder.reduce_embeddings_autoencoder(
            scaled_3d, 
            latent_dim=latent_dim_3d,
            hidden_layers=[input_dim_3d, input_dim_3d // 2],
            max_epochs=100,
            batch_size=32
        )
        # Average over peptide lengths
        ae_3d_avg = np.array([emb.mean(axis=0) for emb in ae_3d_reduced.values()])
        results["Autoencoder (3D→Avg)"] = ae_3d_avg
        
        # VAE 2D
        logger.info(f"Processing {method_name} with VAE_2D")
        vae_2d_reduced = embedder.reduce_embeddings_autoencoder(
            scaled_2d,
            latent_dim=latent_dim_2d,
            hidden_layers=[input_dim_2d, input_dim_2d // 2],
            max_epochs=100,
            batch_size=32,
            autoencoder_type="variational"
        )
        results["VAE (2D)"] = np.array(list(vae_2d_reduced.values()))
        
        # VAE 3D
        logger.info(f"Processing {method_name} with VAE_3D")
        vae_3d_reduced = embedder.reduce_embeddings_autoencoder(
            scaled_3d,
            latent_dim=latent_dim_3d,
            hidden_layers=[input_dim_3d, input_dim_3d // 2],
            max_epochs=100,
            batch_size=32,
            autoencoder_type="variational"
        )
        # Average over peptide lengths
        vae_3d_avg = np.array([emb.mean(axis=0) for emb in vae_3d_reduced.values()])
        results["VAE (3D→Avg)"] = vae_3d_avg
        
        # Create and save the grid plot
        output_file = f"{method_name.lower()}_embeddings_comparison.png"
        plot_embedding_grid(results, peptides, method_name, output_file)


if __name__ == "__main__":
    main()
