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
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Test embedding methods on 2d (with variable lengths to represent peptide lengths) and 1d data.
    Embedding methods include VAE and PCA (in embedder).
    """
    # Load test peptides from CSV
    logger.info("Loading test peptides from CSV...")
    csv_path = "/home/er8813ha/bopep/data/test_data.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    peptides = df['peptide'].tolist()
    logger.info(f"Loaded {len(peptides)} peptides for testing")
    logger.info(f"Peptide length range: {min([len(p) for p in peptides])} to {max([len(p) for p in peptides])}")
    
    # Initialize embedder
    embedder = Embedder()
    
    # Test both sequence and averaged embeddings
    logger.info("Generating embeddings...")
    
    # Generate ESM embeddings - both sequence (2D) and averaged (1D)
    seq_embeddings = embedder.embed_esm(peptides, average=False, model_path=ESM_PATH, batch_size=8, filter=False)
    avg_embeddings = embedder.embed_esm(peptides, average=True, model_path=ESM_PATH, batch_size=8, filter=False)
    
    # Sanity check: Print shapes of embeddings
    logger.info(f"Number of peptides embedded: {len(seq_embeddings)}")
    example_peptide = next(iter(seq_embeddings))
    logger.info(f"Example peptide: {example_peptide}")
    logger.info(f"Shape of sequence embedding: {seq_embeddings[example_peptide].shape}")
    logger.info(f"Shape of averaged embedding: {avg_embeddings[example_peptide].shape}")
    
    # Apply dimension reduction techniques
    logger.info("Reducing dimensions...")
    
    # PCA reduction
    reduced_seq_pca = embedder.reduce_embeddings_pca(seq_embeddings, n_components=10)
    reduced_avg_pca = embedder.reduce_embeddings_pca(avg_embeddings, n_components=10)
    
    # VAE reduction
    reduced_seq_vae = embedder.reduce_embeddings_autoencoder(
        seq_embeddings, latent_dim=10, max_epochs=20, batch_size=8, verbose=False
    )
    reduced_avg_vae = embedder.reduce_embeddings_autoencoder(
        avg_embeddings, latent_dim=10, max_epochs=20, batch_size=8, verbose=False
    )
    
    # Visualization section - Consolidated plots
    logger.info("\nGenerating visualizations...")
    
    # Function to prepare data for visualization
    def prepare_embedding_vectors(embeddings_dict):
        vectors = []
        peptide_list = []
        for peptide, emb in embeddings_dict.items():
            if len(emb.shape) > 1:  # If 2D sequence embedding, take mean along sequence
                vectors.append(np.mean(emb, axis=0))
            else:  # If 1D averaged embedding
                vectors.append(emb)
            peptide_list.append(peptide)
        return np.array(vectors), peptide_list
    
    # Create a combined plot function for comparing methods
    def create_combined_visualization(original, pca_reduced, vae_reduced, is_sequence=True):
        # Prepare data
        orig_X, peptides_list = prepare_embedding_vectors(original)
        pca_X, _ = prepare_embedding_vectors(pca_reduced)
        vae_X, _ = prepare_embedding_vectors(vae_reduced)
        
        # Get peptide lengths for coloring
        lengths = np.array([len(p) for p in peptides_list])
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{'Sequence' if is_sequence else 'Averaged'} Embeddings: Original vs. PCA vs. VAE", fontsize=16)
        
        # Function to apply UMAP and plot
        def apply_umap_and_plot(X, ax, title, lengths):
            if X.shape[0] < 10:  # Not enough samples for meaningful UMAP
                logger.warning(f"Not enough samples for UMAP in {title}")
                ax.text(0.5, 0.5, "Insufficient data", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            try:
                # Configure UMAP
                n_neighbors = min(max(2, X.shape[0] // 5), X.shape[0] - 1)
                umap_model = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, 
                                 min_dist=0.1, metric='euclidean')
                
                # Apply UMAP
                X_umap = umap_model.fit_transform(X)
                
                # Plot with color indicating peptide length
                sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=lengths, cmap='viridis', 
                           alpha=0.8, s=50, edgecolors='k', linewidths=0.2)
                
                # Add color bar indicating peptide length
                if ax == axes[0]:  # Only add colorbar to first subplot
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label('Peptide Length')
                
                # Show a few peptide labels (every nth peptide)
                n = max(1, len(peptides_list) // 10)  # Show ~10 labels
                for i in range(0, len(peptides_list), n):
                    ax.annotate(peptides_list[i][:5]+'...', 
                               (X_umap[i, 0], X_umap[i, 1]), 
                               fontsize=6, alpha=0.7)
                               
                ax.set_title(title)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                
            except Exception as e:
                logger.error(f"Error in UMAP for {title}: {e}")
                ax.text(0.5, 0.5, f"UMAP failed: {str(e)[:20]}...", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(title)
        
        # Apply UMAP and plot for each method
        apply_umap_and_plot(orig_X, axes[0], "Original", lengths)
        apply_umap_and_plot(pca_X, axes[1], "PCA-reduced", lengths)
        apply_umap_and_plot(vae_X, axes[2], "VAE-reduced", lengths)
        
        plt.tight_layout()
        plt.savefig(f"{'sequence' if is_sequence else 'averaged'}_embeddings_comparison.png", dpi=300)
        plt.close()
    
    # Create combined visualizations
    create_combined_visualization(seq_embeddings, reduced_seq_pca, reduced_seq_vae, is_sequence=True)
    create_combined_visualization(avg_embeddings, reduced_avg_pca, reduced_avg_vae, is_sequence=False)
    
    # Create a plot comparing sequence vs. averaged embeddings
    def compare_sequence_vs_averaged():
        # Prepare data
        seq_X, peptides_list = prepare_embedding_vectors(seq_embeddings)
        avg_X, _ = prepare_embedding_vectors(avg_embeddings)
        
        # Get embedding dimension info for title
        seq_dim = seq_embeddings[next(iter(seq_embeddings))].shape[1]
        avg_dim = avg_embeddings[next(iter(avg_embeddings))].shape[0]
        
        # Get peptide lengths for coloring
        lengths = np.array([len(p) for p in peptides_list])
        
        # Create a figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Sequence vs. Averaged Embeddings", fontsize=16)
        
        # Apply UMAP and plot - fixed variable referencing issue
        plot_data = [
            (seq_X, f"Sequence ({seq_dim} dim)", seq_dim),
            (avg_X, f"Averaged ({avg_dim} dim)", avg_dim)
        ]
        
        for i, (X, title, dim) in enumerate(plot_data):
            try:
                # Configure UMAP
                n_neighbors = min(max(2, X.shape[0] // 5), X.shape[0] - 1)
                umap_model = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, 
                                 min_dist=0.1, metric='euclidean')
                
                # Apply UMAP
                X_umap = umap_model.fit_transform(X)
                
                # Plot with color indicating peptide length
                sc = axes[i].scatter(X_umap[:, 0], X_umap[:, 1], c=lengths, cmap='viridis', 
                                  alpha=0.8, s=50, edgecolors='k', linewidths=0.2)
                
                # Add color bar
                if i == 0:
                    cbar = plt.colorbar(sc, ax=axes[i])
                    cbar.set_label('Peptide Length')
                
                axes[i].set_title(title)
                axes[i].set_xlabel('UMAP 1')
                axes[i].set_ylabel('UMAP 2')
                
            except Exception as e:
                logger.error(f"Error in UMAP for {title}: {e}")
                axes[i].text(0.5, 0.5, f"UMAP failed: {str(e)[:20]}...", 
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=8)
                axes[i].set_title(title)
        
        plt.tight_layout()
        plt.savefig("sequence_vs_averaged_comparison.png", dpi=300)
        plt.close()
    
    compare_sequence_vs_averaged()
    
    logger.info("Testing complete. Visualization images saved to current directory.")


if __name__ == "__main__":
    main()
