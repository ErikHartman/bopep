import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from bopep import Embedder
import pandas as pd

def plot_embeddings(embeddings, method_name, reduction_type):
    """
    Visualizes embeddings using PCA or t-SNE.

    :param embeddings: Numpy array of shape (num_samples, num_features)
    :param method_name: String, name of the embedding method (e.g., "ESM", "AAIndex")
    :param reduction_type: String, type of reduction ("Original", "Autoencoder", "PCA", "t-SNE")
    :param color: String, color for plotting
    """
    if embeddings.shape[1] > 2:
        print(f"Reducing {method_name} embeddings using PCA for visualization...")
        umap = UMAP(n_components=2)
        reduced = umap.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, label=f"{method_name} ({reduction_type})")
    plt.title(f"{method_name} - {reduction_type} Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{method_name}_{reduction_type}.png")

if __name__ == "__main__":
    peptides = pd.read_csv("./data/test_data.csv")["peptide"].tolist()
    embedder = Embedder()

    # Generate embeddings
    esm_2d = embedder.embed_esm(peptides, average=True)
    aaindex_2d = embedder.embed_aaindex(peptides, average=True)
    esm_3d = embedder.embed_esm(peptides, average=False)
    aaindex_3d = embedder.embed_aaindex(peptides, average=False)

    # Convert to numpy
    esm_2d_np = np.array(list(esm_2d.values()))
    aaindex_2d_np = np.array(list(aaindex_2d.values()))
    esm_3d_np = np.array(list(esm_3d.values()))
    aaindex_3d_np = np.array(list(aaindex_3d.values()))

    # Plot original embeddings
    plot_embeddings(esm_2d_np, "ESM", "Original", color="red")
    plot_embeddings(aaindex_2d_np, "AAIndex", "Original", color="blue")

    # Autoencoder Reduction
    print("Reducing AAIndex with Autoencoder")
    aaindex_auto = embedder.reduce_embeddings_autoencoder(embedder.scale_embeddings(aaindex_3d), 256, 128)
    print("Reducing ESM with Autoencoder")
    esm_auto = embedder.reduce_embeddings_autoencoder(embedder.scale_embeddings(esm_3d), 256, 128)

    # Convert to numpy
    aaindex_auto_np = np.array(list(aaindex_auto.values()))
    esm_auto_np = np.array(list(esm_auto.values()))

    # Plot autoencoder-reduced embeddings
    plot_embeddings(aaindex_auto_np, "AAIndex", "Autoencoder", color="blue")
    plot_embeddings(esm_auto_np, "ESM", "Autoencoder", color="red")

    # PCA Reduction
    print("Reducing AAIndex with PCA")
    aaindex_pca = embedder.reduce_embeddings_pca(embedder.scale_embeddings(aaindex_2d))
    print("Reducing ESM with PCA")
    esm_pca = embedder.reduce_embeddings_pca(embedder.scale_embeddings(esm_2d))

    # Convert to numpy
    aaindex_pca_np = np.array(list(aaindex_pca.values()))
    esm_pca_np = np.array(list(esm_pca.values()))

    # Plot PCA-reduced embeddings
    plot_embeddings(aaindex_pca_np, "AAIndex", "PCA", color="blue")
    plot_embeddings(esm_pca_np, "ESM", "PCA", color="red")
