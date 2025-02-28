from bopep.embedding.embed_esm import embed_esm
from bopep.embedding.embed_aaindex import embed_aaindex
from bopep.embedding.utils import filter_peptides
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from bopep.embedding.dim_red_autoencoder import (
    PeptideAutoencoder,
    PeptideDataset,
    collate_fn,
)


class Embedder:
    def __init__(self):
        pass

    def embed_esm(self, peptides: list, average: bool, model_path: str = None) -> dict:
        peptides = filter_peptides(peptides)
        embeddings = embed_esm(peptides, model_path, average)
        return embeddings

    def embed_aaindex(self, peptides: list, average: bool) -> dict:
        peptides = filter_peptides(peptides)
        embeddings = embed_aaindex(peptides, average)
        return embeddings
    
    def scale_embeddings(self, embeddings: dict) -> dict:
        """
        TODO: Check this function.
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

    def reduce_embeddings_autoencoder(
        self,
        embeddings: dict,
        hidden_dim: int,
        latent_dim: int,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Works if average == False. Reduces the dimensionality of the embeddings using an autoencoder.
        """
        print(f"Using device: {device}")
        
        dataset = PeptideDataset(embeddings)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

        input_dim = embeddings[list(embeddings.keys())[0]].shape[1]
        print("Autoencoder input dimension: ", input_dim)

        autoencoder = PeptideAutoencoder(input_dim, hidden_dim, latent_dim).to(device)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)

        # Use weight decay in optimizer
        optimizer = torch.optim.AdamW(
            autoencoder.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        patience = 5
        min_delta = 0.001
        patience_counter = 0
        best_loss = float("inf")

        print("Training autoencoder...")
        for epoch in range(1000):
            epoch_loss = 0
            batch_count = 0

            for batch, lengths in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed, latent = autoencoder(batch, lengths)
                mask = torch.zeros_like(batch, dtype=torch.float32, device=device)
                for i, length in enumerate(lengths):
                    mask[i, :length, :] = 1.0
                loss = (torch.abs(reconstructed - batch) * mask).sum() / mask.sum()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            scheduler.step(avg_loss)

            print(f"Epoch {epoch}, loss: {avg_loss:.3f}", end="\r")

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"Final loss: {avg_loss:.2f}")
                    break

        reduced_embeddings = {}
        autoencoder.eval()
        
        with torch.no_grad():
            for peptide, emb in embeddings.items():
                emb_tensor = torch.tensor(emb, dtype=torch.float32)
                if len(emb_tensor.shape) == 2:
                    emb_tensor = emb_tensor.unsqueeze(0)  # Add batch dimension
                emb_tensor = emb_tensor.to(device)
                length = torch.tensor([emb_tensor.shape[1]])  # Sequence length
                _, latent = autoencoder(emb_tensor, length)
                reduced_embeddings[peptide] = latent.squeeze(0).cpu().numpy()

        print(
            "The reduced embeddings are of dim: ", reduced_embeddings[peptide].shape
        )

        return reduced_embeddings

    def reduce_embeddings_pca(
        self, embeddings: dict, explained_variance_ratio: float = 0.95
    ):
        """
        Only works if average == True. Reduces the dimensionality of the embeddings using PCA.
        """
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
