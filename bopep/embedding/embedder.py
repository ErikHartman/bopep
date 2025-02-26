from bopep.embedding.embed_esm import embed_esm
from bopep.embedding.utils import filter_peptides
from sklearn.decomposition import PCA
import numpy as np


class Embedder:
    def __init__(self):
        pass

    def embed_esm(self, peptides : list, model_path : str = None) -> dict:
        peptides = filter_peptides(peptides)
        embeddings = embed_esm(peptides, model_path)
        return embeddings

    def reduce_embeddings_pca(self, embeddings: dict, explained_variance_ratio : float=0.95):
        embedding_array = np.array(list(embeddings.values()))
        peptide_sequences = list(embeddings.keys())

        pca = PCA(n_components=explained_variance_ratio, svd_solver="full")
        embeddings_reduced = pca.fit_transform(embedding_array)

        peptide_embeddings_reduced = {
            peptide_sequences[i]: embeddings_reduced[i]
            for i in range(len(peptide_sequences))
        }

        return peptide_embeddings_reduced
