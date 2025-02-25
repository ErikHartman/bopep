import numpy as np
from sklearn.cluster import MiniBatchKMeans


class PeptideSelector:
    def __init__(self):
        pass

    def select_initial_peptides(embeddings: dict, num_initial: int, random_state: int = None):
        """
        Select the initial peptides using K-Means clustering.

        Parameters:
        - embeddings: Dictionary of peptide embeddings.
        - num_samples: Number of peptides to select (number of clusters).

        Returns:
        - initial_peptides: List of selected peptide sequences.
        """

        peptides = list(embeddings.keys())
        embedding_values = np.array(list(embeddings.values()))

        kmeans = MiniBatchKMeans(n_clusters=num_initial, random_state=random_state, max_iter=1000)
        cluster_labels = kmeans.fit_predict(embedding_values)

        initial_peptides = []
        for cluster_id in range(num_initial):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embedding_values[cluster_indices]

            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            closest_idx_in_cluster = np.argmin(distances)
            closest_index = cluster_indices[closest_idx_in_cluster]

            initial_peptides.append(peptides[closest_index])

        return initial_peptides
    
    def select_next_peptides(self):
        pass
