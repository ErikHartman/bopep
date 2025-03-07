import numpy as np
from sklearn.cluster import MiniBatchKMeans


class PeptideSelector:
    """
    Class to select the initial and subsequent peptides for Bayesian optimization.
    Will reduce the dimensionality of the embeddings if they are 2D.
    """

    def __init__(self):
        pass

    def select_initial_peptides(
        self, embeddings: dict, num_initial: int, random_state: int = 42
    ):
        """
        Select the initial peptides using K-Means clustering.

        Parameters:
        - embeddings: Dictionary {peptide: embedding (ndarray)}.
        - num_initial: Number of peptides to select (clusters).
        - random_state: Seed for reproducible mini-batch K-Means.

        Returns:
        - initial_peptides: List of selected peptide sequences.
        """
        peptides = list(embeddings.keys())
        # Check if embedding values are 1 dimensional or 2 dimensional
        if len(list(embeddings.values())[0].shape) == 1:
            embedding_values = np.array(list(embeddings.values()))
        else:
            print("Embedding values are 2D, averaging over the second dimension...")
            embedding_values = np.array(
                [emb.mean(axis=0) for emb in embeddings.values()]
            )

        kmeans = MiniBatchKMeans(
            n_clusters=num_initial, random_state=random_state, max_iter=1000
        )
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

    def select_next_peptides(self, peptides, acquisition_values, n_select, embeddings):
        """
        Select the next batch of peptides using a "k-center greedy" approach
        weighted by each peptide's acquisition value.

        Algorithm:
        1) Pick the peptide with the highest acquisition value.
        2) For each subsequent selection, compute:
           score(c) = acquisition[c] * min_{s in selected} dist(E(c), E(s))
           Pick the peptide with the highest 'score'.
        3) Repeat until we have 'n_select' peptides or run out of candidates.

        Parameters:
        - peptides: an iterable (e.g., set) of candidate peptide sequences
        - acquisition_values: dict {peptide: float} with the AF for each peptide
        - n_select: how many peptides to select
        - embeddings: dict {peptide: np.ndarray} with each peptide's embedding

        Returns:
        - selected_peptides: a list of length 'n_select' or fewer if fewer remain.
        """

        if len(list(embeddings.values())[0].shape) > 1:
            print("Embedding values are 2D, averaging over the second dimension...")
            embeddings = {
                emb: embeddings[emb].mean(axis=0) for emb in embeddings.keys()
            }

        candidate_list = list(peptides)
        candidate_list.sort(key=lambda p: acquisition_values[p], reverse=True)

        selected = [candidate_list[0]]
        remaining = set(candidate_list[1:])

        while len(selected) < n_select and len(remaining) > 0:
            best_candidate = None
            best_score = -1.0

            for c in remaining:
                distances = [
                    np.linalg.norm(embeddings[c] - embeddings[s]) for s in selected
                ]
                min_dist = min(distances)

                score_c = acquisition_values[c] * min_dist

                if score_c > best_score:
                    best_score = score_c
                    best_candidate = c

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected
