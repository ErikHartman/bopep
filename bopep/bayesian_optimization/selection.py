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
        self,
        embeddings: dict,
        num_initial: int,
        random_state: int = 42,
        method: str = "kmeans",
    ):
        """
        Select initial peptides via one of two methods:
          - "kmeans": MiniBatchKMeans + nearest-to-centroid
          - "kmeans++": k-means++ seeding only

        Parameters:
        - embeddings: dict {peptide: ndarray}
        - num_initial: int, how many peptides to pick
        - random_state: int, seed
        - method: str, either "kmeans" or "kmeans++"

        Returns:
        - List of peptide sequences
        """
        peptides = list(embeddings.keys())

        # flatten 2D embeddings if needed
        first = next(iter(embeddings.values()))
        if first.ndim == 1:
            X = np.vstack([embeddings[p] for p in peptides])
        else:
            X = np.vstack([embeddings[p].mean(axis=0) for p in peptides])

        if method == "kmeans":
            # --- original MiniBatchKMeans + nearest-to-centroid ---
            kmeans = MiniBatchKMeans(
                n_clusters=num_initial,
                random_state=random_state,
                max_iter=1000,
            )
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

            initial = []
            for cid in range(num_initial):
                idxs = np.where(labels == cid)[0]
                if len(idxs) == 0:  # Skip empty clusters
                    continue
                cluster_pts = X[idxs]
                dists = np.linalg.norm(cluster_pts - centers[cid], axis=1)
                best = idxs[np.argmin(dists)]
                initial.append(peptides[best])

            # If we have fewer than num_initial due to empty clusters,
            # fill the remaining with random selection
            if len(initial) < num_initial:
                remaining_peptides = [p for p in peptides if p not in initial]
                rng = np.random.RandomState(random_state)
                additional = rng.choice(remaining_peptides, 
                                     size=min(num_initial - len(initial), len(remaining_peptides)), 
                                     replace=False)
                initial.extend(additional)

            return initial

        elif method == "kmeans++":
            # --- pure k-means++ seeding ---
            rng = np.random.RandomState(random_state)
            n = X.shape[0]
            centers_idx = np.empty(num_initial, dtype=int)
            closest_sq = np.full(n, np.inf)

            # 1) pick first at random
            centers_idx[0] = rng.randint(n)
            d0 = np.sum((X - X[centers_idx[0]])**2, axis=1)
            closest_sq = np.minimum(closest_sq, d0)

            # 2) sample next with prob ∝ distance^2
            for i in range(1, num_initial):
                probs = closest_sq / closest_sq.sum()
                centers_idx[i] = rng.choice(n, p=probs)
                di = np.sum((X - X[centers_idx[i]])**2, axis=1)
                closest_sq = np.minimum(closest_sq, di)

            return [peptides[i] for i in centers_idx]

        else:
            raise ValueError(f"Unknown method '{method}', choose 'kmeans' or 'kmeans++'")

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
            print("Select next peptides: embedding values are 2D, averaging over the second dimension...")
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


def __main__():
    # Example usage
    selector = PeptideSelector()
    embeddings = {f"pep{i}": np.random.rand(200) for i in range(30000)}
    print(len(embeddings), "peptides loaded.")
    initial_peptides = selector.select_initial_peptides(embeddings, num_initial=500, method="kmeans")
    print("Initial peptides:", initial_peptides)

    initial_peptides = selector.select_initial_peptides(embeddings, num_initial=500, method="kmeans++")
    print("Initial peptides (kmeans++):", initial_peptides)

if __name__ == "__main__":
    __main__()