import numpy as np
from sklearn.cluster import MiniBatchKMeans


class SequenceSelector:
    """
    Class to select the initial and subsequent sequences for Bayesian optimization.
    Will reduce the dimensionality of the embeddings if they are 2D.
    """

    def __init__(self):
        pass

    def select_initial_sequences(
        self,
        embeddings: dict,
        num_initial: int,
        random_state: int = 42,
        method: str = "kmeans",
    ):
        """
        Select initial sequences via one of three methods:
          - "kmeans": MiniBatchKMeans + nearest-to-centroid
          - "kmeans++": k-means++ seeding only
          - "random": random selection

        Parameters:
        - embeddings: dict {sequence: ndarray}
        - num_initial: int, how many sequences to pick
        - random_state: int, seed
        - method: str, either "kmeans", "kmeans++", or "random"

        Returns:
        - List of sequence sequences
        """
        sequences = list(embeddings.keys())

        # flatten 2D embeddings if needed
        first = next(iter(embeddings.values()))
        if first.ndim == 1:
            X = np.vstack([embeddings[p] for p in sequences])
        else:
            X = np.vstack([embeddings[p].mean(axis=0) for p in sequences])

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
                initial.append(sequences[best])

            # If we have fewer than num_initial due to empty clusters,
            # fill the remaining with random selection
            if len(initial) < num_initial:
                remaining_sequences = [p for p in sequences if p not in initial]
                rng = np.random.RandomState(random_state)
                additional = rng.choice(remaining_sequences, 
                                     size=min(num_initial - len(initial), len(remaining_sequences)), 
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

            return [sequences[i] for i in centers_idx]

        elif method == "random":
            # --- pure random selection ---
            rng = np.random.RandomState(random_state)
            selected_indices = rng.choice(
                len(sequences), 
                size=min(num_initial, len(sequences)), 
                replace=False
            )
            return [sequences[i] for i in selected_indices]

        else:
            raise ValueError(f"Unknown method '{method}', choose 'kmeans', 'kmeans++', or 'random'")

    def select_next_sequences(self, sequences, acquisition_values, n_select, embeddings):
        """
        Select the next batch of sequences using a "k-center greedy" approach
        weighted by each sequence's acquisition value.

        Algorithm:
        1) Pick the sequence with the highest acquisition value.
        2) For each subsequent selection, compute:
           score(c) = acquisition[c] * min_{s in selected} dist(E(c), E(s))
           Pick the sequence with the highest 'score'.
        3) Repeat until we have 'n_select' sequences or run out of candidates.

        Parameters:
        - sequences: an iterable (e.g., set) of candidate sequence sequences
        - acquisition_values: dict {sequence: float} with the AF for each sequence
        - n_select: how many sequences to select
        - embeddings: dict {sequence: np.ndarray} with each sequence's embedding

        Returns:
        - selected_sequences: a list of length 'n_select' or fewer if fewer remain.
        """

        if len(list(embeddings.values())[0].shape) > 1:
            print("Select next sequences: embedding values are 2D, averaging over the second dimension...")
            embeddings = {
                emb: embeddings[emb].mean(axis=0) for emb in embeddings.keys()
            }

        candidate_list = list(sequences)
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
    selector = SequenceSelector()
    embeddings = {f"pep{i}": np.random.rand(200) for i in range(30000)}
    print(len(embeddings), "sequences loaded.")
    initial_sequences = selector.select_initial_sequences(embeddings, num_initial=500, method="kmeans")
    print("Initial sequences:", initial_sequences)

    initial_sequences = selector.select_initial_sequences(embeddings, num_initial=500, method="kmeans++")
    print("Initial sequences (kmeans++):", initial_sequences)

    initial_sequences = selector.select_initial_sequences(embeddings, num_initial=500, method="random")
    print("Initial sequences (random):", initial_sequences)

if __name__ == "__main__":
    __main__()