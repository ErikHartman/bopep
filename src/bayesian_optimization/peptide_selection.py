import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans

from .ensemble import ensemble_predict
from .acquisition import expected_improvement


def select_next_peptides(
    models,
    scaler,
    y_train,
    embeddings,
    docked_peptides,
    batch_size,
    mode
):
    """
    Select the next batch of peptides using either an exploratory or exploitative approach,
    ensuring diversity in the selected peptides while keeping high uncertainty.
    """
    # Filter out peptides that have already been docked
    remaining_peptides = [
        pep for pep in embeddings if pep not in docked_peptides
    ]

    # Create feature vectors by concatenating embedding with peptide length
    X_remaining = []
    for pep in remaining_peptides:
        embedding = embeddings[pep]
        peptide_length = len(pep)
        feature_vector = np.concatenate((embedding, [peptide_length]))
        X_remaining.append(feature_vector)

    X_remaining = np.array(X_remaining)
    mu, sigma = ensemble_predict(models, scaler, X_remaining)

    if mode == "exploration":
        # Exploration: Use a combination of uncertainty (sigma) and diversity (distance) using k-Center Greedy
        selected_indices = k_center_greedy_with_scores(X_remaining, sigma, batch_size)
    elif mode == "exploitation":
        # Exploitation: Select peptides using Expected Improvement (EI)
        best_score = max(y_train)
        acquisition_scores = expected_improvement(mu, sigma, best_score)
        selected_indices = np.argsort(acquisition_scores)[-batch_size:]  # pure EI
    elif mode == "exploitation_weighted":
        best_score = max(y_train)
        acquisition_scores = expected_improvement(mu, sigma, best_score)
        selected_indices = k_center_greedy_with_scores(
            X_remaining, acquisition_scores, batch_size
        )
    elif mode == "exploitation_pure":
        selected_indices = np.argsort(mu)[-batch_size:]
    else:
        raise ValueError(
            "Invalid mode. Choose either 'exploration' or 'exploitation' or 'exploitation_weighted'."
        )

    next_peptides = [remaining_peptides[idx] for idx in selected_indices]

    return next_peptides


def k_center_greedy_with_scores(X, score, num_samples):
    """
    Perform k-Center Greedy selection to ensure diversity and also prioritize high score.

    Parameters:
    - X: Array of feature vectors.
    - score: Array of uncertainty scores for the input samples.
    - num_samples: Number of samples to select.

    Returns:
    - selected_indices: Indices of the selected diverse and high-score samples.
    """
    num_points = X.shape[0]
    selected_indices = []

    # Start with the point that has the highest uncertainty (sigma)
    initial_index = np.argmax(score)
    selected_indices.append(initial_index)

    # Initialize distances to infinity
    distances = np.full(num_points, np.inf)

    # Iteratively select samples that are maximally distant and have high uncertainty
    for _ in range(num_samples - 1):
        current_point = X[selected_indices[-1]].reshape(1, -1)
        dist = cdist(X, current_point, metric="euclidean").reshape(-1)
        distances = np.minimum(distances, dist)
        combined_score = distances * score  # Here is the key
        next_index = np.argmax(combined_score)
        selected_indices.append(next_index)

    return selected_indices


def select_initial_peptides(embeddings, num_samples):
    """
    Select the initial peptides using K-Means clustering to ensure diversity.

    Parameters:
    - embeddings: Dictionary of peptide embeddings.
    - num_samples: Number of peptides to select (number of clusters).

    Returns:
    - initial_peptides: List of selected peptide sequences.
    """

    peptides = list(embeddings.keys())
    embedding_values = np.array(list(embeddings.values()))

    kmeans = MiniBatchKMeans(n_clusters=num_samples, random_state=42, max_iter=1000)
    cluster_labels = kmeans.fit_predict(embedding_values)

    initial_peptides = []
    for cluster_id in range(num_samples):
        # Get indices of peptides in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_embeddings = embedding_values[cluster_indices]

        # Compute distances to the cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Find the index of the peptide closest to the centroid
        closest_idx_in_cluster = np.argmin(distances)
        closest_index = cluster_indices[closest_idx_in_cluster]

        initial_peptides.append(peptides[closest_index])

    return initial_peptides
