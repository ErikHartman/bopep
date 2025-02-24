import numpy as np
import time
import logging
from joblib import Parallel, delayed

from .model_training import train_single_model


def train_deep_ensemble_parallel(X_scaled, y, num_models=5, params=None, threads=2):

    # Set n_jobs appropriately; Colab may not benefit from using too many threads
    n_jobs = min(num_models, threads)  # Use up to 2 threads

    start_time = time.time()
    logging.info(
        f"Training deep ensemble with {num_models} models using threading backend..."
    )

    # Train models in parallel using threading backend
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(train_single_model)(X_scaled, y, i, params) for i in range(num_models)
    )

    models, val_scores = zip(*results)

    # Log validation score for each model
    for i, val_score in enumerate(val_scores):
        logging.info(f"Model {i+1} validation R² score: {val_score:.4f}")

    val_scores_array = np.array(val_scores)
    mean_val_score = np.mean(val_scores_array)
    std_val_score = np.std(val_scores_array)

    logging.info(
        f"Mean validation R² score: {mean_val_score:.3f}, std: {std_val_score:.2f}"
    )

    end_time = time.time()
    total_duration = end_time - start_time
    logging.info(f"Deep ensemble training completed in {total_duration:.2f} seconds.")

    return models, val_scores


def ensemble_predict(models, scaler, X_new):
    # Scale the new peptides using the existing scaler
    X_new_scaled = scaler.transform(X_new)
    predictions = np.array([model.predict(X_new_scaled) for model in models])
    mu = np.mean(predictions, axis=0)
    sigma = np.std(predictions, axis=0)
    return mu, sigma
