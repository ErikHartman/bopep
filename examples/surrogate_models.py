import torch
import numpy as np
from typing import Dict, Tuple
import logging
import time


from bopep.surrogate_model.mc_dropout import MonteCarloDropout
from bopep.surrogate_model.nn_ensemble import NeuralNetworkEnsemble
from bopep.surrogate_model.deep_evidential_regression import DeepEvidentialRegression


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def generate_synthetic_data(
    n_samples: int = 100,
    seq_length: int = 15,
    embedding_dim: int = 20,
    variable_length: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Generate synthetic peptide embeddings and scores for testing.

    Args:
        n_samples: Number of peptides to generate
        seq_length: Maximum sequence length
        embedding_dim: Embedding dimension
        variable_length: If True, generate variable length sequences

    Returns:
        embedding_dict: Dictionary mapping peptide IDs to embeddings
        scores_dict: Dictionary mapping peptide IDs to scores
    """
    embedding_dict = {}
    scores_dict = {}

    for i in range(n_samples):
        peptide_id = f"peptide_{i}"

        # Generate random length if variable length
        if variable_length:
            actual_length = np.random.randint(3, seq_length + 1)
        else:
            actual_length = seq_length

        # Generate random embedding - for variable length, shape is (L, D)
        if variable_length:
            embedding = np.random.randn(actual_length, embedding_dim).astype(np.float32)
        else:
            # For fixed length, use average embedding (shape D)
            embedding = np.random.randn(embedding_dim).astype(np.float32)

        # Generate score - add some correlation with embedding mean
        base_score = np.abs(embedding.mean())
        noise = np.random.normal(0, 0.5)
        score = base_score + noise

        embedding_dict[peptide_id] = embedding
        scores_dict[peptide_id] = float(score)

    return embedding_dict, scores_dict


def train_and_evaluate_model(
    model_type: str,
    network_type: str,
    embedding_dict: Dict[str, np.ndarray],
    scores_dict: Dict[str, float],
    test_ratio: float = 0.2,
):
    """
    Train and evaluate a surrogate model.

    Args:
        model_type: Type of uncertainty model ('mc_dropout', 'ensemble', 'evidential')
        network_type: Type of network architecture ('mlp', 'bilstm', 'bigru')
        embedding_dict: Dictionary of peptide embeddings
        scores_dict: Dictionary of peptide scores
        test_ratio: Fraction of data to use for testing
    """
    # Split data into train and test
    peptides = list(embedding_dict.keys())
    np.random.shuffle(peptides)

    split_idx = int(len(peptides) * (1 - test_ratio))
    train_peptides = peptides[:split_idx]
    test_peptides = peptides[split_idx:]

    train_embedding_dict = {p: embedding_dict[p] for p in train_peptides}
    train_scores_dict = {p: scores_dict[p] for p in train_peptides}

    test_embedding_dict = {p: embedding_dict[p] for p in test_peptides}
    test_scores_dict = {p: scores_dict[p] for p in test_peptides}

    # Get input dimension
    sample_embedding = next(iter(embedding_dict.values()))
    if len(sample_embedding.shape) == 1:  # Fixed length (D,)
        input_dim = sample_embedding.shape[0]
    else:  # Variable length (L, D)
        input_dim = sample_embedding.shape[1]

    # Common parameters
    common_params = {
        "input_dim": input_dim,
        "hidden_dims": [128, 64],
        "hidden_dim": 128,
        "num_layers": 2 if network_type in ["bilstm", "bigru"] else 1,
    }

    if model_type == "mc_dropout":
        model = MonteCarloDropout(
            dropout_rate=0.1, mc_samples=20, network_type=network_type, **common_params
        )
    elif model_type == "ensemble":
        model = NeuralNetworkEnsemble(
            n_networks=5, network_type=network_type, **common_params
        )
    elif model_type == "evidential":
        model = DeepEvidentialRegression(
            network_type=network_type, evidential_regularization=0.1, **common_params
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Move model to device
    model = model.to(device)
    logger.info(f"Created {model_type.upper()} model with {network_type} network")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set device for training
    try:
        # Train model with device specified
        start_time = time.time()
        train_loss = model.fit_dict(
            embedding_dict=train_embedding_dict,
            scores_dict=train_scores_dict,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            device=device,  # Pass device to fit_dict
        )
        training_time = time.time() - start_time
        logger.info(
            f"Training completed in {training_time:.2f} seconds with final loss: {train_loss:.4f}"
        )

        # Evaluate model with device specified
        predictions = model.predict_dict(
            embedding_dict=test_embedding_dict,
            batch_size=32,
            device=device,  # Pass device to predict_dict
        )
    except TypeError:
        # Fallback if the model doesn't accept device parameter
        logger.warning(
            "Model doesn't accept device parameter. Modifying helpers.py is recommended."
        )
        start_time = time.time()
        train_loss = model.fit_dict(
            embedding_dict=train_embedding_dict,
            scores_dict=train_scores_dict,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
        )
        training_time = time.time() - start_time
        logger.info(
            f"Training completed in {training_time:.2f} seconds with final loss: {train_loss:.4f}"
        )

        # Evaluate model
        predictions = model.predict_dict(test_embedding_dict)

    # Calculate metrics
    true_values = np.array([test_scores_dict[p] for p in test_peptides])
    pred_means = np.array([predictions[p][0] for p in test_peptides])
    pred_stds = np.array([predictions[p][1] for p in test_peptides])

    mse = np.mean((true_values - pred_means) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_values - pred_means))

    # Calculate uncertainty calibration
    z_scores = np.abs(true_values - pred_means) / pred_stds
    well_calibrated = np.mean(z_scores <= 2)  # Percentage within 2 std

    logger.info(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    logger.info(f"Calibration (% within 2σ): {well_calibrated * 100:.1f}%")


def main():
    """Run comprehensive tests of surrogate models."""
    logger.info("Starting surrogate model tests")

    results = []

    # Test with fixed-length data (simulates averaged embeddings)
    logger.info("Testing with fixed-length embeddingss")
    fixed_embeddings, fixed_scores = generate_synthetic_data(
        n_samples=200, seq_length=10, embedding_dim=20, variable_length=False
    )

    # Test with variable-length data
    logger.info("Testing with variable-length embeddings")
    var_embeddings, var_scores = generate_synthetic_data(
        n_samples=200, seq_length=10, embedding_dim=20, variable_length=True
    )

    # Test different model configurations with fixed-length data
    for model_type in ["mc_dropout", "ensemble", "evidential"]:
        # MLP only works with fixed-length data
        result = train_and_evaluate_model(
            model_type=model_type,
            network_type="mlp",
            embedding_dict=fixed_embeddings,
            scores_dict=fixed_scores,
        )
        results.append(result)

    # Test different model configurations with variable-length data
    for model_type in ["mc_dropout", "ensemble", "evidential"]:
        for network_type in ["bilstm", "bigru"]:
            # Test standard architecture
            result = train_and_evaluate_model(
                model_type=model_type,
                network_type=network_type,
                embedding_dict=var_embeddings,
                scores_dict=var_scores,
            )
            results.append(result)

    logger.info("All tests completed")


if __name__ == "__main__":
    main()
