from bopep import Embedder
from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
)

import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    peptides = pd.read_csv("./data/test_data.csv", usecols=["peptide"])[
        "peptide"
    ].tolist()
    logger.info(f"Loaded {len(peptides)} peptides")

    # Initialize embedder
    embedder = Embedder()

    logger.info("Generating AAIndex embeddings")
    aaindex_embeddings = {
        "2D": embedder.embed_aaindex(peptides, average=True),
        "3D": embedder.embed_aaindex(peptides, average=False),
    }
    scores_dict = {peptide: np.random.rand() for peptide in peptides}
    
    
    # MC 2D
    logger.info("Fitting Monte Carlo Dropout model")
    model = MonteCarloDropout(
        input_dim=553,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.1,
        mc_samples=10,
        network_type="mlp",
    )
    loss = model.fit_dict(aaindex_embeddings["2D"], scores_dict=scores_dict)
    print("Loss:", loss)

    # Ensemble 2D
    logger.info("Fitting Neural Network Ensemble model")
    model = NeuralNetworkEnsemble(
        input_dim=553, hidden_dims=[128, 64, 32], n_networks=5, network_type="mlp"
    )
    loss = model.fit_dict(aaindex_embeddings["2D"], scores_dict=scores_dict)
    print("Loss:", loss)

    # Deep Evidential 2D
    logger.info("Fitting Deep Evidential Regression model")
    model = DeepEvidentialRegression(
        input_dim=553, hidden_dims=[128, 64, 32], network_type="mlp"
    )
    loss = model.fit_dict(aaindex_embeddings["2D"], scores_dict=scores_dict)
    print("Loss:", loss)

    # MC 3D
    logger.info("Fitting Monte Carlo Dropout BiLSTM model")
    model = MonteCarloDropout(
        input_dim=553,
        lstm_hidden_dim=128,
        lstm_layers=1,
        dropout_rate=0.1,
        mc_samples=10,
        network_type="bilstm",
    )
    loss = model.fit_dict(aaindex_embeddings["3D"], scores_dict=scores_dict)
    print("Loss:", loss)


    # Ensemble 3D
    logger.info("Fitting Neural Network Ensemble BiLSTM model")
    model = NeuralNetworkEnsemble(
        input_dim=553, lstm_hidden_dim=128, lstm_layers=1, network_type="bilstm"
    )
    loss = model.fit_dict(aaindex_embeddings["3D"], scores_dict=scores_dict)
    print("Loss:", loss)

    # Deep Evidential 3D
    logger.info("Fitting Deep Evidential Regression BiLSTM model")
    model = DeepEvidentialRegression(
        input_dim=553, lstm_hidden_dim=128, lstm_layers=1, network_type="bilstm"
    )
    model.fit_dict(aaindex_embeddings["3D"], scores_dict=scores_dict)
    print("Loss:", loss)


if __name__ == "__main__":
    main()
