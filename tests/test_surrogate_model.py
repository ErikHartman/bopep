import pytest
import numpy as np
import torch
from typing import Dict, Tuple

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression
)
from bopep.surrogate_model.helpers import pad_sequence
from bopep.surrogate_model.base_models import RNNetwork, MLPNetwork

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device("cpu")


def generate_synthetic_data(
    n_samples: int = 100, 
    seq_length: int = 15, 
    embedding_dim: int = 20, 
    variable_length: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Generate synthetic data for testing surrogate models."""
    embedding_dict = {}
    objective_dict = {}

    for i in range(n_samples):
        peptide_id = f"peptide_{i}"

        if variable_length:
            # For variable length, create embeddings with different sequence lengths
            actual_length = np.random.randint(5, seq_length + 1)
            embedding = np.random.randn(actual_length, embedding_dim).astype(np.float32)
        else:
            # For fixed length, use average embedding
            embedding = np.random.randn(embedding_dim).astype(np.float32)

        # Generate score with correlation to embedding mean
        base_score = np.abs(embedding.mean())
        noise = np.random.normal(0, 0.5)
        score = base_score + noise

        embedding_dict[peptide_id] = embedding
        objective_dict[peptide_id] = float(score)

    return embedding_dict, objective_dict


class TestMLPNetwork:
    def test_initialization(self):
        model = MLPNetwork(input_dim=20, hidden_dims=[64, 32])
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        model = MLPNetwork(input_dim=20, hidden_dims=[64, 32])
        x = torch.randn(10, 20)  # Batch of 10, dim 20
        output = model(x)
        assert output.shape == (10, 1)


class TestRNNetwork:
    def test_initialization(self):
        model = RNNetwork(input_dim=20, hidden_dim=64, num_layers=2)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_with_variable_length(self):
        model = RNNetwork(input_dim=20, hidden_dim=64, num_layers=1)

        # Create variable length batch
        x1 = torch.randn(5, 20)  # Seq length 5
        x2 = torch.randn(7, 20)  # Seq length 7
        x3 = torch.randn(3, 20)  # Seq length 3

        lengths = torch.tensor([5, 7, 3])
        batch = [x1, x2, x3]

        # Pad and pack batch
        padded_batch = pad_sequence(batch, batch_first=True)
        output = model(padded_batch, lengths)
        assert output.shape == (3, 1)


class TestMonteCarloDropout:
    def test_initialization(self):
        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            network_type="mlp"
        )

    def test_fit_dict_mlp(self):
        # Generate fixed-length data for MLP
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=50, seq_length=10, embedding_dim=20, variable_length=False
        )

        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            network_type="mlp"
        )

        loss = mc_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            device=device
        )

        assert isinstance(loss, float)

    def test_predict_dict_mlp(self):
        # Generate fixed-length data for MLP
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=30, seq_length=10, embedding_dim=20, variable_length=False
        )

        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[64, 32],
            dropout_rate=0.1,
            network_type="mlp"
        )

        # First fit the model
        mc_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            device=device
        )

        # Test prediction
        predictions = mc_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=16,
            device=device
        )

        # Check prediction format and contents
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)

        for peptide_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0  # Standard deviation should be non-negative

    def test_rnn_model(self):
        # Generate variable-length data for RNN
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=30, seq_length=15, embedding_dim=20, variable_length=True
        )

        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[],
            dropout_rate=0.1,
            network_type="bilstm",
            lstm_layers=1,
            lstm_hidden_dim=64
        )

        # Try to fit the model (just ensure it runs)
        try:
            mc_model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=2,
                batch_size=8,
                learning_rate=0.001,
                device=device
            )
            predictions = mc_model.predict_dict(
                embedding_dict=embedding_dict,
                batch_size=8,
                device=device
            )
            assert True
        except Exception as e:
            pytest.fail(f"BiLSTM training/prediction failed with error: {e}")


class TestNeuralNetworkEnsemble:
    def test_initialization(self):
        ensemble = NeuralNetworkEnsemble(
            input_dim=20,
            hidden_dims=[64, 32],
            n_networks=3,
            network_type="mlp"
        )
        assert len(ensemble.networks) == 3

    def test_predict_with_uncertainty(self):
        # Generate fixed-length data for MLP
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=30, seq_length=10, embedding_dim=20, variable_length=False
        )

        ensemble = NeuralNetworkEnsemble(
            input_dim=20,
            hidden_dims=[64, 32],
            n_networks=3,
            network_type="mlp"
        )

        # Train the ensemble
        ensemble.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=16,
            learning_rate=0.001,
            device=device
        )

        # Test prediction
        predictions = ensemble.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=16,
            device=device
        )

        # Check prediction format and contents
        for peptide_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0


class TestDeepEvidentialRegression:
    def test_initialization(self):
        der_model = DeepEvidentialRegression(
            input_dim=20,
            hidden_dims=[64, 32],
            network_type="mlp"
        )

    def test_evidential_loss(self):
        # Simple test to ensure loss function doesn't crash
        der_model = DeepEvidentialRegression(
            input_dim=20,
            hidden_dims=[64, 32],
            network_type="mlp"
        )

        y_true = torch.tensor([[1.0], [2.0], [3.0]])
        evidential_output = torch.ones((3, 4))  # mu, v, alpha, beta

        # Pass tensors as named arguments
        loss = der_model.evidential_loss(
            mu=evidential_output[:, 0],
            v=evidential_output[:, 1],
            alpha=evidential_output[:, 2],
            beta=evidential_output[:, 3],
            targets=y_true
        )
        assert isinstance(loss.item(), float)

    def test_fit_and_predict(self):
        # Generate fixed-length data for MLP
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=30, seq_length=10, embedding_dim=20, variable_length=False
        )

        der_model = DeepEvidentialRegression(
            input_dim=20,
            hidden_dims=[64, 32],
            network_type="mlp"
        )

        # Train the model
        der_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            device=device
        )

        # Test prediction
        predictions = der_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=16,
            device=device
        )

        # Check prediction format and uncertainty estimation
        for peptide_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0  # Standard deviation should be non-negative


if __name__ == "__main__":
    # Run tests manually
    test_mlp = TestMLPNetwork()
    test_mlp.test_initialization()
    test_mlp.test_forward_pass()

    test_rnn = TestRNNetwork()
    test_rnn.test_initialization()
    test_rnn.test_forward_pass_with_variable_length()

    test_mc = TestMonteCarloDropout()
    test_mc.test_initialization()
    test_mc.test_fit_dict_mlp()
    test_mc.test_predict_dict_mlp()
    test_mc.test_rnn_model()

    test_ensemble = TestNeuralNetworkEnsemble()
    test_ensemble.test_initialization()
    test_ensemble.test_predict_with_uncertainty()   

    test_der = TestDeepEvidentialRegression()
    test_der.test_initialization()
    test_der.test_evidential_loss()
    test_der.test_fit_and_predict()


    print("All surrogate model tests passed!")