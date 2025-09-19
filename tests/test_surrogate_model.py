"""
Comprehensive test suite for surrogate models.

This module tests all surrogate model functionality including:

1. Basic model initialization and forward passes
2. Training with both single and multi-objective data
3. Prediction with uncertainty quantification
4. Multi-objective capabilities and tensor shape handling
5. Variable length sequence support (RNN models)
6. Training convergence and loss reduction
7. Prediction format consistency

The training convergence tests use simple polynomial functions:
- Single objective: f(x) = x^3
- Multi-objective: f1(x) = x^3, f2(x) = x^3 + (x^2)^3

These tests help catch regressions in loss calculations, tensor shapes,
and training functionality.
"""

import pytest
import numpy as np
import torch
from typing import Dict, Tuple, Union, List

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
    MVE
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


def generate_multiobjective_data(
    n_samples: int = 100,
    seq_length: int = 15,
    embedding_dim: int = 20,
    n_objectives: int = 3,
    variable_length: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, list]]:
    """Generate synthetic multiobjective data for testing surrogate models."""
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

        # Generate multiple objectives with different correlations to embedding
        objectives = []
        for obj_idx in range(n_objectives):
            if obj_idx == 0:
                # First objective: correlated with embedding mean
                base_score = np.abs(embedding.mean())
            elif obj_idx == 1:
                # Second objective: correlated with embedding std
                base_score = embedding.std() + 1.0  # Add 1 to avoid negative values
            else:
                # Additional objectives: random with some correlation
                base_score = np.abs(embedding.mean()) * (obj_idx * 0.5) + np.random.rand()
            
            noise = np.random.normal(0, 0.1)
            score = float(base_score + noise)
            objectives.append(score)

        embedding_dict[peptide_id] = embedding
        objective_dict[peptide_id] = objectives

    return embedding_dict, objective_dict


def generate_training_data(
    n_samples: int = 50,
    x_range: tuple = (-2, 2),
    multiobjective: bool = False
) -> tuple[Dict[str, np.ndarray], Dict[str, Union[float, List[float]]]]:
    """
    Generate simple synthetic training data for testing loss convergence.
    
    Args:
        n_samples: Number of samples to generate
        x_range: Range for x values
        multiobjective: If True, generate 2 objectives (x^3, x^3 + y^3 where y=x^2)
        
    Returns:
        embedding_dict: Dictionary of embeddings
        objective_dict: Dictionary of objectives
    """
    # Generate x values
    x_values = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    embedding_dict = {}
    objective_dict = {}
    
    for i, x in enumerate(x_values):
        peptide_id = f"sample_{i}"
        
        # Create 1D embedding (just x value)
        embedding_dict[peptide_id] = np.array([x], dtype=np.float32)
        
        if multiobjective:
            # Two objectives: x^3 and x^3 + (x^2)^3
            y = x * x  # y = x^2
            obj1 = x ** 3
            obj2 = x ** 3 + y ** 3
            objective_dict[peptide_id] = [float(obj1), float(obj2)]
        else:
            # Single objective: x^3
            objective_dict[peptide_id] = float(x ** 3)
    
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
        assert isinstance(mc_model, MonteCarloDropout)

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
        assert isinstance(mc_model, MonteCarloDropout)

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
            assert isinstance(predictions, dict)
            assert isinstance(mc_model, MonteCarloDropout)
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
        assert isinstance(der_model, DeepEvidentialRegression)

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


class TestMultiObjectiveModels:
    """Test all surrogate models with multiobjective capabilities."""
    
    def test_multiobjective_mlp_networks(self):
        """Test MLP networks with multiple objectives."""
        # Test with 3 objectives
        model = MLPNetwork(input_dim=20, hidden_dims=[64, 32], n_objectives=3)
        x = torch.randn(10, 20)
        output = model(x)
        assert output.shape == (10, 3, 1)  # Should output [batch_size, n_objectives, output_dim]
        
    def test_multiobjective_rnn_networks(self):
        """Test RNN networks with multiple objectives."""
        model = RNNetwork(input_dim=20, hidden_dim=64, num_layers=1, n_objectives=2)
        
        # Create variable length batch
        x1 = torch.randn(5, 20)
        x2 = torch.randn(7, 20)
        lengths = torch.tensor([5, 7])
        batch = [x1, x2]
        
        padded_batch = pad_sequence(batch, batch_first=True)
        output = model(padded_batch, lengths)
        assert output.shape == (2, 2, 1)  # [batch_size, n_objectives, output_dim]
    
    def test_monte_carlo_dropout_multiobjective(self):
        """Test MonteCarloDropout with multiple objectives."""
        # Generate multiobjective data
        embedding_dict, objective_dict = generate_multiobjective_data(
            n_samples=30, embedding_dim=20, n_objectives=3, variable_length=False
        )
        
        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[32, 16],
            dropout_rate=0.1,
            network_type="mlp",
            n_objectives=3
        )
        
        # Train the model
        loss = mc_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)
        
        # Test predictions
        predictions = mc_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        # Check multiobjective prediction format
        for peptide_id, (means, stds) in predictions.items():
            assert isinstance(means, list)
            assert isinstance(stds, list)
            assert len(means) == 3
            assert len(stds) == 3
            assert all(isinstance(m, float) for m in means)
            assert all(isinstance(s, float) for s in stds)
            assert all(s >= 0 for s in stds)
    
    def test_neural_network_ensemble_multiobjective(self):
        """Test NeuralNetworkEnsemble with multiple objectives."""
        embedding_dict, objective_dict = generate_multiobjective_data(
            n_samples=25, embedding_dim=20, n_objectives=2, variable_length=False
        )
        
        ensemble = NeuralNetworkEnsemble(
            input_dim=20,
            hidden_dims=[32, 16],
            n_networks=2,
            network_type="mlp",
            n_objectives=2
        )
        
        # Train the ensemble
        loss = ensemble.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)
        
        # Test predictions
        predictions = ensemble.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        # Check multiobjective prediction format
        for peptide_id, (means, stds) in predictions.items():
            assert isinstance(means, list)
            assert isinstance(stds, list)
            assert len(means) == 2
            assert len(stds) == 2
            assert all(isinstance(m, float) for m in means)
            assert all(isinstance(s, float) for s in stds)
            assert all(s >= 0 for s in stds)
    
    def test_deep_evidential_regression_multiobjective(self):
        """Test DeepEvidentialRegression with multiple objectives."""
        embedding_dict, objective_dict = generate_multiobjective_data(
            n_samples=30, embedding_dim=20, n_objectives=3, variable_length=False
        )
        
        der_model = DeepEvidentialRegression(
            input_dim=20,
            hidden_dims=[32, 16],
            network_type="mlp",
            n_objectives=3
        )
        
        # Train the model
        loss = der_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)
        
        # Test predictions
        predictions = der_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        # Check multiobjective prediction format
        for peptide_id, (means, stds) in predictions.items():
            assert isinstance(means, list)
            assert isinstance(stds, list)
            assert len(means) == 3
            assert len(stds) == 3
            assert all(isinstance(m, float) for m in means)
            assert all(isinstance(s, float) for s in stds)
            assert all(s >= 0 for s in stds)
    
    def test_mve_multiobjective(self):
        """Test MVE with multiple objectives."""
        embedding_dict, objective_dict = generate_multiobjective_data(
            n_samples=25, embedding_dim=20, n_objectives=2, variable_length=False
        )
        
        mve_model = MVE(
            input_dim=20,
            hidden_dims=[32, 16],
            network_type="mlp",
            n_objectives=2
        )
        
        # Train the model
        loss = mve_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)
        
        # Test predictions
        predictions = mve_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        # Check multiobjective prediction format
        for peptide_id, (means, stds) in predictions.items():
            assert isinstance(means, list)
            assert isinstance(stds, list)
            assert len(means) == 2
            assert len(stds) == 2
            assert all(isinstance(m, float) for m in means)
            assert all(isinstance(s, float) for s in stds)
            assert all(s >= 0 for s in stds)
    
    def test_backward_compatibility(self):
        """Test that all models still work with single objectives (backward compatibility)."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=20, embedding_dim=20, variable_length=False
        )
        
        models = [
            MonteCarloDropout(input_dim=20, hidden_dims=[32], network_type="mlp"),
            NeuralNetworkEnsemble(input_dim=20, hidden_dims=[32], n_networks=2, network_type="mlp"),
            DeepEvidentialRegression(input_dim=20, hidden_dims=[32], network_type="mlp"),
            MVE(input_dim=20, hidden_dims=[32], network_type="mlp")
        ]
        
        for model in models:
            # Train
            loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=1,
                batch_size=8,
                learning_rate=0.001,
                device=device
            )
            assert isinstance(loss, float)
            
            # Predict
            predictions = model.predict_dict(
                embedding_dict=embedding_dict,
                batch_size=8,
                device=device
            )
            
            # Check single objective format
            for peptide_id, (mean, std) in predictions.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_mixed_objective_formats(self):
        """Test handling of mixed single/multi objective formats."""
        # Create mixed format data but ensure consistency for the test
        # All objectives will be converted to lists for n_objectives=2
        embedding_dict = {}
        objective_dict = {}
        
        for i in range(10):
            peptide_id = f"peptide_{i}"
            embedding = np.random.randn(20).astype(np.float32)
            embedding_dict[peptide_id] = embedding
            
            if i < 5:
                # Single objective - but we'll convert to 2-objective format
                single_val = float(np.random.rand())
                objective_dict[peptide_id] = [single_val, single_val]  # Duplicate for 2 objectives
            else:
                # Multi objective
                objective_dict[peptide_id] = [float(np.random.rand()), float(np.random.rand())]
        
        # This should work with n_objectives=2
        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[32],
            network_type="mlp",
            n_objectives=2
        )
        
        # Train (should handle the format)
        loss = mc_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=1,
            batch_size=8,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)


class TestMultiObjectiveWithVariableLength:
    """Test multiobjective models with variable length sequences (RNN)."""
    
    def test_bilstm_multiobjective(self):
        """Test BiLSTM with multiple objectives and variable length sequences."""
        embedding_dict, objective_dict = generate_multiobjective_data(
            n_samples=20, seq_length=15, embedding_dim=20, n_objectives=2, variable_length=True
        )
        
        mc_model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[],
            dropout_rate=0.1,
            network_type="bilstm",
            lstm_layers=1,
            lstm_hidden_dim=32,
            n_objectives=2
        )
        
        # Train
        loss = mc_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=4,
            learning_rate=0.001,
            device=device
        )
        
        assert isinstance(loss, float)
        
        # Predict
        predictions = mc_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=4,
            device=device
        )
        
        # Check multiobjective format
        for peptide_id, (means, stds) in predictions.items():
            assert isinstance(means, list)
            assert isinstance(stds, list)
            assert len(means) == 2
            assert len(stds) == 2


class TestTrainingConvergence:
    """Test that all surrogate models show decreasing loss during training."""
    
    def test_single_objective_convergence(self):
        """Test that all models converge for single objective case."""
        # Generate simple training data
        embedding_dict, objective_dict = generate_training_data(
            n_samples=30, multiobjective=False
        )
        
        models = [
            ("MonteCarloDropout", MonteCarloDropout(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp", dropout_rate=0.1
            )),
            ("NeuralNetworkEnsemble", NeuralNetworkEnsemble(
                input_dim=1, hidden_dims=[16, 8], n_networks=2, network_type="mlp"
            )),
            ("DeepEvidentialRegression", DeepEvidentialRegression(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp"
            )),
            ("MVE", MVE(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp"
            ))
        ]
        
        for model_name, model in models:
            # Get initial loss by training for 1 epoch
            initial_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=1,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Train for more epochs
            final_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=20,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Loss should decrease (or at least not increase significantly)
            assert final_loss <= initial_loss * 1.1, \
                f"{model_name} loss did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_multiobjective_convergence(self):
        """Test that all models converge for multi-objective case."""
        # Generate multi-objective training data
        embedding_dict, objective_dict = generate_training_data(
            n_samples=30, multiobjective=True
        )
        
        models = [
            ("MonteCarloDropout", MonteCarloDropout(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp", 
                dropout_rate=0.1, n_objectives=2
            )),
            ("NeuralNetworkEnsemble", NeuralNetworkEnsemble(
                input_dim=1, hidden_dims=[16, 8], n_networks=2, 
                network_type="mlp", n_objectives=2
            )),
            ("DeepEvidentialRegression", DeepEvidentialRegression(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp", n_objectives=2
            )),
            ("MVE", MVE(
                input_dim=1, hidden_dims=[16, 8], network_type="mlp", n_objectives=2
            ))
        ]
        
        for model_name, model in models:
            # Get initial loss
            initial_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=1,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Train for more epochs
            final_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=20,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Loss should decrease (or at least not increase significantly)
            assert final_loss <= initial_loss * 1.1, \
                f"{model_name} (multiobjective) loss did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_prediction_format_after_training(self):
        """Test that prediction formats are consistent after training."""
        # Single objective test
        embedding_dict, objective_dict = generate_training_data(
            n_samples=20, multiobjective=False
        )
        
        mc_single = MonteCarloDropout(
            input_dim=1, hidden_dims=[8], network_type="mlp", n_objectives=1
        )
        
        mc_single.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=5,
            batch_size=16,
            device=device,
            verbose=False
        )
        
        predictions_single = mc_single.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=16,
            device=device
        )
        
        # Check single objective format
        for peptide_id, (mean, std) in predictions_single.items():
            assert isinstance(mean, float), f"Single objective mean should be float, got {type(mean)}"
            assert isinstance(std, float), f"Single objective std should be float, got {type(std)}"
        
        # Multi-objective test
        embedding_dict_multi, objective_dict_multi = generate_training_data(
            n_samples=20, multiobjective=True
        )
        
        mc_multi = MonteCarloDropout(
            input_dim=1, hidden_dims=[8], network_type="mlp", n_objectives=2
        )
        
        mc_multi.fit_dict(
            embedding_dict=embedding_dict_multi,
            objective_dict=objective_dict_multi,
            epochs=5,
            batch_size=16,
            device=device,
            verbose=False
        )
        
        predictions_multi = mc_multi.predict_dict(
            embedding_dict=embedding_dict_multi,
            batch_size=16,
            device=device
        )
        
        # Check multi-objective format
        for peptide_id, (means, stds) in predictions_multi.items():
            assert isinstance(means, list), f"Multi objective means should be list, got {type(means)}"
            assert isinstance(stds, list), f"Multi objective stds should be list, got {type(stds)}"
            assert len(means) == 2, f"Should have 2 mean values, got {len(means)}"
            assert len(stds) == 2, f"Should have 2 std values, got {len(stds)}"


if __name__ == "__main__":
    # Run tests manually
    print("Running basic model tests...")
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

    print("Running multiobjective tests...")
    test_multi = TestMultiObjectiveModels()
    test_multi.test_multiobjective_mlp_networks()
    test_multi.test_multiobjective_rnn_networks()
    test_multi.test_monte_carlo_dropout_multiobjective()
    test_multi.test_neural_network_ensemble_multiobjective()
    test_multi.test_deep_evidential_regression_multiobjective()
    test_multi.test_mve_multiobjective()
    test_multi.test_backward_compatibility()
    test_multi.test_mixed_objective_formats()
    
    print("Running variable length multiobjective tests...")
    test_var_multi = TestMultiObjectiveWithVariableLength()
    test_var_multi.test_bilstm_multiobjective()

    print("Running training convergence tests...")
    test_conv = TestTrainingConvergence()
    test_conv.test_single_objective_convergence()
    test_conv.test_multiobjective_convergence()
    test_conv.test_prediction_format_after_training()

    print("All surrogate model tests passed!")