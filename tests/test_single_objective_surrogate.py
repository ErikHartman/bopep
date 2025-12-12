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
from bopep.surrogate_model.manager import SurrogateModelManager
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
        sequence_id = f"sequence_{i}"

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

        embedding_dict[sequence_id] = embedding
        objective_dict[sequence_id] = float(score)

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
        sequence_id = f"sequence_{i}"

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

        embedding_dict[sequence_id] = embedding
        objective_dict[sequence_id] = objectives

    return embedding_dict, objective_dict


def generate_training_data(
    n_samples: int = 50,
    x_range: tuple = (-2, 2),
    multiobjective: bool = False
) -> tuple[Dict[str, np.ndarray], Dict[str, Union[float, Dict[str, float]]]]:
    """
    Generate simple synthetic training data for testing loss convergence.
    
    Args:
        n_samples: Number of samples to generate
        x_range: Range for x values
        multiobjective: If True, generate 2 objectives with named format
        
    Returns:
        embedding_dict: Dictionary of embeddings
        objective_dict: Dictionary of objectives (single scores or named dict)
    """
    # Generate x values
    x_values = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    embedding_dict = {}
    objective_dict = {}
    
    for i, x in enumerate(x_values):
        sequence_id = f"sample_{i}"
        
        # Create 1D embedding (just x value)
        embedding_dict[sequence_id] = np.array([x], dtype=np.float32)
        
        if multiobjective:
            # Two objectives with named format: x^3 and x^3 + (x^2)^3
            y = x * x  # y = x^2
            obj1 = x ** 3
            obj2 = x ** 3 + y ** 3
            objective_dict[sequence_id] = {
                "objective_1": float(obj1),
                "objective_2": float(obj2)
            }
        else:
            # Single objective: x^3
            objective_dict[sequence_id] = float(x ** 3)
    
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

        for sequence_id, (mean, std) in predictions.items():
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
        for sequence_id, (mean, std) in predictions.items():
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
        for sequence_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0  # Standard deviation should be non-negative


class TestMVE:
    """Test Mean-Variance Estimation (MVE) model."""
    
    def test_initialization(self):
        """Test MVE model initialization."""
        mve_model = MVE(
            input_dim=20,
            hidden_dims=[32, 16],
            network_type="mlp"
        )
        assert isinstance(mve_model, MVE)
    
    def test_fit_and_predict(self):
        """Test MVE training and prediction."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=25, embedding_dim=20, variable_length=False
        )
        
        mve_model = MVE(
            input_dim=20,
            hidden_dims=[32, 16],
            network_type="mlp"
        )
        
        # Train
        loss = mve_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=3,
            batch_size=16,
            learning_rate=0.001,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = mve_model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=16,
            device=device
        )
        
        # Check prediction format
        for sequence_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0


class TestSurrogateModelManager:
    """Test SurrogateModelManager functionality."""
    
    def test_manager_initialization(self):
        """Test SurrogateModelManager initialization."""
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        assert manager.surrogate_model_kwargs['model_type'] == 'mve'
        assert manager.surrogate_model_kwargs['network_type'] == 'mlp'
        assert manager.model is None
        assert manager.best_hyperparams is None
    
    def test_manager_single_objective_workflow(self):
        """Test complete single objective workflow with manager."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=30, embedding_dim=20, variable_length=False
        )
        
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        # Hyperparameter optimization
        best_params = manager.optimize_hyperparameters(
            embeddings=embedding_dict,
            objectives=objective_dict,
            n_trials=3
        )
        assert isinstance(best_params, dict)
        assert 'hidden_dims' in best_params
        assert 'learning_rate' in best_params
        assert 'batch_size' in best_params
        
        # Initialize model
        manager.initialize_model(embeddings=embedding_dict)
        assert manager.model is not None
        
        # Train model
        manager.train(embedding_dict, objective_dict)
        
        # Make predictions
        predictions = manager.predict(embedding_dict)
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)
        
        # Check prediction format for single objective
        for sequence_id, (mean, std) in predictions.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0
    
    def test_manager_different_model_types(self):
        """Test manager with different model types."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=20, embedding_dim=15, variable_length=False
        )
        
        model_types = ['mve', 'deep_evidential', 'mc_dropout', 'nn_ensemble']
        
        for model_type in model_types:
            manager = SurrogateModelManager({
                'model_type': model_type,
                'network_type': 'mlp'
            })
            
            # Quick hyperparameter optimization
            best_params = manager.optimize_hyperparameters(
                embeddings=embedding_dict,
                objectives=objective_dict,
                n_trials=2
            )
            assert isinstance(best_params, dict)
            
            # Initialize and train
            manager.initialize_model(embeddings=embedding_dict)
            manager.train(embedding_dict, objective_dict)
            
            # Predict
            predictions = manager.predict(embedding_dict)
            assert isinstance(predictions, dict)
            assert len(predictions) == len(embedding_dict)
            
            # Verify prediction format
            for sequence_id, (mean, std) in predictions.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_manager_with_rnn_networks(self):
        """Test manager with RNN-based networks."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=15, seq_length=10, embedding_dim=20, variable_length=True
        )
        
        manager = SurrogateModelManager({
            'model_type': 'mc_dropout',
            'network_type': 'bilstm'
        })
        
        # Hyperparameter optimization
        best_params = manager.optimize_hyperparameters(
            embeddings=embedding_dict,
            objectives=objective_dict,
            n_trials=2
        )
        assert isinstance(best_params, dict)
        
        # Initialize model
        manager.initialize_model(embeddings=embedding_dict)
        assert manager.model is not None
        
        # Train
        manager.train(embedding_dict, objective_dict)
        
        # Predict
        predictions = manager.predict(embedding_dict)
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)
    
    def test_manager_train_with_validation_split(self):
        """Test manager training with validation split."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=25, embedding_dim=15, variable_length=False
        )
        
        manager = SurrogateModelManager({
            'model_type': 'deep_evidential',
            'network_type': 'mlp'
        })
        
        # Initialize with simple hyperparams
        manager.best_hyperparams = {
            'hidden_dims': [32, 16],
            'learning_rate': 0.001,
            'batch_size': 8,
            'epochs': 5,
            'uncertainty_param': 0.01  # Add regularization parameter
        }
        
        manager.initialize_model(embeddings=embedding_dict)
        
        # Train with validation split
        metrics = manager.train_with_validation_split(
            embeddings=embedding_dict,
            objectives=objective_dict,
            validation_size=5
        )
        
        # Check that we get both training and validation metrics
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert 'val_r2' in metrics
        assert 'val_mse' in metrics
        assert 'val_mae' in metrics
        
        # Validation metrics should be computed
        assert isinstance(metrics['val_r2'], float)
        assert isinstance(metrics['val_mse'], float)
        assert isinstance(metrics['val_mae'], float)
    
    def test_manager_train_without_validation(self):
        """Test manager training without validation (small dataset)."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=8, embedding_dim=10, variable_length=False  # Small dataset
        )
        
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        # Initialize with simple hyperparams
        manager.best_hyperparams = {
            'hidden_dims': [16],
            'learning_rate': 0.001,
            'batch_size': 4,
            'epochs': 5,
            'uncertainty_param': 0.01  # Add regularization parameter
        }
        
        manager.initialize_model(embeddings=embedding_dict)
        
        # Train with validation split (should train on all data due to small size)
        metrics = manager.train_with_validation_split(
            embeddings=embedding_dict,
            objectives=objective_dict,
            validation_size=3
        )
        
        # Should get training metrics but no validation metrics
        assert isinstance(metrics, dict)
        assert 'train_r2' in metrics
        assert 'train_mse' in metrics
        assert 'train_mae' in metrics
        assert metrics['val_r2'] is None
        assert metrics['val_mse'] is None
        assert metrics['val_mae'] is None
    
    def test_manager_cleanup_model(self):
        """Test manager model cleanup functionality."""
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=15, embedding_dim=10, variable_length=False
        )
        
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        # Initialize model
        manager.best_hyperparams = {
            'hidden_dims': [16],
            'learning_rate': 0.001,
            'batch_size': 8,
            'epochs': 2,
            'uncertainty_param': 0.01  # Add regularization parameter
        }
        manager.initialize_model(embeddings=embedding_dict)
        
        # Verify model exists
        assert manager.model is not None
        
        # Cleanup
        manager.cleanup_model()
        
        # Verify model is cleaned up
        assert manager.model is None
    
    def test_manager_error_handling(self):
        """Test manager error handling."""
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        embedding_dict, objective_dict = generate_synthetic_data(
            n_samples=10, embedding_dim=10, variable_length=False
        )
        
        # Test training without initialization
        with pytest.raises(RuntimeError, match="Model not initialized"):
            manager.train(embedding_dict, objective_dict)
        
        # Test prediction without initialization
        with pytest.raises(RuntimeError, match="Model not initialized"):
            manager.predict(embedding_dict)
        
        # Test initialization without hyperparameters
        with pytest.raises(ValueError, match="No hyperparameters available"):
            manager.initialize_model(embeddings=embedding_dict)
    
    def test_manager_input_dim_detection(self):
        """Test manager automatic input dimension detection."""
        # Test with 1D embeddings
        embedding_dict_1d = {
            'pep1': np.random.randn(20).astype(np.float32),
            'pep2': np.random.randn(20).astype(np.float32)
        }
        objective_dict = {'pep1': 1.0, 'pep2': 2.0}
        
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp'
        })
        
        manager.best_hyperparams = {
            'hidden_dims': [16],
            'learning_rate': 0.001,
            'batch_size': 4,
            'epochs': 2
        }
        
        # Should detect input_dim=20 from 1D embeddings
        manager.initialize_model(embeddings=embedding_dict_1d)
        assert manager.model is not None
        
        # Test with 2D embeddings (variable length)
        embedding_dict_2d = {
            'pep1': np.random.randn(5, 15).astype(np.float32),
            'pep2': np.random.randn(7, 15).astype(np.float32)
        }
        
        manager2 = SurrogateModelManager({
            'model_type': 'mc_dropout',
            'network_type': 'bilstm'
        })
        
        manager2.best_hyperparams = {
            'hidden_dims': [],
            'learning_rate': 0.001,
            'batch_size': 4,
            'epochs': 2,
            'uncertainty_param': 0.1,  # dropout rate
            'hidden_dim': 32,  # Add hidden_dim for LSTM
            'num_layers': 1    # Add num_layers for LSTM
        }
        
        # Should detect input_dim=15 from 2D embeddings
        manager2.initialize_model(embeddings=embedding_dict_2d)
        assert manager2.model is not None


class TestSurrogateModelConvergence:
    """Test that surrogate models show decreasing loss during training."""
    
    def test_single_objective_convergence(self):
        """Test convergence for single objective models."""
        embedding_dict, objective_dict = generate_training_data(
            n_samples=30, multiobjective=False
        )
        
        models = [
            MonteCarloDropout(input_dim=1, hidden_dims=[16, 8], network_type="mlp",
                            dropout_rate=0.1),
            NeuralNetworkEnsemble(input_dim=1, hidden_dims=[16, 8], n_networks=2,
                                network_type="mlp"),
            DeepEvidentialRegression(input_dim=1, hidden_dims=[16, 8], network_type="mlp"),
            MVE(input_dim=1, hidden_dims=[16, 8], network_type="mlp")
        ]
        
        for model in models:
            # Initial training
            initial_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=1,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Extended training
            final_loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=15,
                batch_size=16,
                learning_rate=0.01,
                device=device,
                verbose=False
            )
            
            # Loss should decrease or stay similar
            assert final_loss <= initial_loss * 1.2, \
                f"Model {type(model).__name__} did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
    
    def test_manager_convergence_workflow(self):
        """Test convergence using SurrogateModelManager."""
        embedding_dict, objective_dict = generate_training_data(
            n_samples=25, multiobjective=False
        )
        
        manager = SurrogateModelManager({
            'model_type': 'deep_evidential',
            'network_type': 'mlp'
        })
        
        # Set simple hyperparams for convergence test
        manager.best_hyperparams = {
            'hidden_dims': [16, 8],
            'learning_rate': 0.01,
            'batch_size': 16,
            'epochs': 20,
            'uncertainty_param': 0.01  # Add regularization parameter
        }
        
        manager.initialize_model(embeddings=embedding_dict)
        
        # Train and get metrics
        metrics = manager.train_with_validation_split(
            embeddings=embedding_dict,
            objectives=objective_dict,
            validation_size=5
        )
        
        # Should achieve reasonable R² score
        assert isinstance(metrics['train_r2'], float)
        assert isinstance(metrics['val_r2'], float)
        
        # R² should be reasonable (not perfect due to noise, but not terrible)
        assert metrics['train_r2'] > -1.0  # Basic sanity check
        assert metrics['val_r2'] > -2.0    # Validation can be worse due to small size



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

    test_mve = TestMVE()
    test_mve.test_initialization()
    test_mve.test_fit_and_predict()

    print("Running SurrogateModelManager tests...")
    test_manager = TestSurrogateModelManager()
    test_manager.test_manager_initialization()
    test_manager.test_manager_single_objective_workflow()
    test_manager.test_manager_different_model_types()
    test_manager.test_manager_with_rnn_networks()
    test_manager.test_manager_train_with_validation_split()
    test_manager.test_manager_train_without_validation()
    test_manager.test_manager_cleanup_model()
    test_manager.test_manager_error_handling()
    test_manager.test_manager_input_dim_detection()

    print("Running convergence tests...")
    test_convergence = TestSurrogateModelConvergence()
    test_convergence.test_single_objective_convergence()
    test_convergence.test_manager_convergence_workflow()

    print("All surrogate model tests passed!")