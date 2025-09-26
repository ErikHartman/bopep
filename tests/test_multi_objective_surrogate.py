"""
Comprehensive test suite for multi-objective surrogate models.

This module tests multi-objective functionality using only the current 
named objective format: {peptide: {obj_name: value}}

Tests include:
1. All surrogate model types with multiple objectives
2. MultiModelWrapper (separate models per objective)
3. SurrogateModelManager with multi-objective data
4. Variable length sequences with multi-objectives
5. Training convergence for multi-objective cases
6. Prediction format consistency
7. Objective ordering consistency
"""

import pytest
import numpy as np
import torch
from typing import Dict, Tuple, List

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
    MVE
)
from bopep.surrogate_model.manager import SurrogateModelManager
from bopep.surrogate_model.multi_model import MultiModelWrapper
from bopep.surrogate_model.helpers import pad_sequence
from bopep.surrogate_model.base_models import RNNetwork, MLPNetwork

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cpu")


def generate_multi_objective_data(
    n_samples: int = 50,
    embedding_dim: int = 20,
    objective_names: List[str] = None,
    variable_length: bool = False,
    seq_length: int = 15
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Generate synthetic multi-objective data with named objectives.
    
    Returns:
        embedding_dict: {peptide_id: embedding_array}
        objective_dict: {peptide_id: {obj_name: value}}
    """
    if objective_names is None:
        objective_names = ["binding_affinity", "stability", "selectivity"]
    
    embedding_dict = {}
    objective_dict = {}
    
    for i in range(n_samples):
        peptide_id = f"peptide_{i}"
        
        if variable_length:
            # Variable sequence length
            actual_length = np.random.randint(5, seq_length + 1)
            embedding = np.random.randn(actual_length, embedding_dim).astype(np.float32)
        else:
            # Fixed embedding dimension
            embedding = np.random.randn(embedding_dim).astype(np.float32)
        
        # Generate correlated objectives
        base_val = np.abs(embedding.mean())
        
        objectives = {}
        for j, obj_name in enumerate(objective_names):
            # Each objective has different correlation patterns
            if j == 0:
                score = base_val + np.random.normal(0, 0.1)
            elif j == 1:
                score = embedding.std() + 1.0 + np.random.normal(0, 0.1)
            else:
                score = base_val * (0.5 + j * 0.2) + np.random.normal(0, 0.1)
            
            objectives[obj_name] = float(score)
        
        embedding_dict[peptide_id] = embedding
        objective_dict[peptide_id] = objectives
    
    return embedding_dict, objective_dict


def generate_simple_training_data(
    n_samples: int = 30,
    embedding_dim: int = 1,
    objective_names: List[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Generate simple training data for convergence tests."""
    if objective_names is None:
        objective_names = ["obj1", "obj2"]
    
    embedding_dict = {}
    objective_dict = {}
    
    # Generate x values in range [-2, 2]
    x_values = np.random.uniform(-2, 2, n_samples)
    
    for i, x in enumerate(x_values):
        peptide_id = f"sample_{i}"
        embedding_dict[peptide_id] = np.array([x], dtype=np.float32)
        
        # Simple polynomial relationships
        objectives = {}
        if len(objective_names) >= 1:
            objectives[objective_names[0]] = float(x ** 3)
        if len(objective_names) >= 2:
            objectives[objective_names[1]] = float(x ** 3 + (x ** 2) ** 3)
        
        # Add any additional objectives
        for j in range(2, len(objective_names)):
            objectives[objective_names[j]] = float(x ** 2 * j + np.random.normal(0, 0.1))
        
        objective_dict[peptide_id] = objectives
    
    return embedding_dict, objective_dict


class TestMultiObjectiveNetworks:
    """Test basic network architectures with multi-objective support."""
    
    def test_mlp_networks_multi_objective(self):
        """Test MLP networks with multiple objectives."""
        model = MLPNetwork(input_dim=20, hidden_dims=[32, 16], n_objectives=3)
        x = torch.randn(10, 20)
        output = model(x)
        assert output.shape == (10, 3, 1)
        
    def test_rnn_networks_multi_objective(self):
        """Test RNN networks with multiple objectives."""
        model = RNNetwork(input_dim=20, hidden_dim=32, num_layers=1, n_objectives=2)
        
        # Variable length sequences
        x1 = torch.randn(5, 20)
        x2 = torch.randn(7, 20)
        lengths = torch.tensor([5, 7])
        batch = [x1, x2]
        
        padded_batch = pad_sequence(batch, batch_first=True)
        output = model(padded_batch, lengths)
        assert output.shape == (2, 2, 1)


class TestTraditionalMultiObjectiveModels:
    """Test traditional models (single model with multiple outputs)."""
    
    def test_monte_carlo_dropout_multi_objective(self):
        """Test MonteCarloDropout with named multi-objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=25, embedding_dim=20
        )
        
        model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[32, 16],
            dropout_rate=0.1,
            network_type="mlp",
            n_objectives=3
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        # Check format: {peptide: {obj_name: (mean, std)}}
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"binding_affinity", "stability", "selectivity"}
            for obj_name, (mean, std) in pred_dict.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_neural_network_ensemble_multi_objective(self):
        """Test NeuralNetworkEnsemble with named multi-objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=20, embedding_dim=15, objective_names=["obj1", "obj2"]
        )
        
        model = NeuralNetworkEnsemble(
            input_dim=15,
            hidden_dims=[32],
            n_networks=2,
            network_type="mlp",
            n_objectives=2
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"obj1", "obj2"}
            for obj_name, (mean, std) in pred_dict.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_deep_evidential_regression_multi_objective(self):
        """Test DeepEvidentialRegression with named multi-objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=25, embedding_dim=20, objective_names=["affinity", "stability"]
        )
        
        model = DeepEvidentialRegression(
            input_dim=20,
            hidden_dims=[32, 16],
            network_type="mlp",
            n_objectives=2
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"affinity", "stability"}
    
    def test_mve_multi_objective(self):
        """Test MVE with named multi-objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=20, embedding_dim=15, objective_names=["score1", "score2", "score3"]
        )
        
        model = MVE(
            input_dim=15,
            hidden_dims=[24],
            network_type="mlp",
            n_objectives=3
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=8,
            device=device
        )
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"score1", "score2", "score3"}
    
    def test_all_model_types_consistency(self):
        """Test that all model types work with named objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=15, embedding_dim=10, objective_names=["obj_a", "obj_b"]
        )
        
        models = [
            MonteCarloDropout(input_dim=10, hidden_dims=[16], network_type="mlp", 
                            dropout_rate=0.1, n_objectives=2),
            NeuralNetworkEnsemble(input_dim=10, hidden_dims=[16], n_networks=2, 
                                network_type="mlp", n_objectives=2),
            DeepEvidentialRegression(input_dim=10, hidden_dims=[16], network_type="mlp", 
                                   n_objectives=2),
            MVE(input_dim=10, hidden_dims=[16], network_type="mlp", n_objectives=2)
        ]
        
        for model in models:
            # Train
            loss = model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=1,
                batch_size=8,
                device=device
            )
            assert isinstance(loss, float)
            
            # Predict
            predictions = model.predict_dict(
                embedding_dict=embedding_dict,
                batch_size=8,
                device=device
            )
            
            # Check consistent format
            for peptide_id, pred_dict in predictions.items():
                assert isinstance(pred_dict, dict)
                assert set(pred_dict.keys()) == {"obj_a", "obj_b"}


class TestMultiModelWrapper:
    """Test MultiModelWrapper (separate models per objective)."""
    
    def test_multi_model_wrapper_initialization(self):
        """Test MultiModelWrapper setup."""
        multi_model = MultiModelWrapper(
            model_class=DeepEvidentialRegression,
            input_dim=20,
            n_objectives=3,
            network_type='mlp',
            hidden_dims=[32, 16]
        )
        
        assert len(multi_model.models) == 3
        assert multi_model.n_objectives == 3
        assert multi_model.objective_names is None  # Not set until training
    
    def test_multi_model_wrapper_training_and_prediction(self):
        """Test MultiModelWrapper complete workflow."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=25, embedding_dim=20, objective_names=["aff", "stab", "sel"]
        )
        
        multi_model = MultiModelWrapper(
            model_class=MVE,
            input_dim=20,
            n_objectives=3,
            network_type='mlp',
            hidden_dims=[32, 16]
        )
        
        # Train
        avg_loss = multi_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=3,
            batch_size=8,
            learning_rate=0.001
        )
        
        assert isinstance(avg_loss, float)
        assert multi_model.objective_names is not None
        assert len(multi_model.objective_names) == 3
        assert set(multi_model.objective_names) == {"aff", "stab", "sel"}
        
        # Predict
        predictions = multi_model.predict_dict(embedding_dict)
        
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"aff", "stab", "sel"}
            for obj_name, (mean, std) in pred_dict.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_multi_model_wrapper_objective_ordering(self):
        """Test objective ordering consistency in MultiModelWrapper."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=15, embedding_dim=10,
            objective_names=["z_last", "a_first", "m_middle"]  # Unsorted
        )
        
        multi_model = MultiModelWrapper(
            model_class=MonteCarloDropout,
            input_dim=10,
            n_objectives=3,
            network_type='mlp',
            hidden_dims=[16],
            dropout_rate=0.1
        )
        
        # Train
        multi_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=8
        )
        
        # Check objectives are stored in sorted order
        assert multi_model.objective_names == ["a_first", "m_middle", "z_last"]
        
        # Predictions should maintain sorted order
        predictions = multi_model.predict_dict(embedding_dict)
        for peptide_id, pred_dict in predictions.items():
            assert list(pred_dict.keys()) == ["a_first", "m_middle", "z_last"]
    
    def test_multi_model_wrapper_different_model_types(self):
        """Test MultiModelWrapper with different underlying model types."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=20, embedding_dim=15, objective_names=["obj1", "obj2"]
        )
        
        model_configs = [
            (DeepEvidentialRegression, {}),
            (MVE, {}),
            (MonteCarloDropout, {"dropout_rate": 0.1}),
            (NeuralNetworkEnsemble, {"n_networks": 2})
        ]
        
        for model_class, extra_kwargs in model_configs:
            multi_model = MultiModelWrapper(
                model_class=model_class,
                input_dim=15,
                n_objectives=2,
                network_type='mlp',
                hidden_dims=[16],
                **extra_kwargs
            )
            
            # Train
            loss = multi_model.fit_dict(
                embedding_dict=embedding_dict,
                objective_dict=objective_dict,
                epochs=2,
                batch_size=8
            )
            assert isinstance(loss, float)
            
            # Predict
            predictions = multi_model.predict_dict(embedding_dict)
            assert len(predictions) == len(embedding_dict)
            
            for peptide_id, pred_dict in predictions.items():
                assert isinstance(pred_dict, dict)
                assert set(pred_dict.keys()) == {"obj1", "obj2"}


class TestMultiObjectiveWithVariableLength:
    """Test multi-objective models with variable length sequences."""
    
    def test_bilstm_multi_objective_variable_length(self):
        """Test BiLSTM with multi-objectives and variable length sequences."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=15, embedding_dim=20, variable_length=True,
            objective_names=["binding", "stability"]
        )
        
        model = MonteCarloDropout(
            input_dim=20,
            hidden_dims=[],
            dropout_rate=0.1,
            network_type="bilstm",
            lstm_layers=1,
            lstm_hidden_dim=32,
            n_objectives=2
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=4,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=4,
            device=device
        )
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"binding", "stability"}
            for obj_name, (mean, std) in pred_dict.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_ensemble_variable_length_multi_objective(self):
        """Test ensemble with variable length and multi-objectives."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=12, embedding_dim=20, variable_length=True,
            objective_names=["score_a", "score_b", "score_c"]
        )
        
        model = NeuralNetworkEnsemble(
            input_dim=20,
            hidden_dims=[],
            n_networks=2,
            network_type="bilstm",
            lstm_layers=1,
            lstm_hidden_dim=32,
            n_objectives=3
        )
        
        # Train
        loss = model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=2,
            batch_size=4,
            device=device
        )
        assert isinstance(loss, float)
        
        # Predict
        predictions = model.predict_dict(
            embedding_dict=embedding_dict,
            batch_size=4,
            device=device
        )
        
        for peptide_id, pred_dict in predictions.items():
            assert isinstance(pred_dict, dict)
            assert set(pred_dict.keys()) == {"score_a", "score_b", "score_c"}


class TestMultiObjectiveTrainingConvergence:
    """Test training convergence for multi-objective models."""
    
    def test_traditional_models_convergence(self):
        """Test convergence for traditional multi-objective models."""
        embedding_dict, objective_dict = generate_simple_training_data(
            n_samples=30, objective_names=["obj1", "obj2"]
        )
        
        models = [
            MonteCarloDropout(input_dim=1, hidden_dims=[16, 8], network_type="mlp",
                            dropout_rate=0.1, n_objectives=2),
            NeuralNetworkEnsemble(input_dim=1, hidden_dims=[16, 8], n_networks=2,
                                network_type="mlp", n_objectives=2),
            DeepEvidentialRegression(input_dim=1, hidden_dims=[16, 8], network_type="mlp",
                                   n_objectives=2),
            MVE(input_dim=1, hidden_dims=[16, 8], network_type="mlp", n_objectives=2)
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
    
    def test_multi_model_wrapper_convergence(self):
        """Test MultiModelWrapper convergence."""
        embedding_dict, objective_dict = generate_simple_training_data(
            n_samples=25, objective_names=["obj1", "obj2"]
        )
        
        multi_model = MultiModelWrapper(
            model_class=DeepEvidentialRegression,
            input_dim=1,
            n_objectives=2,
            network_type='mlp',
            hidden_dims=[16, 8]
        )
        
        # Initial training
        initial_loss = multi_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=1,
            batch_size=16,
            learning_rate=0.01
        )
        
        # Extended training
        final_loss = multi_model.fit_dict(
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            epochs=15,
            batch_size=16,
            learning_rate=0.01
        )
        
        assert final_loss <= initial_loss * 1.2, \
            f"MultiModelWrapper did not converge: {initial_loss:.4f} -> {final_loss:.4f}"


class TestSurrogateModelManagerMultiObjective:
    """Test SurrogateModelManager with multi-objective functionality."""
    
    def test_manager_traditional_multi_objective_workflow(self):
        """Test complete workflow with traditional multi-objective approach."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=25, embedding_dim=20, objective_names=["aff", "stab", "sel"]
        )
        
        manager = SurrogateModelManager({
            'model_type': 'mve',
            'network_type': 'mlp',
            'n_objectives': 3,
            'multi_model': False
        })
        
        # Optimize hyperparameters
        best_params = manager.optimize_hyperparameters(
            embeddings=embedding_dict,
            objectives=objective_dict,
            n_trials=3
        )
        assert isinstance(best_params, dict)
        
        # Initialize and train
        manager.initialize_model(embeddings=embedding_dict)
        manager.train(embedding_dict, objective_dict)
        
        # Predict
        predictions = manager.predict(embedding_dict)
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)
        
        for peptide_id, pred in predictions.items():
            assert isinstance(pred, dict)
            assert set(pred.keys()) == {"aff", "stab", "sel"}
            for obj_name, (mean, std) in pred.items():
                assert isinstance(mean, float)
                assert isinstance(std, float)
                assert std >= 0
    
    def test_manager_multi_model_wrapper_workflow(self):
        """Test SurrogateModelManager with MultiModelWrapper."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=20, embedding_dim=15, objective_names=["bind", "stab"]
        )
        
        manager = SurrogateModelManager({
            'model_type': 'deep_evidential',
            'network_type': 'mlp',
            'multi_model': True
        })
        
        # Optimize hyperparameters
        best_params = manager.optimize_hyperparameters(
            embeddings=embedding_dict,
            objectives=objective_dict,
            n_trials=3
        )
        assert isinstance(best_params, dict)
        
        # Initialize model (should create MultiModelWrapper)
        manager.initialize_model(embeddings=embedding_dict, objectives=objective_dict)
        assert manager.model is not None
        assert isinstance(manager.model, MultiModelWrapper)
        
        # Train
        manager.train(embedding_dict, objective_dict)
        
        # Predict
        predictions = manager.predict(embedding_dict)
        assert isinstance(predictions, dict)
        assert len(predictions) == len(embedding_dict)
        
        for peptide_id, pred in predictions.items():
            assert isinstance(pred, dict)
            assert set(pred.keys()) == {"bind", "stab"}
    
    def test_manager_model_type_comparison(self):
        """Test different model types with manager."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=15, embedding_dim=10, objective_names=["obj1", "obj2"]
        )
        
        model_types = ['mve', 'deep_evidential', 'mc_dropout', 'nn_ensemble']
        
        for model_type in model_types:
            # Test both traditional and multi-model approaches
            for use_multi_model in [False, True]:
                config = {
                    'model_type': model_type,
                    'network_type': 'mlp',
                    'multi_model': use_multi_model
                }
                
                if not use_multi_model:
                    config['n_objectives'] = 2
                
                manager = SurrogateModelManager(config)
                
                # Quick hyperparameter optimization
                best_params = manager.optimize_hyperparameters(
                    embeddings=embedding_dict,
                    objectives=objective_dict,
                    n_trials=2
                )
                
                # Initialize, train, and predict
                if use_multi_model:
                    manager.initialize_model(embeddings=embedding_dict, objectives=objective_dict)
                else:
                    manager.initialize_model(embeddings=embedding_dict)
                
                manager.train(embedding_dict, objective_dict)
                predictions = manager.predict(embedding_dict)
                
                # Verify format consistency
                for peptide_id, pred in predictions.items():
                    assert isinstance(pred, dict)
                    assert set(pred.keys()) == {"obj1", "obj2"}


class TestMultiObjectiveValidation:
    """Test validation and error handling."""
    
    def test_multi_model_wrapper_input_validation(self):
        """Test MultiModelWrapper validates named objective format."""
        multi_model = MultiModelWrapper(
            model_class=MVE,
            input_dim=10,
            n_objectives=2,
            network_type='mlp',
            hidden_dims=[16]
        )
        
        embedding_dict = {"pep1": np.random.randn(10).astype(np.float32)}
        
        # Test with single float (should fail)
        objective_dict_wrong = {"pep1": 1.0}
        with pytest.raises(ValueError, match="MultiModelWrapper expects multi-objective format"):
            multi_model.fit_dict(embedding_dict, objective_dict_wrong)
        
        # Test with correct format (should work)
        objective_dict_correct = {"pep1": {"obj1": 1.0, "obj2": 2.0}}
        try:
            multi_model.fit_dict(embedding_dict, objective_dict_correct, epochs=1)
        except ValueError:
            pytest.fail("Should accept correct named objective format")
    
    def test_multi_model_wrapper_objective_count_validation(self):
        """Test MultiModelWrapper validates objective count."""
        multi_model = MultiModelWrapper(
            model_class=DeepEvidentialRegression,
            input_dim=10,
            n_objectives=3,  # Expects 3 objectives
            network_type='mlp',
            hidden_dims=[16]
        )
        
        embedding_dict = {"pep1": np.random.randn(10).astype(np.float32)}
        objective_dict = {"pep1": {"obj1": 1.0, "obj2": 2.0}}  # Only 2 objectives
        
        with pytest.raises(ValueError, match="Expected 3 objectives, got 2"):
            multi_model.fit_dict(embedding_dict, objective_dict)
    
    def test_prediction_format_consistency(self):
        """Test prediction formats are consistent between traditional and multi-model."""
        embedding_dict, objective_dict = generate_multi_objective_data(
            n_samples=12, embedding_dim=8, objective_names=["a", "b"]
        )
        
        # Traditional model
        traditional = MonteCarloDropout(
            input_dim=8, hidden_dims=[16], network_type="mlp",
            dropout_rate=0.1, n_objectives=2
        )
        traditional.fit_dict(embedding_dict, objective_dict, epochs=1, device=device)
        trad_pred = traditional.predict_dict(embedding_dict, device=device)
        
        # MultiModelWrapper
        multi_model = MultiModelWrapper(
            model_class=MonteCarloDropout,
            input_dim=8, n_objectives=2, network_type='mlp',
            hidden_dims=[16], dropout_rate=0.1
        )
        multi_model.fit_dict(embedding_dict, objective_dict, epochs=1)
        multi_pred = multi_model.predict_dict(embedding_dict)
        
        # Both should have same format
        assert len(trad_pred) == len(multi_pred)
        
        for peptide_id in embedding_dict.keys():
            assert isinstance(trad_pred[peptide_id], dict)
            assert isinstance(multi_pred[peptide_id], dict)
            assert set(trad_pred[peptide_id].keys()) == set(multi_pred[peptide_id].keys())
            
            for obj_name in trad_pred[peptide_id].keys():
                # Both should be (mean, std) tuples
                assert isinstance(trad_pred[peptide_id][obj_name], tuple)
                assert isinstance(multi_pred[peptide_id][obj_name], tuple)
                assert len(trad_pred[peptide_id][obj_name]) == 2
                assert len(multi_pred[peptide_id][obj_name]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
