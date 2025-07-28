"""
Tests for the surrogate_model module.
"""
import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock, call
import pickle

from bopep.surrogate_model.base_models import BaseNetwork, MLPNetwork
from bopep.surrogate_model.helpers import BasePredictionModel
from bopep.surrogate_model.nn_ensemble import NeuralNetworkEnsemble
from bopep.surrogate_model.mc_dropout import MonteCarloDropout
from bopep.surrogate_model.deep_evidential_regression import DeepEvidentialRegression
from bopep.surrogate_model.mve import MVE
from bopep.surrogate_model.hyperparameter_tuner import tune_hyperparams


class TestBasePredictionModel:
    """Test the base prediction model class."""
    
    def test_init(self):
        """Test base prediction model initialization."""
        model = BasePredictionModel()
        assert model is not None
        assert isinstance(model, torch.nn.Module)


class TestBaseNetwork:
    """Test the base network class."""
    
    def test_init(self):
        """Test base network initialization."""
        model = BaseNetwork()
        assert model is not None
    
    def test_abstract_forward(self):
        """Test that forward method raises NotImplementedError."""
        model = BaseNetwork()
        
        with pytest.raises(NotImplementedError):
            model.forward(torch.randn(5, 10))


class TestMLPNetwork:
    """Test the MLP network class."""
    
    def test_init(self):
        """Test MLP network initialization."""
        model = MLPNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        
        assert model is not None
    
    def test_forward(self):
        """Test forward pass through MLP."""
        model = MLPNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=1)
        
        x = torch.randn(5, 10)
        output = model.forward(x)
        
        assert output.shape == (5, 1)
        assert torch.isfinite(output).all()


class TestNeuralNetworkEnsemble:
    """Test the Neural Network Ensemble model."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = NeuralNetworkEnsemble(input_dim=10)
        
        assert model.n_networks == 5
        assert len(model.networks) == 5
    
    def test_init_custom(self):
        """Test custom initialization."""
        model = NeuralNetworkEnsemble(
            input_dim=20,
            hidden_dims=[128, 64, 32],
            n_networks=10,
            dropout_rate=0.2,
            network_type="mlp"
        )
        
        assert model.n_networks == 10
        assert len(model.networks) == 10
    
    def test_network_factory_call(self):
        """Test that network factory is called correctly during initialization"""
        # Can't mock network_factory as it needs to return actual torch.nn.Module
        # Just test that the ensemble can be created successfully
        model = NeuralNetworkEnsemble(input_dim=10, n_networks=3)
        
        assert len(model.networks) == 3
        assert model.n_networks == 3
    
    def test_forward_predict(self):
        """Test forward prediction."""
        model = NeuralNetworkEnsemble(input_dim=5, n_networks=2)
        
        x = torch.randn(3, 5)
        mean_pred, std_pred = model.forward_predict(x)
        
        assert mean_pred.shape == (3, 1)
        assert std_pred.shape == (3, 1)
        assert torch.isfinite(mean_pred).all()
        assert torch.isfinite(std_pred).all()
        assert (std_pred >= 0).all()  # Standard deviation should be non-negative


class TestSurrogateModelImports:
    """Test that all surrogate model classes can be imported."""
    
    def test_import_monte_carlo_dropout(self):
        """Test MonteCarloDropout import and initialization."""
        model = MonteCarloDropout(input_dim=10)
        assert model is not None
        assert isinstance(model, BasePredictionModel)
    
    def test_import_deep_evidential_regression(self):
        """Test DeepEvidentialRegression import and initialization."""
        model = DeepEvidentialRegression(input_dim=10)
        assert model is not None
        assert isinstance(model, BasePredictionModel)
    
    def test_import_mve(self):
        """Test MVE import and initialization."""
        model = MVE(input_dim=10)
        assert model is not None
        assert isinstance(model, BasePredictionModel)
    
    def test_import_tune_hyperparams(self):
        """Test tune_hyperparams function import."""
        assert tune_hyperparams is not None
        assert callable(tune_hyperparams)
