import pytest
import numpy as np
from bopep.bayes.acquisition import AcquisitionFunction
from bopep.search.selection import SequenceSelector


class TestAcquisitionFunction:
    """Test acquisition functions"""

    def test_init_default(self):
        """Test default initialization"""
        acq_func = AcquisitionFunction()
        assert acq_func.best_so_far_ei == 0.0
        assert acq_func.best_so_far_pi == 0.0

    def test_compute_acquisition_upper_confidence_bound(self):
        """Test upper confidence bound calculation"""
        acq_func = AcquisitionFunction()
        predictions = {"ACDEF": (0.5, 0.1), "GHIKL": (0.8, 0.2)}
        
        result = acq_func.compute_acquisition(predictions, "upper_confidence_bound")
        
        assert len(result) == 2
        assert "ACDEF" in result
        assert "GHIKL" in result

    def test_compute_acquisition_expected_improvement(self):
        """Test expected improvement calculation"""
        acq_func = AcquisitionFunction()
        predictions = {"ACDEF": (0.5, 0.1), "GHIKL": (0.8, 0.2)}
        
        result = acq_func.compute_acquisition(predictions, "expected_improvement")
        
        assert len(result) == 2
        assert "ACDEF" in result
        assert "GHIKL" in result

    def test_compute_acquisition_invalid_function(self):
        """Test calculation with invalid acquisition function"""
        acq_func = AcquisitionFunction()
        predictions = {"ACDEF": (0.5, 0.1)}
        
        with pytest.raises(ValueError):
            acq_func.compute_acquisition(predictions, "invalid_function")


class TestSequenceSelector:
    """Test sequence selection methods"""

    def test_init_default(self):
        """Test default initialization"""
        selector = SequenceSelector()
        # SequenceSelector has no attributes in __init__, just check it's created
        assert selector is not None

    def test_select_initial_sequences_kmeans(self):
        """Test initial sequence selection with k-means"""
        selector = SequenceSelector()
        embeddings = {
            "ACDEF": np.random.rand(10),
            "GHIKL": np.random.rand(10),
            "MNPQR": np.random.rand(10)
        }
        
        result = selector.select_initial_sequences(embeddings, num_initial=2, method="kmeans")
        
        assert len(result) == 2
        assert all(p in embeddings.keys() for p in result)

    def test_select_initial_sequences_kmeans_plus(self):
        """Test initial sequence selection with k-means++"""
        selector = SequenceSelector()
        embeddings = {
            "ACDEF": np.random.rand(10),
            "GHIKL": np.random.rand(10),
            "MNPQR": np.random.rand(10)
        }
        
        result = selector.select_initial_sequences(embeddings, num_initial=2, method="kmeans++")
        
        assert len(result) == 2
        assert all(p in embeddings.keys() for p in result)

    def test_select_initial_sequences_invalid_method(self):
        """Test initial sequence selection with invalid method"""
        selector = SequenceSelector()
        embeddings = {"ACDEF": np.random.rand(10)}
        
        with pytest.raises(ValueError):
            selector.select_initial_sequences(embeddings, num_initial=1, method="invalid")
