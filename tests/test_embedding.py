import numpy as np

from bopep.embedding.embedder import Embedder
from bopep.embedding.utils import filter_sequences


class TestEmbedder:
    """Test the main Embedder class"""

    def test_init(self):
        """Test embedder initialization"""
        embedder = Embedder()
        assert embedder is not None

    def test_scale_embeddings_1d(self):
        """Test scaling of 1D embeddings"""
        embedder = Embedder()
        sample_embeddings = {
            "ACDEF": np.array([1.0, 2.0, 3.0]),
            "GHIKL": np.array([4.0, 5.0, 6.0])
        }
        
        scaled = embedder.scale_embeddings(sample_embeddings)
        
        assert len(scaled) == 2
        assert "ACDEF" in scaled
        assert "GHIKL" in scaled
        assert isinstance(scaled["ACDEF"], np.ndarray)

    def test_scale_embeddings_2d(self):
        """Test scaling of 2D embeddings"""
        embedder = Embedder()
        sample_2d_embeddings = {
            "ACDEF": np.random.rand(5, 1280),
            "GHIKL": np.random.rand(5, 1280)
        }
        
        scaled = embedder.scale_embeddings(sample_2d_embeddings)
        
        assert len(scaled) == 2
        assert "ACDEF" in scaled
        assert "GHIKL" in scaled
        assert scaled["ACDEF"].shape == (5, 1280)


class TestUtils:
    """Test embedding utility functions"""

    def test_filter_sequences_valid(self):
        """Test filtering with valid sequences"""
        sequences = ["ACDEFGHIKL", "MNPQRSTVWY"]  # Remove AAAAAA as it fails fraction test
        
        result = filter_sequences(sequences)
        
        assert len(result) == 2
        assert "ACDEFGHIKL" in result
        assert "MNPQRSTVWY" in result

    def test_filter_sequences_invalid(self):
        """Test filtering with invalid characters"""
        sequences = ["ACDEFGHIKL", "XYZ123", "MNPQRSTVWY"]
        
        result = filter_sequences(sequences)
        
        assert "XYZ123" not in result
        assert "ACDEFGHIKL" in result
        assert "MNPQRSTVWY" in result

    def test_filter_sequences_empty(self):
        """Test filtering with empty list"""
        result = filter_sequences([])
        assert result == []

    def test_filter_sequences_length_limits(self):
        """Test filtering based on length limits"""
        sequences = ["AA", "ACDEFG", "ACDEFGHIKLMNPQRSTVWY", "A" * 100]
        
        result = filter_sequences(sequences, min_length=3, max_length=25)
        
        # Only ACDEFG and ACDEFGHIKLMNPQRSTVWY should pass
        expected = ["ACDEFG", "ACDEFGHIKLMNPQRSTVWY"]
        assert len(result) == len(expected)
        for sequence in expected:
            assert sequence in result
