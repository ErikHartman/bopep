import pytest
import os
import numpy as np
import tempfile
from pathlib import Path

from bopep.embedding.embedder import Embedder
from bopep.embedding.utils import filter_peptides

# Test peptides for embedding testing
TEST_PEPTIDES = [
    "ACDPGHIKLM",
    "NQRSTYVWFG",
    "ACDEFGHIKLMNPQRSTVWY",
    "GACEG",
    "KLFGPS",
]


class TestEmbedder:
    """Test class for the Embedder module."""

    def test_initialization(self):
        """Test that the Embedder class initializes correctly."""
        embedder = Embedder()
        assert isinstance(embedder, Embedder)
    
    def test_aaindex_embed_1d(self):
        """Test AAIndex embedding with average=True (1D output)."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=True)
        
        # Verify output is a dictionary with the right keys
        assert isinstance(embeddings, dict)
        assert set(embeddings.keys()) == set(TEST_PEPTIDES)
        
        # Verify 1D shape (averaged)
        sample_embedding = embeddings[TEST_PEPTIDES[0]]
        assert sample_embedding.ndim == 1
        
        # Verify embeddings are numerical and finite
        assert np.isfinite(sample_embedding).all()
        
    def test_aaindex_embed_2d(self):
        """Test AAIndex embedding with average=False (2D output)."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=False)
        
        # Verify output structure
        assert isinstance(embeddings, dict)
        assert set(embeddings.keys()) == set(TEST_PEPTIDES)
        
        # Verify 2D shape (sequence-based)
        for peptide, embedding in embeddings.items():
            assert embedding.ndim == 2
            assert embedding.shape[0] == len(peptide)  # Length should match peptide length
            
            # Verify embeddings are numerical and finite
            assert np.isfinite(embedding).all()
    
    def test_esm_embed_1d(self):
        """Test ESM embedding with average=True (1D output)."""
        # Skip this test if running in CI or environment where model is not available
        if not os.path.exists("esm2_t33_650M_UR50D.pt") and not os.environ.get("FULL_TEST_SUITE"):
            pytest.skip("ESM model not available - skipping ESM embedding test")
            
        embedder = Embedder()
        # Use a small batch size for testing
        embeddings = embedder.embed_esm(
            TEST_PEPTIDES,  # Use fewer peptides for faster testing 
            average=True, 
            model_path=None,  # Use default model path (will download)
            batch_size=2,
            filter=False
        )
        
        # Verify output structure
        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        
        # Verify 1D shape (averaged)
        sample_embedding = next(iter(embeddings.values()))
        assert sample_embedding.ndim == 1
        
        # Verify embeddings are numerical and finite
        assert np.isfinite(sample_embedding).all()
    
    def test_esm_embed_2d(self):
        """Test ESM embedding with average=False (2D output)."""
        # Skip this test if running in CI or environment where model is not available
        if not os.path.exists("esm2_t33_650M_UR50D.pt") and not os.environ.get("FULL_TEST_SUITE"):
            pytest.skip("ESM model not available - skipping ESM embedding test")
            
        embedder = Embedder()
        # Use a small batch size for testing
        embeddings = embedder.embed_esm(
            TEST_PEPTIDES,  # Use fewer peptides for faster testing 
            average=False, 
            model_path=None,  # Use default model path (will download)
            batch_size=2,
            filter=False
        )
        
        # Verify output structure
        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0
        
        # Verify 2D shape (sequence-based)
        for peptide, embedding in embeddings.items():
            assert embedding.ndim == 2
            assert embedding.shape[0] == len(peptide)  # Length should match peptide length
            
            # Verify embeddings are numerical and finite
            assert np.isfinite(embedding).all()
    
    def test_scaling(self):
        """Test the scaling functionality of embeddings."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=True)
        
        # Scale embeddings
        scaled_embeddings = embedder.scale_embeddings(embeddings)
        
        # Check that keys are preserved
        assert set(scaled_embeddings.keys()) == set(embeddings.keys())
        
        # Stack original and scaled for comparison
        original_stacked = np.vstack([emb for emb in embeddings.values()])
        scaled_stacked = np.vstack([emb for emb in scaled_embeddings.values()])
        
        # Check that scaling changed the values
        assert not np.allclose(original_stacked, scaled_stacked)
        
        # Check that scaled data has approximately mean 0 and std 1
        assert -0.1 < scaled_stacked.mean() < 0.1
        assert 0.9 < scaled_stacked.std() < 1.1
    
    def test_pca_reduction_1d(self):
        """Test PCA dimensionality reduction for 1D embeddings."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=True)
        
        # Get original dimensionality 
        original_dim = next(iter(embeddings.values())).shape[0]
        
        # Apply PCA reduction with specific n_components
        n_components = 5
        reduced_embeddings = embedder.reduce_embeddings_pca(
            embeddings, n_components=n_components
        )
        
        # Check that keys are preserved
        assert set(reduced_embeddings.keys()) == set(embeddings.keys())
        
        # Check that dimensions are reduced properly
        sample_reduced = next(iter(reduced_embeddings.values()))
        assert sample_reduced.shape[0] == n_components
        assert sample_reduced.shape[0] < original_dim
        
        # Verify reduced embeddings are numerical and finite
        assert np.isfinite(sample_reduced).all()
    
    def test_pca_reduction_2d(self):
        """Test PCA dimensionality reduction for 2D embeddings."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=False)
        
        # Get original dimensionality
        original_dim = next(iter(embeddings.values())).shape[1]
        
        # Apply PCA reduction with specific n_components
        n_components = 5
        reduced_embeddings = embedder.reduce_embeddings_pca(
            embeddings, n_components=n_components
        )
        
        # Check that keys are preserved
        assert set(reduced_embeddings.keys()) == set(embeddings.keys())
        
        # Check that dimensions are reduced properly
        for peptide, embedding in reduced_embeddings.items():
            assert embedding.shape[0] == len(peptide)  # Sequence length preserved
            assert embedding.shape[1] == n_components  # Dimensionality reduced
            assert embedding.shape[1] < original_dim
            
            # Verify reduced embeddings are numerical and finite
            assert np.isfinite(embedding).all()
    
    def test_pca_too_many_components(self):
        """Test that PCA raises an error when n_components is too large."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=True)
        
        # Get original dimensionality 
        original_dim = next(iter(embeddings.values())).shape[0]
        
        # Try to apply PCA with too many components
        with pytest.raises(ValueError):
            _ = embedder.reduce_embeddings_pca(
                embeddings, n_components=original_dim + 1
            )
    
    def test_vae_reduction_1d(self):
        """Test VAE dimensionality reduction for 1D embeddings."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=True)
        
        # Apply VAE reduction
        latent_dim = 5
        vae_embeddings = embedder.reduce_embeddings_autoencoder(
            embeddings, 
            latent_dim=latent_dim,
            hidden_layers=[16, 8],  # Small network for testing
            batch_size=2,
            max_epochs=2,  # Just a couple epochs for testing
            verbose=False
        )
        
        # Check that keys are preserved
        assert set(vae_embeddings.keys()) == set(embeddings.keys())
        
        # Check latent dimensionality
        sample_embedding = next(iter(vae_embeddings.values()))
        assert sample_embedding.shape[0] == latent_dim
        
        # Verify embeddings are numerical and finite
        assert np.isfinite(sample_embedding).all()
    
    def test_vae_reduction_2d(self):
        """Test VAE dimensionality reduction for 2D embeddings."""
        embedder = Embedder()
        embeddings = embedder.embed_aaindex(TEST_PEPTIDES, average=False)
        
        # Flatten 2D embeddings to make them compatible with VAE reduction
        flattened_embeddings = {}
        for peptide, embedding in embeddings.items():
            flattened_embeddings[peptide] = embedding.flatten()
        
        # Apply VAE reduction
        latent_dim = 5
        vae_embeddings = embedder.reduce_embeddings_autoencoder(
            flattened_embeddings, 
            latent_dim=latent_dim,
            hidden_layers=[16, 8],  # Small network for testing
            batch_size=2,
            max_epochs=2,  # Just a couple epochs for testing
            verbose=False
        )
        
        # Check that keys are preserved
        assert set(vae_embeddings.keys()) == set(embeddings.keys())
        
        # Check latent dimensionality
        sample_embedding = next(iter(vae_embeddings.values()))
        assert sample_embedding.shape[0] == latent_dim
        
        # Verify embeddings are numerical and finite
        assert np.isfinite(sample_embedding).all()
    
    def test_empty_input(self):
        """Test that empty input list is handled properly."""
        embedder = Embedder()
        
        # Test with empty list
        with pytest.raises(ValueError):
            embedder.embed_aaindex([], average=False)


class TestEmbeddingUtils:
    """Test class for embedding utility functions."""
    
    def test_filter_peptides(self):
        """Test peptide filtering functionality."""
        # Test valid peptides
        valid_peptides = ["GHIKLM", "NPQRST"]
        filtered = filter_peptides(valid_peptides)
        assert filtered == valid_peptides
        
        # Test peptides with invalid characters
        invalid_peptides = ["ACDBEF", "GHIK1LM", "NPQ-RST", "ACBX"]
        filtered = filter_peptides(invalid_peptides)
        assert len(filtered) < len(invalid_peptides)
        
        # Test mixed case
        mixed_case = ["acdef", "GHIklm", "npQRSt"]
        filtered = filter_peptides(mixed_case)
        # Should convert to uppercase
        assert all(p.isupper() for p in filtered)
    
    def test_filter_peptides_all_invalid(self):
        """Test filter_peptides with all invalid peptides."""
        all_invalid = ["123", "B-X-Z", "#$@!"]
        filtered = filter_peptides(all_invalid)
        assert filtered == []


if __name__ == "__main__":
    # Run tests manually
    test_embedder = TestEmbedder()
    test_embedder.test_initialization()
    test_embedder.test_aaindex_embed_1d()
    test_embedder.test_aaindex_embed_2d()
    # test_embedder.test_esm_embed_1d()
    # test_embedder.test_esm_embed_2d()
    test_embedder.test_scaling()
    test_embedder.test_pca_reduction_1d()
    test_embedder.test_pca_reduction_2d()
    test_embedder.test_pca_too_many_components()
    test_embedder.test_vae_reduction_1d()
    test_embedder.test_vae_reduction_2d()
    test_embedder.test_saving_embeddings()
    test_embedder.test_empty_input()
    
    test_utils = TestEmbeddingUtils()
    test_utils.test_filter_peptides()
    test_utils.test_filter_peptides_empty_input()
    test_utils.test_filter_peptides_all_invalid()
    
    print("All embedding tests passed!")
