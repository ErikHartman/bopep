"""
Test configuration and fixtures for BOPEP tests.
"""
import pytest
import tempfile
import numpy as np
import torch


@pytest.fixture
def sample_peptides():
    """Sample peptide sequences for testing."""
    return [
        "ACDEFGHIKLMNPQRSTVWY",  # All amino acids
        "GGGAAA",  # Simple peptide
        "KKKRRRHHH",  # Charged peptide
        "FFFWWWYYY",  # Aromatic peptide
        "ILVMAF",  # Hydrophobic peptide
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pdb_content():
    """Sample PDB file content for testing."""
    return """HEADER    PEPTIDE BINDING                         01-JAN-00   TEST
ATOM      1  N   ALA A   1      20.154  16.967  12.931  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.715  12.025  1.00 20.00           C  
ATOM      3  C   ALA A   1      18.573  15.277  12.028  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.943  14.836  12.975  1.00 20.00           O  
ATOM      5  CB  ALA A   1      17.857  17.609  12.380  1.00 20.00           C  
ATOM      6  N   GLY A   2      18.839  14.606  10.916  1.00 20.00           N  
ATOM      7  CA  GLY A   2      18.409  13.241  10.829  1.00 20.00           C  
ATOM      8  C   GLY A   2      17.047  13.087  10.189  1.00 20.00           C  
ATOM      9  O   GLY A   2      16.983  12.689   9.037  1.00 20.00           O  
END
"""


@pytest.fixture
def sample_scores_dict():
    """Sample scores dictionary for testing."""
    return {
        "PEPTIDE1": {
            "rosetta_score": -10.5,
            "interface_sasa": 500.0,
            "distance_score": 0.8,
            "molecular_weight": 1200.5
        },
        "PEPTIDE2": {
            "rosetta_score": -8.2,
            "interface_sasa": 450.0,
            "distance_score": 0.6,
            "molecular_weight": 1100.0
        }
    }


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)  # For reproducible tests
    return {
        "ACDEF": np.random.randn(1280),
        "GHIKL": np.random.randn(1280),
        "MNPQR": np.random.randn(1280),
    }


@pytest.fixture
def sample_2d_embeddings():
    """Sample 2D embeddings (sequence-level) for testing."""
    np.random.seed(42)
    return {
        "ACDEF": np.random.randn(5, 1280),  # 5 amino acids, 1280 dims
        "GHIKL": np.random.randn(5, 1280),
        "MNPQR": np.random.randn(5, 1280),
    }


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_esm():
    """Skip test if ESM is not available."""
    try:
        import esm
    except ImportError:
        pytest.skip("ESM not available")
