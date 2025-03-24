import pytest
import os
import tempfile
from pathlib import Path
import numpy as np

from bopep.scoring.scorer import Scorer
from bopep.docking.utils import extract_sequence_from_pdb

# Sample PDB path for testing
TEST_PDB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "1ssc.pdb")
# If the test PDB doesn't exist, we'll skip tests that require it
SKIP_PDB_TESTS = not os.path.exists(TEST_PDB_PATH)

# Sample colab directory for testing
TEST_COLAB_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "data", 
    "sample_colab_dir"
)
# Skip colab dir tests if directory doesn't exist
SKIP_COLAB_TESTS = not os.path.exists(TEST_COLAB_DIR)

# Test peptide sequences
TEST_PEPTIDES = [
    "ACDPGHIKLM",
    "NQRSTYVWFG",
    "ACDEFGHIKLMNPQRSTVWY",
]

# Sample binding site residue indices
TEST_BINDING_SITE = [10, 11, 12, 15, 16, 17, 20, 21, 22]


class TestScorer:
    """Test class for the Scorer module."""

    def test_initialization(self):
        """Test that the Scorer class initializes correctly."""
        scorer = Scorer()
        assert isinstance(scorer, Scorer)
        assert hasattr(scorer, 'available_scores')
        assert isinstance(scorer.available_scores, list)
        assert len(scorer.available_scores) > 0
    
    def test_available_scores(self):
        """Test that the Scorer class has the expected available scores."""
        scorer = Scorer()
        expected_scores = [
            "all_rosetta_scores", "rosetta_score", "interface_sasa", 
            "interface_dG", "interface_delta_hbond_unsat", "packstat",
            "distance_score", "iptm", "in_binding_site", "peptide_properties",
            "molecular_weight", "aromaticity", "instability_index", 
            "isoelectric_point", "gravy", "helix_fraction", "turn_fraction",
            "sheet_fraction", "hydrophobic_aa_percent", "polar_aa_percent",
            "positively_charged_aa_percent", "negatively_charged_aa_percent",
            "delta_net_charge_frac", "uHrel"
        ]
        for score in expected_scores:
            assert score in scorer.available_scores
    
    def test_score_peptide_sequence_only(self):
        """Test scoring using only a peptide sequence."""
        scorer = Scorer()
        peptide = TEST_PEPTIDES[0]
        
        # Test all peptide property scores
        scores_to_include = [
            "peptide_properties", "molecular_weight", "aromaticity",
            "instability_index", "isoelectric_point", "gravy",
            "helix_fraction", "turn_fraction", "sheet_fraction",
            "hydrophobic_aa_percent", "polar_aa_percent",
            "positively_charged_aa_percent", "negatively_charged_aa_percent",
            "delta_net_charge_frac", "uHrel"
        ]
        
        result = scorer.score(
            scores_to_include=scores_to_include,
            peptide_sequence=peptide
        )
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert peptide in result
        assert isinstance(result[peptide], dict)
        
        # Check that all requested scores are present
        for score in scores_to_include:
            if score != "peptide_properties":  # This adds all properties directly
                assert score in result[peptide]
    
    def test_invalid_score_name(self):
        """Test that an invalid score name raises an error."""
        scorer = Scorer()
        with pytest.raises(ValueError, match="is not a valid score"):
            scorer.score(
                scores_to_include=["invalid_score_name"],
                peptide_sequence=TEST_PEPTIDES[0]
            )
    
    def test_missing_required_inputs(self):
        """Test that appropriate errors are raised when required inputs are missing."""
        scorer = Scorer()
        
        # No inputs provided
        with pytest.raises(ValueError, match="Either pdb_file, colab_dir, or peptide_sequence must be provided"):
            scorer.score(scores_to_include=["molecular_weight"])
        
        # Requesting Rosetta score without PDB file
        with pytest.raises(ValueError, match="requires a PDB file or colab_dir"):
            scorer.score(
                scores_to_include=["rosetta_score"],
                peptide_sequence=TEST_PEPTIDES[0]
            )
        
        # Requesting binding site score without binding site residues
        with pytest.raises(ValueError, match="requires a PDB file or colab_dir"):
            scorer.score(
                scores_to_include=["in_binding_site"],
                peptide_sequence=TEST_PEPTIDES[0]
            )
    
    @pytest.mark.skipif(SKIP_PDB_TESTS, reason="Test PDB file not available")
    def test_score_with_pdb_file(self):
        """Test scoring using a PDB file."""
        scorer = Scorer()
        
        # Test a combination of Rosetta scores and peptide properties
        scores_to_include = [
            "rosetta_score", "interface_sasa", "molecular_weight", 
            "aromaticity"
        ]
        
        result = scorer.score(
            scores_to_include=scores_to_include,
            pdb_file=TEST_PDB_PATH
        )
        
        # Get the peptide sequence from the PDB file
        peptide = extract_sequence_from_pdb(TEST_PDB_PATH, chain_id="B")
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert peptide in result
        assert isinstance(result[peptide], dict)
        
        # Check that all requested scores are present
        for score in scores_to_include:
            assert score in result[peptide]
    
    @pytest.mark.skipif(SKIP_PDB_TESTS, reason="Test PDB file not available")
    def test_binding_site_score_with_pdb(self):
        """Test binding site score calculation with a PDB file."""
        scorer = Scorer()
        
        result = scorer.score(
            scores_to_include=["in_binding_site"],
            pdb_file=TEST_PDB_PATH,
            binding_site_residue_indices=TEST_BINDING_SITE
        )
        
        # Get the peptide sequence from the PDB file
        peptide = extract_sequence_from_pdb(TEST_PDB_PATH, chain_id="B")
        
        # Verify the result structure and binding site score
        assert isinstance(result, dict)
        assert peptide in result
        assert "in_binding_site" in result[peptide]
        assert isinstance(result[peptide]["in_binding_site"], bool)
    
    @pytest.mark.skipif(SKIP_COLAB_TESTS, reason="Test colab directory not available")
    def test_score_with_colab_dir(self):
        """Test scoring using a colab directory."""
        scorer = Scorer()
        
        # Test scores that require a colab directory
        scores_to_include = ["iptm", "in_binding_site"]
        
        result = scorer.score(
            scores_to_include=scores_to_include,
            colab_dir=TEST_COLAB_DIR,
            binding_site_residue_indices=TEST_BINDING_SITE
        )
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert len(result) == 1  # Should have one peptide
        
        # Get the first (and only) peptide key
        peptide = next(iter(result.keys()))
        
        # Check that requested scores are present
        assert "iptm" in result[peptide]
        assert "in_binding_site" in result[peptide]
        assert "fraction_in_binding_site" in result[peptide]
        
        # Verify types
        assert isinstance(result[peptide]["iptm"], float)
        assert isinstance(result[peptide]["in_binding_site"], bool)
        assert isinstance(result[peptide]["fraction_in_binding_site"], float)
        assert 0 <= result[peptide]["fraction_in_binding_site"] <= 1
    
    def test_print_scores(self):
        """Test the print_scores method (simple existence check)."""
        scorer = Scorer()
        # This just calls print(), so we're just checking that it runs without error
        scorer.print_scores()


if __name__ == "__main__":
    # Run tests manually
    test_scorer = TestScorer()
    test_scorer.test_initialization()
    test_scorer.test_available_scores()
    test_scorer.test_score_peptide_sequence_only()
    test_scorer.test_invalid_score_name()
    test_scorer.test_missing_required_inputs()
    
    if not SKIP_PDB_TESTS:
        test_scorer.test_score_with_pdb_file()
        test_scorer.test_binding_site_score_with_pdb()
    
    if not SKIP_COLAB_TESTS:
        test_scorer.test_score_with_colab_dir()
    
    test_scorer.test_print_scores()
    
    print("All scorer tests passed!")
