"""
Tests for monomer protein scoring (unconditional generation).

This module tests the MonomerScorer class which handles scoring of
single-chain proteins without a target structure.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from bopep.scoring.monomer_scorer import MonomerScorer
from bopep.scoring.peptide_properties import PeptideProperties


class TestMonomerScorer:
    """Test the MonomerScorer class for unconditional protein generation"""

    def test_init(self):
        """Test MonomerScorer initialization"""
        scorer = MonomerScorer()
        assert hasattr(scorer, 'confidence_scores')
        assert hasattr(scorer, 'sequence_property_scores')
        assert hasattr(scorer, 'dssp_scores')
        assert hasattr(scorer, 'rmsd_scores')
        
        # Check that essential scores are available
        all_scores = scorer._all_possible_scores
        assert 'plddt' in all_scores
        assert 'ptm' in all_scores
        assert 'molecular_weight' in all_scores
        assert 'dssp_helix_fraction' in all_scores

    def test_available_scores(self):
        """Test that available scores are properly listed"""
        scorer = MonomerScorer()
        available = scorer.available_scores
        
        # Should include confidence scores
        assert 'plddt' in available
        assert 'ptm' in available
        assert 'pae' in available
        
        # Should include sequence property scores
        assert 'molecular_weight' in available
        assert 'aromaticity' in available
        assert 'isoelectric_point' in available
        
        # Should include DSSP scores
        assert 'dssp_helix_fraction' in available
        assert 'dssp_strand_fraction' in available
        assert 'dssp_loop_fraction' in available

    def test_get_available_scores_no_context(self):
        """Test available scores with no specific context"""
        scorer = MonomerScorer()
        available = scorer.get_available_scores()
        
        # Should always have sequence property scores
        assert 'molecular_weight' in available
        assert 'aromaticity' in available

    def test_get_available_scores_with_structure(self):
        """Test available scores when structure file is provided"""
        scorer = MonomerScorer()
        
        # Create a dummy PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
            temp_pdb = f.name
        
        try:
            available = scorer.get_available_scores(structure_file=temp_pdb)
            
            # Should have sequence properties
            assert 'molecular_weight' in available
            
            # Should have DSSP scores
            assert 'dssp_helix_fraction' in available
            assert 'dssp_strand_fraction' in available
        finally:
            os.unlink(temp_pdb)

    def test_get_available_scores_with_processed_dir(self):
        """Test available scores when processed directory with AlphaFold output is provided"""
        scorer = MonomerScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock alphafold_metrics.json
            metrics = {
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "ptm": 0.85,
                "plddt_vector": [80.0] * 20,
                "pae_matrix": [[5.0] * 20 for _ in range(20)]
            }
            metrics_path = os.path.join(tmpdir, "alphafold_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            available = scorer.get_available_scores(processed_dir=tmpdir)
            
            # Should have confidence scores
            assert 'plddt' in available
            assert 'ptm' in available
            assert 'pae' in available

    def test_score_sequence_only(self):
        """Test scoring with sequence only"""
        scorer = MonomerScorer()
        test_sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        scores_to_include = ["molecular_weight", "aromaticity", "isoelectric_point"]
        results = scorer.score(
            scores_to_include=scores_to_include,
            sequence=test_sequence
        )
        
        assert test_sequence in results
        assert 'molecular_weight' in results[test_sequence]
        assert 'aromaticity' in results[test_sequence]
        assert 'isoelectric_point' in results[test_sequence]
        
        # Verify reasonable values
        assert results[test_sequence]['molecular_weight'] > 0
        assert 0 <= results[test_sequence]['aromaticity'] <= 1

    def test_score_with_processed_dir(self):
        """Test scoring with processed AlphaFold directory"""
        scorer = MonomerScorer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_sequence = "ACDEFGHIKLMNPQRSTVWY"
            
            # Create mock alphafold_metrics.json
            metrics = {
                "sequence": test_sequence,
                "ptm": 0.85,
                "plddt_vector": [80.0] * 20,
                "pae_matrix": [[5.0] * 20 for _ in range(20)]
            }
            metrics_path = os.path.join(tmpdir, "alphafold_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            
            scores_to_include = ["ptm", "plddt", "pae", "molecular_weight"]
            results = scorer.score(
                scores_to_include=scores_to_include,
                processed_dir=tmpdir
            )
            
            assert test_sequence in results
            assert results[test_sequence]['ptm'] == 0.85
            assert results[test_sequence]['plddt'] == 80.0
            assert results[test_sequence]['molecular_weight'] > 0

    def test_score_with_structure_file(self):
        """Test scoring with structure file for DSSP analysis"""
        from bopep.structure.parser import extract_sequence_from_structure
        
        scorer = MonomerScorer()
        
        # Use a real test PDB file if available
        test_pdb = Path(__file__).parent.parent / "data" / "1ssc.pdb"
        if not test_pdb.exists():
            pytest.skip("Test PDB file not available")
        
        scores_to_include = ["dssp_helix_fraction", "molecular_weight"]
        
        # Test with chain A (monomer)
        with patch('bopep.scoring.base_scorer.DSSPAnalyzer') as mock_dssp:
            mock_dssp_instance = Mock()
            mock_dssp.return_value = mock_dssp_instance
            mock_dssp_instance.get_dssp_helix_fraction.return_value = 0.4
            
            results = scorer.score(
                scores_to_include=scores_to_include,
                structure_file=str(test_pdb),
                chain_id="A"
            )
            
            assert len(results) == 1
            sequence = list(results.keys())[0]
            assert 'dssp_helix_fraction' in results[sequence]
            assert 'molecular_weight' in results[sequence]

    def test_score_invalid_score_name(self):
        """Test that invalid score names raise an error"""
        scorer = MonomerScorer()
        
        with pytest.raises(ValueError, match="not a valid score"):
            scorer.score(
                scores_to_include=["invalid_score"],
                sequence="ACDEFG"
            )

    def test_score_no_input(self):
        """Test that scoring without any input raises an error"""
        scorer = MonomerScorer()
        
        with pytest.raises(ValueError, match="Must provide"):
            scorer.score(scores_to_include=["molecular_weight"])

    def test_confidence_scores_without_processed_dir(self):
        """Test that confidence scores are None when no processed_dir is provided"""
        scorer = MonomerScorer()
        
        # Requesting confidence scores without processed_dir should not include them
        # or return None values
        results = scorer.score(
            scores_to_include=["molecular_weight"],
            sequence="ACDEFG"
        )
        
        assert len(results) == 1
        assert 'molecular_weight' in results["ACDEFG"]

    def test_all_sequence_properties(self):
        """Test all sequence property scores"""
        scorer = MonomerScorer()
        test_sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        sequence_scores = [
            'molecular_weight', 'aromaticity', 'instability_index',
            'isoelectric_point', 'gravy', 'helix_fraction', 'turn_fraction',
            'sheet_fraction', 'hydrophobic_aa_percent', 'polar_aa_percent',
            'positively_charged_aa_percent', 'negatively_charged_aa_percent',
            'delta_net_charge_frac', 'uHrel'
        ]
        
        results = scorer.score(
            scores_to_include=sequence_scores,
            sequence=test_sequence
        )
        
        assert test_sequence in results
        for score_name in sequence_scores:
            assert score_name in results[test_sequence]
            assert results[test_sequence][score_name] is not None


class TestMonomerScorerIntegration:
    """Integration tests for MonomerScorer with real data"""

    def test_score_with_alphafold_output_structure(self):
        """Test scoring with actual AlphaFold monomer output if available"""
        test_folding_dir = Path("/home/er8813ha/bopep/test_folding")
        
        if not test_folding_dir.exists():
            pytest.skip("AlphaFold test output not available")
        
        # Find a processed directory
        processed_dirs = list((test_folding_dir / "processed").glob("seq_*"))
        if not processed_dirs:
            pytest.skip("No processed AlphaFold output available")
        
        processed_dir = str(processed_dirs[0])
        
        scorer = MonomerScorer()
        scores_to_include = ["ptm", "plddt", "molecular_weight", "aromaticity"]
        
        results = scorer.score(
            scores_to_include=scores_to_include,
            processed_dir=processed_dir
        )
        
        assert len(results) == 1
        sequence = list(results.keys())[0]
        assert results[sequence]['ptm'] is not None
        assert results[sequence]['plddt'] is not None
        assert results[sequence]['molecular_weight'] > 0

    def test_end_to_end_monomer_scoring(self):
        """Test end-to-end scoring workflow"""
        scorer = MonomerScorer()
        test_sequence = "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV"
        
        # Score with available metrics
        scores_to_include = [
            'molecular_weight',
            'aromaticity',
            'isoelectric_point',
            'helix_fraction',
            'hydrophobic_aa_percent'
        ]
        
        results = scorer.score(
            scores_to_include=scores_to_include,
            sequence=test_sequence
        )
        
        assert test_sequence in results
        scores = results[test_sequence]
        
        # Verify all requested scores are present
        for score_name in scores_to_include:
            assert score_name in scores
            assert scores[score_name] is not None
        
        # Verify reasonable ranges
        assert scores['molecular_weight'] > 1000
        assert 0 <= scores['aromaticity'] <= 1
        assert 0 <= scores['isoelectric_point'] <= 14
        assert 0 <= scores['helix_fraction'] <= 1
        assert 0 <= scores['hydrophobic_aa_percent'] <= 100


class TestMonomerScorerComparison:
    """Test MonomerScorer in comparison to ComplexScorer"""

    def test_monomer_vs_complex_scorer_properties(self):
        """Test that MonomerScorer has different properties than ComplexScorer"""
        from bopep.scoring.complex_scorer import ComplexScorer
        
        monomer_scorer = MonomerScorer()
        complex_scorer = ComplexScorer()
        
        # MonomerScorer should have confidence scores
        assert hasattr(monomer_scorer, 'confidence_scores')
        
        # ComplexScorer should have docking scores
        assert hasattr(complex_scorer, 'core_docking_scores')
        
        # Both inherit from BaseScorer so both should have sequence properties
        assert 'molecular_weight' in monomer_scorer.available_scores
        assert 'molecular_weight' in complex_scorer.available_scores

    def test_monomer_scorer_no_binding_scores(self):
        """Test that MonomerScorer doesn't include binding-related scores"""
        scorer = MonomerScorer()
        available = scorer.available_scores
        
        # Should not have binding-specific scores
        assert 'n_contacts' not in available
        assert 'binding_site_n_contacts' not in available
        assert 'peptide_rmsd' not in available
        
        # Should have confidence scores
        assert 'ptm' in available
        assert 'plddt' in available
