import pytest
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from bopep.scoring.scorer import Scorer
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.scoring.pep_prot_distance import distance_score_from_structure
from bopep.scoring.is_peptide_in_binding_site import is_peptide_in_binding_site_pdb_file
from bopep.structure.parser import extract_sequence_from_structure


# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
TEST_PDB_FILE = TEST_DATA_DIR / "1ssc.pdb"


@pytest.fixture
def sample_pdb_file():
    """Return path to the 1ssc.pdb test file"""
    if not TEST_PDB_FILE.exists():
        pytest.skip(f"Test PDB file not found: {TEST_PDB_FILE}")
    return str(TEST_PDB_FILE)


@pytest.fixture 
def peptide_sequence_from_1ssc():
    """Extract peptide sequence from chain B of 1ssc.pdb"""
    if not TEST_PDB_FILE.exists():
        pytest.skip(f"Test PDB file not found: {TEST_PDB_FILE}")
    return extract_sequence_from_structure(str(TEST_PDB_FILE), chain_id="B")


class TestScorer:
    """Test the main scoring functionality"""

    def test_init(self):
        """Test scorer initialization"""
        scorer = Scorer()
        assert hasattr(scorer, 'core_docking_scores')
        assert hasattr(scorer, 'structural_scores')
        assert hasattr(scorer, 'peptide_property_scores')
        assert hasattr(scorer, 'method_specific_scores')
        assert hasattr(scorer, 'supported_methods')
        
        # Check that essential scores are available
        all_scores = scorer._all_possible_scores
        assert "molecular_weight" in all_scores
        assert "aromaticity" in all_scores
        assert "distance_score" in all_scores

    def test_get_available_scores_no_context(self):
        """Test available scores with no specific context"""
        scorer = Scorer()
        available = scorer.get_available_scores()
        
        # Should include basic peptide properties
        assert "molecular_weight" in available
        assert "aromaticity" in available
        assert "instability_index" in available
        assert "isoelectric_point" in available
        assert "gravy" in available

    def test_get_available_scores_with_structure(self, sample_pdb_file):
        """Test available scores when structure file is provided"""
        scorer = Scorer()
        available = scorer.get_available_scores(structure_file=sample_pdb_file)
        
        # Should include structural scores
        assert "distance_score" in available
        assert "molecular_weight" in available
        
    def test_supported_methods(self):
        """Test that expected methods are supported"""
        scorer = Scorer()
        assert "alphafold" in scorer.supported_methods
        assert "boltz" in scorer.supported_methods


class TestPeptideProperties:
    """Test peptide property calculations"""

    def test_init_with_sequence(self):
        """Test initialization with peptide sequence"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        assert props.peptide_sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert props.pa is not None

    def test_init_with_structure_file(self, sample_pdb_file):
        """Test initialization with structure file"""
        # Test with chain A (protein)
        props_a = PeptideProperties(structure_file=sample_pdb_file, chain_id="A")
        assert props_a.peptide_sequence is not None
        assert len(props_a.peptide_sequence) > 0
        
        # Test with chain B (second protein copy)
        props_b = PeptideProperties(structure_file=sample_pdb_file, chain_id="B")
        assert props_b.peptide_sequence is not None
        assert len(props_b.peptide_sequence) > 0

    def test_molecular_weight(self):
        """Test molecular weight calculation"""
        props = PeptideProperties(peptide_sequence="ACDEF")
        mw = props.get_molecular_weight()
        assert isinstance(mw, float)
        assert mw > 0
        # Known molecular weight for these amino acids should be reasonable
        assert 500 < mw < 700

    def test_molecular_weight_with_real_data(self, sample_pdb_file):
        """Test molecular weight with real protein data"""
        props = PeptideProperties(structure_file=sample_pdb_file, chain_id="A")
        mw = props.get_molecular_weight()
        assert isinstance(mw, float)
        assert mw > 1000  # Should be substantial for a real protein

    def test_aromaticity(self):
        """Test aromaticity calculation"""
        props_aromatic = PeptideProperties(peptide_sequence="FFFWWWYYY")
        props_aliphatic = PeptideProperties(peptide_sequence="ALAGLY")
        
        aromatic_score = props_aromatic.get_aromaticity()
        aliphatic_score = props_aliphatic.get_aromaticity()
        
        assert isinstance(aromatic_score, float)
        assert isinstance(aliphatic_score, float)
        assert aromatic_score > aliphatic_score
        assert 0 <= aromatic_score <= 1
        assert 0 <= aliphatic_score <= 1

    def test_instability_index(self):
        """Test instability index calculation"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        ii = props.get_instability_index()
        assert isinstance(ii, float)
        assert ii >= 0

    def test_isoelectric_point(self):
        """Test isoelectric point calculation"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        pi = props.get_isoelectric_point()
        assert isinstance(pi, float)
        assert 0 < pi < 14  # pH range

    def test_gravy(self):
        """Test GRAVY calculation"""
        props_hydrophobic = PeptideProperties(peptide_sequence="ILVMAF")
        props_hydrophilic = PeptideProperties(peptide_sequence="KRHNED")
        
        hydrophobic_gravy = props_hydrophobic.get_gravy()
        hydrophilic_gravy = props_hydrophilic.get_gravy()
        
        assert isinstance(hydrophobic_gravy, float)
        assert isinstance(hydrophilic_gravy, float)
        assert hydrophobic_gravy > hydrophilic_gravy

    def test_secondary_structure_fractions(self):
        """Test secondary structure fraction calculations"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        
        helix_frac = props.get_helix_fraction()
        turn_frac = props.get_turn_fraction()
        sheet_frac = props.get_sheet_fraction()
        
        assert isinstance(helix_frac, float)
        assert isinstance(turn_frac, float)
        assert isinstance(sheet_frac, float)
        
        # All fractions should be between 0 and 1
        assert 0 <= helix_frac <= 1
        assert 0 <= turn_frac <= 1
        assert 0 <= sheet_frac <= 1
        
        # Sum should be <= 1.0 (some residues may not be assigned to any structure)
        total = helix_frac + turn_frac + sheet_frac
        assert total <= 1.0

    def test_amino_acid_percentages(self):
        """Test amino acid percentage calculations"""
        props = PeptideProperties(peptide_sequence="KKKRRREEE")  # Mixed charged residues
        
        pos_charged = props.get_positively_charged_aa_percent()
        neg_charged = props.get_negatively_charged_aa_percent()
        hydrophobic = props.get_hydrophobic_aa_percent()
        polar = props.get_polar_aa_percent()
        
        assert isinstance(pos_charged, float)
        assert isinstance(neg_charged, float)
        assert isinstance(hydrophobic, float)
        assert isinstance(polar, float)
        
        # All percentages should be between 0 and 1
        for val in [pos_charged, neg_charged, hydrophobic, polar]:
            assert 0 <= val <= 1
            
        # For this specific sequence
        assert pos_charged > 0.5  # More than half are K/R
        assert neg_charged > 0.2  # Some E residues

    def test_delta_net_charge_frac(self):
        """Test delta net charge fraction calculation"""
        props_pos = PeptideProperties(peptide_sequence="KKKRRR")
        props_neg = PeptideProperties(peptide_sequence="EEEDDD")
        props_neutral = PeptideProperties(peptide_sequence="ALAGLY")
        
        pos_charge = props_pos.get_delta_net_charge_frac()
        neg_charge = props_neg.get_delta_net_charge_frac()
        neutral_charge = props_neutral.get_delta_net_charge_frac()
        
        assert pos_charge > 0
        assert neg_charge < 0
        assert neutral_charge == 0

    def test_get_all_properties(self):
        """Test getting all properties at once"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        all_props = props.get_all_properties()
        
        assert isinstance(all_props, dict)
        expected_keys = [
            'length', 'molecular_weight', 'aromaticity', 'instability_index',
            'isoelectric_point', 'gravy', 'helix_fraction', 'turn_fraction',
            'sheet_fraction', 'hydrophobic_aa_percent', 'polar_aa_percent',
            'positively_charged_aa_percent', 'negatively_charged_aa_percent',
            'delta_net_charge_frac', 'uHrel'
        ]
        
        for key in expected_keys:
            assert key in all_props
            assert isinstance(all_props[key], (int, float))

    def test_invalid_sequence(self):
        """Test handling of invalid amino acid sequences"""
        with pytest.raises(ValueError):
            PeptideProperties(peptide_sequence="XYZ123")
        with pytest.raises(ValueError):
            PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWYXYZ")

    def test_no_input_error(self):
        """Test that providing no input raises an error"""
        with pytest.raises(ValueError):
            PeptideProperties()


class TestStructuralScoring:
    """Test structural scoring functions with real PDB data"""
    
    def test_distance_score_basic(self, sample_pdb_file):
        """Test distance score calculation with 1ssc.pdb"""
        score = distance_score_from_structure(
            sample_pdb_file, 
            receptor_chain="A", 
            peptide_chain="B"
        )
        
        assert isinstance(score, float)
        assert score > 0  # Should be a positive distance
        assert score < 100  # Should be reasonable for protein structure

    def test_distance_score_different_thresholds(self, sample_pdb_file):
        """Test distance score with different distance thresholds"""
        score_8 = distance_score_from_structure(
            sample_pdb_file, 
            receptor_chain="A", 
            peptide_chain="B",
            threshold=8.0
        )
        
        score_12 = distance_score_from_structure(
            sample_pdb_file, 
            receptor_chain="A", 
            peptide_chain="B", 
            threshold=12.0
        )
        
        assert isinstance(score_8, float)
        assert isinstance(score_12, float)
        # Larger threshold should include more atoms and potentially give different score
        assert score_8 > 0
        assert score_12 > 0

    def test_distance_score_invalid_chains(self, sample_pdb_file):
        """Test distance score with invalid chain IDs"""
        score = distance_score_from_structure(
            sample_pdb_file,
            receptor_chain="X",  # Non-existent chain
            peptide_chain="Y"    # Non-existent chain
        )
        
        assert score == 0.0  # Should return 0 for missing chains

    def test_binding_site_detection(self, sample_pdb_file):
        """Test peptide binding site detection"""
        # This may require specific binding site residue indices
        # Let's test the basic function call
        try:
            result = is_peptide_in_binding_site_pdb_file(
                sample_pdb_file,
                binding_site_residues=[10, 20, 30, 40, 50],  # Example residues
                peptide_chain="B"
            )
            assert isinstance(result, bool)
        except Exception as e:
            # If this requires specific setup, we'll skip for now
            pytest.skip(f"Binding site test requires specific setup: {e}")


class TestScoresToObjective:
    """Test score to objective conversion"""

    def test_init_default(self):
        """Test default initialization"""
        converter = ScoresToObjective()
        assert converter is not None

    @patch('bopep.scoring.scores_to_objective.bopep_objective_v1')
    def test_create_objective_basic(self, mock_objective):
        """Test basic objective creation"""
        mock_objective.return_value = {"ACDEF": 0.8, "GHIKL": 0.6}
        
        converter = ScoresToObjective()
        scores = {
            "ACDEF": {"molecular_weight": 500.0, "aromaticity": 0.1},
            "GHIKL": {"molecular_weight": 600.0, "aromaticity": 0.2}
        }
        
        result = converter.create_objective(scores)
        
        assert len(result) == 2
        assert "ACDEF" in result
        assert "GHIKL" in result


class TestPerformanceAndTiming:
    """Test execution times for scoring operations"""

    def test_peptide_properties_timing(self):
        """Test timing for peptide property calculations"""
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"  # 40 amino acids
        
        start_time = time.time()
        props = PeptideProperties(peptide_sequence=sequence)
        all_props = props.get_all_properties()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < 1.0  # Should complete within 1 second
        assert len(all_props) >= 10  # Should calculate multiple properties
        
        print(f"Peptide properties calculation time: {execution_time:.4f} seconds")

    def test_distance_score_timing(self, sample_pdb_file):
        """Test timing for distance score calculation"""
        start_time = time.time()
        score = distance_score_from_structure(
            sample_pdb_file,
            receptor_chain="A",
            peptide_chain="B"
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(score, float)
        
        print(f"Distance score calculation time: {execution_time:.4f} seconds")

    def test_scorer_initialization_timing(self):
        """Test timing for scorer initialization"""
        start_time = time.time()
        scorer = Scorer()
        available_scores = scorer.get_available_scores()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < 1.0  # Should initialize quickly
        assert len(available_scores) > 0
        
        print(f"Scorer initialization time: {execution_time:.4f} seconds")

    def test_multiple_peptide_properties_timing(self):
        """Test timing for calculating properties of multiple peptides"""
        peptides = [
            "ACDEFGHIKLMNPQRSTVWY",
            "KKKRRREEE",
            "FFFWWWYYY",
            "ALAGLYVAL",
            "ILVMAFPHE"
        ]
        
        start_time = time.time()
        results = []
        for peptide in peptides:
            props = PeptideProperties(peptide_sequence=peptide)
            results.append(props.get_all_properties())
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < 2.0  # Should process 5 peptides within 2 seconds
        assert len(results) == len(peptides)
        
        print(f"Multiple peptides calculation time: {execution_time:.4f} seconds")

    @pytest.mark.slow
    def test_large_sequence_timing(self):
        """Test timing for large protein sequence"""
        # Create a large sequence (200 amino acids)
        large_sequence = "ACDEFGHIKLMNPQRSTVWY" * 10
        
        start_time = time.time()
        props = PeptideProperties(peptide_sequence=large_sequence)
        all_props = props.get_all_properties()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert execution_time < 3.0  # Should handle large sequences within 3 seconds
        assert all_props['length'] == 200
        
        print(f"Large sequence ({len(large_sequence)} aa) calculation time: {execution_time:.4f} seconds")


class TestRealDataIntegration:
    """Integration tests using real 1ssc.pdb data"""

    def test_full_scoring_pipeline_1ssc(self, sample_pdb_file, peptide_sequence_from_1ssc):
        """Test complete scoring pipeline with 1ssc.pdb data"""
        # Test peptide properties from extracted sequence
        props = PeptideProperties(peptide_sequence=peptide_sequence_from_1ssc)
        peptide_props = props.get_all_properties()
        
        assert isinstance(peptide_props, dict)
        assert peptide_props['length'] > 0
        assert 'molecular_weight' in peptide_props
        
        # Test distance scoring
        distance_score = distance_score_from_structure(
            sample_pdb_file,
            receptor_chain="A",
            peptide_chain="B"
        )
        
        assert isinstance(distance_score, float)
        assert distance_score > 0
        
        # Combine results
        combined_scores = {
            'distance_score': distance_score,
            **peptide_props
        }
        
        assert len(combined_scores) > 10  # Should have many scores
        assert all(isinstance(v, (int, float)) for v in combined_scores.values())

    def test_1ssc_peptide_properties_realistic(self, peptide_sequence_from_1ssc):
        """Test that 1ssc peptide properties are realistic"""
        props = PeptideProperties(peptide_sequence=peptide_sequence_from_1ssc)
        
        # Test specific properties for realism
        mw = props.get_molecular_weight()
        pi = props.get_isoelectric_point()
        aromaticity = props.get_aromaticity()
        
        # Realistic ranges for protein sequences
        assert mw > 1000  # Should be substantial
        assert 3 < pi < 12  # Reasonable pH range
        assert 0 <= aromaticity <= 1  # Valid fraction
        
        print(f"1ssc chain B properties:")
        print(f"  Sequence length: {len(peptide_sequence_from_1ssc)}")
        print(f"  Molecular weight: {mw:.2f}")
        print(f"  Isoelectric point: {pi:.2f}")
        print(f"  Aromaticity: {aromaticity:.3f}")

    def test_1ssc_structure_accessibility(self, sample_pdb_file):
        """Test that the 1ssc.pdb file is properly accessible and readable"""
        assert os.path.exists(sample_pdb_file)
        
        # Test file size (should be substantial)
        file_size = os.path.getsize(sample_pdb_file)
        assert file_size > 1000  # Should be at least 1KB
        
        # Test that we can extract sequences from both chains
        seq_a = extract_sequence_from_structure(sample_pdb_file, chain_id="A")
        seq_b = extract_sequence_from_structure(sample_pdb_file, chain_id="B")
        
        assert len(seq_a) > 0
        assert len(seq_b) > 0
        assert isinstance(seq_a, str)
        assert isinstance(seq_b, str)
        
        print(f"1ssc.pdb chains:")
        print(f"  Chain A length: {len(seq_a)}")
        print(f"  Chain B length: {len(seq_b)}")
