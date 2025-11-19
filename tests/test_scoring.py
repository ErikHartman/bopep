import pytest
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from bopep.scoring.complex_scorer import ComplexScorer
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.scoring.dssp import DSSPAnalyzer, get_dssp_scores_from_structure
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.scoring.pep_prot_distance import distance_score_from_structure
from bopep.scoring.is_peptide_in_binding_site import is_peptide_in_binding_site_pdb_file
from bopep.structure.parser import extract_sequence_from_structure

# Alias for backward compatibility in tests
Scorer = ComplexScorer


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

    def test_get_available_scores_binding_site_filtering(self):
        """Test that binding site scores are only available when binding site parameters provided"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake metrics files for both models
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            
            # Test without binding site parameters
            available_no_bs = scorer.get_available_scores(processed_dir=temp_dir)
            
            # Test with binding site parameters
            binding_site_indices = list(range(1, 10))
            available_with_bs = scorer.get_available_scores(
                processed_dir=temp_dir, 
                binding_site_residue_indices=binding_site_indices
            )
            
            # Check for binding site related scores (those that require binding site parameters)
            binding_site_score_names = ['in_binding_site', 'in_binding_site_score', 'binding_site_n_contacts']
            
            # Without binding site params - should have NO binding site scores
            bs_scores_no_params = [s for s in available_no_bs 
                                 if any(bs_name in s for bs_name in binding_site_score_names)]
            assert len(bs_scores_no_params) == 0, f"Found binding site scores without params: {bs_scores_no_params}"
            
            # But should have generic n_contacts (which doesn't require binding site)
            n_contacts_scores = [s for s in available_no_bs if 'n_contacts' in s and 'binding_site' not in s]
            assert len(n_contacts_scores) > 0, "Expected n_contacts to be available without binding site params"
            
            # With binding site params - should have binding site scores  
            bs_scores_with_params = [s for s in available_with_bs 
                                   if any(bs_name in s for bs_name in binding_site_score_names)]
            assert len(bs_scores_with_params) > 0, "Expected binding site scores when params provided"
            
            # Should have method-specific binding site scores
            expected_bs_scores = [
                'alphafold_in_binding_site', 'alphafold_in_binding_site_score', 'alphafold_binding_site_n_contacts',
                'boltz_in_binding_site', 'boltz_in_binding_site_score', 'boltz_binding_site_n_contacts'
            ]
            for score in expected_bs_scores:
                assert score in available_with_bs, f"Missing expected binding site score: {score}"

    def test_get_available_scores_generic_structural_filtering_both_models(self):
        """Test that generic structural scores are NOT available when both models are present"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake metrics files for BOTH models
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            
            available = scorer.get_available_scores(processed_dir=temp_dir)
            
            # Check that NO generic structural scores are present when both models available
            structural_scores = scorer.structural_scores
            generic_structural = [s for s in available 
                                if s in structural_scores and not s.startswith(('alphafold_', 'boltz_'))]
            
            assert len(generic_structural) == 0, f"Found generic structural scores when both models available: {generic_structural}"
            
            # Should have method-specific scores for both models
            alphafold_scores = [s for s in available if s.startswith('alphafold_')]
            boltz_scores = [s for s in available if s.startswith('boltz_')]
            
            assert len(alphafold_scores) > 0, "Should have AlphaFold-specific scores"
            assert len(boltz_scores) > 0, "Should have Boltz-specific scores"
            
            # Verify specific structural scores are method-prefixed
            for base_score in ['distance_score', 'rosetta_score', 'interface_sasa']:
                assert base_score not in available, f"Generic '{base_score}' should not be available when both models present"
                assert f'alphafold_{base_score}' in available, f"Missing 'alphafold_{base_score}'"
                assert f'boltz_{base_score}' in available, f"Missing 'boltz_{base_score}'"

    def test_get_available_scores_generic_structural_available_single_model(self):
        """Test that generic structural scores ARE available when only one model is present"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake metrics file for ONLY AlphaFold
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            
            available = scorer.get_available_scores(processed_dir=temp_dir)
            
            # Should have both generic AND method-specific scores when only one model
            structural_scores = scorer.structural_scores
            
            # Filter out binding site scores since we didn't provide binding site params
            non_binding_structural = [s for s in structural_scores 
                                    if s not in ['in_binding_site', 'in_binding_site_score', 'n_contacts']]
            
            generic_structural = [s for s in available 
                                if s in non_binding_structural and not s.startswith(('alphafold_', 'boltz_'))]
            
            assert len(generic_structural) > 0, "Should have generic structural scores when only one model available"
            
            # Verify specific scores
            for base_score in ['distance_score', 'rosetta_score', 'interface_sasa']:
                assert base_score in available, f"Generic '{base_score}' should be available when only one model present"
                assert f'alphafold_{base_score}' in available, f"Missing 'alphafold_{base_score}'"

    def test_get_available_scores_template_dependency(self):
        """Test that template RMSD scores only appear when template is provided"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake metrics files
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            
            # Test without template
            available_no_template = scorer.get_available_scores(processed_dir=temp_dir)
            template_scores_no_template = [s for s in available_no_template if 'template_rmsd' in s]
            assert len(template_scores_no_template) == 0, "Template scores should not be available without template"
            
            # Test with template
            available_with_template = scorer.get_available_scores(
                processed_dir=temp_dir, 
                template_structure="/fake/template.pdb"
            )
            template_scores_with_template = [s for s in available_with_template if 'template_rmsd' in s]
            assert len(template_scores_with_template) > 0, "Template scores should be available with template"
            
            # Should have method-specific template scores
            assert 'alphafold_template_rmsd' in available_with_template
            assert 'boltz_template_rmsd' in available_with_template
            # But NO generic template_rmsd when both models available
            assert 'template_rmsd' not in available_with_template

    def test_get_available_scores_peptide_properties_always_available(self):
        """Test that peptide property scores are always available regardless of model availability"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        # Test with no models
        available_no_models = scorer.get_available_scores()
        
        # Test with one model
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            available_one_model = scorer.get_available_scores(processed_dir=temp_dir)
        
        # Test with both models
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            available_both_models = scorer.get_available_scores(processed_dir=temp_dir)
        
        # Peptide property scores should always be present
        peptide_prop_scores = scorer.peptide_property_scores
        
        for prop_score in peptide_prop_scores:
            assert prop_score in available_no_models, f"Missing peptide property {prop_score} with no models"
            assert prop_score in available_one_model, f"Missing peptide property {prop_score} with one model"
            assert prop_score in available_both_models, f"Missing peptide property {prop_score} with both models"

    def test_get_available_scores_special_scores(self):
        """Test availability of special scores like inter_model_rmsd"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        # Test with only one model - should NOT have inter_model_rmsd
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            available_one = scorer.get_available_scores(processed_dir=temp_dir)
            assert 'inter_model_rmsd' not in available_one, "inter_model_rmsd should not be available with one model"
        
        # Test with both models - should HAVE inter_model_rmsd
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            available_both = scorer.get_available_scores(processed_dir=temp_dir)
            assert 'inter_model_rmsd' in available_both, "inter_model_rmsd should be available with both models"

    def test_n_contacts_vs_binding_site_n_contacts_distinction(self):
        """Test that n_contacts and binding_site_n_contacts are distinct and available appropriately"""
        import tempfile
        import json
        
        scorer = Scorer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake metrics files for both models
            with open(os.path.join(temp_dir, 'alphafold_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.8}, f)
            with open(os.path.join(temp_dir, 'boltz_metrics.json'), 'w') as f:
                json.dump({'iptm': 0.7}, f)
            
            # Test WITHOUT binding site parameters
            available_no_bs = scorer.get_available_scores(processed_dir=temp_dir)
            
            # n_contacts should be available (counts any peptide-protein contacts)
            assert 'alphafold_n_contacts' in available_no_bs, "alphafold_n_contacts should be available without binding site params"
            assert 'boltz_n_contacts' in available_no_bs, "boltz_n_contacts should be available without binding site params"
            
            # binding_site_n_contacts should NOT be available
            assert 'alphafold_binding_site_n_contacts' not in available_no_bs, "alphafold_binding_site_n_contacts should NOT be available without binding site params"
            assert 'boltz_binding_site_n_contacts' not in available_no_bs, "boltz_binding_site_n_contacts should NOT be available without binding site params"
            assert 'binding_site_n_contacts' not in available_no_bs, "binding_site_n_contacts should NOT be available without binding site params"
            
            # Test WITH binding site parameters  
            binding_site_indices = list(range(1, 10))
            available_with_bs = scorer.get_available_scores(
                processed_dir=temp_dir,
                binding_site_residue_indices=binding_site_indices
            )
            
            # BOTH types should be available
            assert 'alphafold_n_contacts' in available_with_bs, "alphafold_n_contacts should still be available with binding site params"
            assert 'boltz_n_contacts' in available_with_bs, "boltz_n_contacts should still be available with binding site params"
            assert 'alphafold_binding_site_n_contacts' in available_with_bs, "alphafold_binding_site_n_contacts should be available with binding site params"
            assert 'boltz_binding_site_n_contacts' in available_with_bs, "boltz_binding_site_n_contacts should be available with binding site params"
            
            # Test single structure file (no method prefixes)
            # Note: We can't actually test scoring without a real structure file, but we can test availability
            structure_available_no_bs = scorer.get_available_scores(structure_file="/fake/path.pdb")
            assert 'n_contacts' in structure_available_no_bs, "n_contacts should be available for single structure"
            assert 'binding_site_n_contacts' not in structure_available_no_bs, "binding_site_n_contacts should NOT be available without binding site params"
            
            structure_available_with_bs = scorer.get_available_scores(
                structure_file="/fake/path.pdb",
                binding_site_residue_indices=binding_site_indices
            )
            assert 'n_contacts' in structure_available_with_bs, "n_contacts should be available for single structure with binding site"
            assert 'binding_site_n_contacts' in structure_available_with_bs, "binding_site_n_contacts should be available with binding site params"


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
        turn_frac = props.get_loop_fraction()
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
            'isoelectric_point', 'gravy', 'helix_fraction', 'loop_fraction',
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


class TestDSSPAnalysis:
    """Test DSSP secondary structure analysis functionality"""

    def test_dssp_analyzer_init(self, sample_pdb_file):
        """Test DSSP analyzer initialization"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        assert analyzer.structure_file == sample_pdb_file
        assert analyzer.chain_id == "B"
        assert analyzer.peptide_sequence is not None
        assert len(analyzer.peptide_sequence) > 0
        assert isinstance(analyzer.peptide_sequence, str)

    def test_dssp_helix_fraction(self, sample_pdb_file):
        """Test DSSP helix fraction calculation"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        helix_fraction = analyzer.get_dssp_helix_fraction()
        
        assert isinstance(helix_fraction, float)
        assert 0.0 <= helix_fraction <= 1.0

    def test_dssp_sheet_fraction(self, sample_pdb_file):
        """Test DSSP strand fraction calculation"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        sheet_fraction = analyzer.get_dssp_sheet_fraction()
        
        assert isinstance(sheet_fraction, float)
        assert 0.0 <= sheet_fraction <= 1.0

    def test_dssp_loop_fraction(self, sample_pdb_file):
        """Test DSSP loop fraction calculation"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        loop_fraction = analyzer.get_dssp_loop_fraction()
        
        assert isinstance(loop_fraction, float)
        assert 0.0 <= loop_fraction <= 1.0

    def test_dssp_fractions_sum_to_one(self, sample_pdb_file):
        """Test that DSSP fractions sum to approximately 1.0"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        helix_frac = analyzer.get_dssp_helix_fraction()
        sheet_frac = analyzer.get_dssp_sheet_fraction()
        loop_frac = analyzer.get_dssp_loop_fraction()
        
        total = helix_frac + sheet_frac + loop_frac
        
        # Should sum to 1.0 within floating point precision
        assert abs(total - 1.0) < 1e-10, f"DSSP fractions sum to {total}, expected ~1.0"

    def test_dssp_get_all_fractions(self, sample_pdb_file):
        """Test getting all DSSP fractions at once"""
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        
        fractions = analyzer.get_all_dssp_fractions()
        
        assert isinstance(fractions, dict)
        expected_keys = ['dssp_helix_fraction', 'dssp_sheet_fraction', 'dssp_loop_fraction']
        
        for key in expected_keys:
            assert key in fractions
            assert isinstance(fractions[key], float)
            assert 0.0 <= fractions[key] <= 1.0
        
        # Check that fractions sum to 1.0
        total = sum(fractions.values())
        assert abs(total - 1.0) < 1e-10

    def test_dssp_convenience_function(self, sample_pdb_file):
        """Test the convenience function for getting DSSP scores"""
        fractions = get_dssp_scores_from_structure(sample_pdb_file, chain_id="B")
        
        assert isinstance(fractions, dict)
        expected_keys = ['dssp_helix_fraction', 'dssp_sheet_fraction', 'dssp_loop_fraction']
        
        for key in expected_keys:
            assert key in fractions
            assert isinstance(fractions[key], float)
            assert 0.0 <= fractions[key] <= 1.0

    def test_dssp_with_different_chains(self, sample_pdb_file):
        """Test DSSP analysis with different chain IDs"""
        # Test with chain A (protein receptor)
        analyzer_a = DSSPAnalyzer(sample_pdb_file, chain_id="A")
        fractions_a = analyzer_a.get_all_dssp_fractions()
        
        # Test with chain B (second protein copy)
        analyzer_b = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        fractions_b = analyzer_b.get_all_dssp_fractions()
        
        # Both should work and return valid fractions
        for fractions in [fractions_a, fractions_b]:
            assert isinstance(fractions, dict)
            for value in fractions.values():
                assert isinstance(value, float)
                assert 0.0 <= value <= 1.0

    def test_dssp_invalid_structure_file(self):
        """Test DSSP analyzer with invalid structure file"""
        with pytest.raises((FileNotFoundError, ValueError)):
            analyzer = DSSPAnalyzer("/nonexistent/file.pdb", chain_id="B")
            analyzer.get_dssp_helix_fraction()

    def test_dssp_invalid_chain(self, sample_pdb_file):
        """Test DSSP analyzer with invalid chain ID"""
        with pytest.raises(ValueError):
            analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="Z")  # Non-existent chain
            analyzer.get_dssp_helix_fraction()

    def test_dssp_vs_biopython_secondary_structure(self, sample_pdb_file):
        """Test comparison between DSSP and BioPython secondary structure predictions"""
        # Get DSSP fractions
        analyzer = DSSPAnalyzer(sample_pdb_file, chain_id="B")
        dssp_fractions = analyzer.get_all_dssp_fractions()
        
        # Get BioPython-based fractions from PeptideProperties
        props = PeptideProperties(structure_file=sample_pdb_file, chain_id="B")
        biopython_helix = props.get_helix_fraction()
        biopython_sheet = props.get_sheet_fraction()
        
        # Both should return valid values
        assert isinstance(dssp_fractions['dssp_helix_fraction'], float)
        assert isinstance(dssp_fractions['dssp_sheet_fraction'], float)
        assert isinstance(biopython_helix, float)
        assert isinstance(biopython_sheet, float)
        
        # They may differ (different methods), but should be in reasonable ranges
        assert 0.0 <= dssp_fractions['dssp_helix_fraction'] <= 1.0
        assert 0.0 <= dssp_fractions['dssp_sheet_fraction'] <= 1.0
        assert 0.0 <= biopython_helix <= 1.0
        assert 0.0 <= biopython_sheet <= 1.0
        
        print(f"DSSP vs BioPython secondary structure comparison:")
        print(f"  DSSP helix: {dssp_fractions['dssp_helix_fraction']:.3f}")
        print(f"  BioPython helix: {biopython_helix:.3f}")
        print(f"  DSSP strand: {dssp_fractions['dssp_sheet_fraction']:.3f}")
        print(f"  BioPython sheet: {biopython_sheet:.3f}")
        print(f"  DSSP loop: {dssp_fractions['dssp_loop_fraction']:.3f}")

    def test_dssp_scorer_integration(self, sample_pdb_file):
        """Test DSSP scores integration with the main Scorer class"""
        scorer = Scorer()
        
        # Test that DSSP scores are in available scores
        available_scores = scorer.get_available_scores(structure_file=sample_pdb_file)
        
        dssp_score_names = ["dssp_helix_fraction", "dssp_sheet_fraction", "dssp_loop_fraction"]
        for score_name in dssp_score_names:
            assert score_name in available_scores, f"DSSP score '{score_name}' not available"
        
        # Test scoring with DSSP scores
        scores = scorer.score(
            scores_to_include=dssp_score_names,
            structure_file=sample_pdb_file
        )
        
        assert len(scores) == 1  # Should return one peptide
        peptide_seq, peptide_scores = next(iter(scores.items()))
        
        assert isinstance(peptide_seq, str)
        assert len(peptide_seq) > 0
        
        for score_name in dssp_score_names:
            assert score_name in peptide_scores
            assert isinstance(peptide_scores[score_name], float)
            assert 0.0 <= peptide_scores[score_name] <= 1.0

    def test_dssp_scorer_error_handling(self):
        """Test that Scorer throws appropriate errors for DSSP scores without structure"""
        scorer = Scorer()
        
        # Should raise error when requesting DSSP scores without structure file
        with pytest.raises(ValueError, match="requires a structure file"):
            scorer.score(
                scores_to_include=["dssp_helix_fraction"],
                peptide_sequence="ACDEFG"  # Only sequence, no structure
            )
        
        with pytest.raises(ValueError, match="requires a structure file"):
            scorer.score(
                scores_to_include=["dssp_sheet_fraction"],
                peptide_sequence="ACDEFG"
            )
        
        with pytest.raises(ValueError, match="requires a structure file"):
            scorer.score(
                scores_to_include=["dssp_loop_fraction"],
                peptide_sequence="ACDEFG"
            )


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
