from unittest.mock import patch
import pytest
from bopep.scoring.scorer import Scorer
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.scoring.scores_to_objective import ScoresToObjective


class TestScorer:
    """Test the main scoring functionality"""

    def test_init(self):
        """Test scorer initialization"""
        scorer = Scorer()
        assert scorer.available_scores is not None
        assert len(scorer.available_scores) > 0
        assert "rosetta_score" in scorer.available_scores
        assert "molecular_weight" in scorer.available_scores

    def test_available_scores_comprehensive(self):
        """Test that all expected score types are available"""
        scorer = Scorer()
        expected_scores = [
            "rosetta_score", 
            "interface_sasa", 
            "molecular_weight", 
            "aromaticity",
            "distance_score"
        ]
        
        for score in expected_scores:
            assert score in scorer.available_scores


class TestPeptideProperties:
    """Test peptide property calculations"""

    def test_init_with_sequence(self):
        """Test initialization with peptide sequence"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        assert props.peptide_sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert props.pa is not None

    def test_molecular_weight(self):
        """Test molecular weight calculation"""
        props = PeptideProperties(peptide_sequence="ACDEF")
        mw = props.get_molecular_weight()
        assert isinstance(mw, float)
        assert mw > 0

    def test_aromaticity(self):
        """Test aromaticity calculation"""
        props_aromatic = PeptideProperties(peptide_sequence="FFFWWWYYY")
        props_aliphatic = PeptideProperties(peptide_sequence="ALAGLY")
        
        aromatic_score = props_aromatic.get_aromaticity()
        aliphatic_score = props_aliphatic.get_aromaticity()
        
        assert isinstance(aromatic_score, float)
        assert isinstance(aliphatic_score, float)
        assert aromatic_score > aliphatic_score

    def test_instability_index(self):
        """Test instability index calculation"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        ii = props.get_instability_index()
        assert isinstance(ii, float)

    def test_isoelectric_point(self):
        """Test isoelectric point calculation"""
        props = PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWY")
        pi = props.get_isoelectric_point()
        assert isinstance(pi, float)
        assert pi > 0

    def test_gravy(self):
        """Test GRAVY calculation"""
        props_hydrophobic = PeptideProperties(peptide_sequence="ILVMAF")
        props_hydrophilic = PeptideProperties(peptide_sequence="KRHNED")
        
        hydrophobic_gravy = props_hydrophobic.get_gravy()
        hydrophilic_gravy = props_hydrophilic.get_gravy()
        
        assert isinstance(hydrophobic_gravy, float)
        assert isinstance(hydrophilic_gravy, float)
        assert hydrophobic_gravy > hydrophilic_gravy

    def test_invalid_sequence(self):
        """Test handling of invalid amino acid sequences"""
        with pytest.raises(ValueError):
            PeptideProperties(peptide_sequence="XYZ123")
        with pytest.raises(ValueError):
            PeptideProperties(peptide_sequence="ACDEFGHIKLMNPQRSTVWYXYZ")


class TestScoresToObjective:
    """Test score to objective conversion"""

    def test_init_default(self):
        """Test default initialization"""
        converter = ScoresToObjective()
        assert converter is not None

    @patch('bopep.scoring.scores_to_objective.bopep_objective')
    def test_create_objective_basic(self, mock_objective):
        """Test basic objective creation"""
        mock_objective.return_value = {"ACDEF": 0.8, "GHIKL": 0.6}
        
        converter = ScoresToObjective()
        scores = {
            "ACDEF": {"rosetta_score": -5.0, "interface_sasa": 100.0},
            "GHIKL": {"rosetta_score": -3.0, "interface_sasa": 80.0}
        }
        
        result = converter.create_objective(scores)
        
        assert len(result) == 2
        assert "ACDEF" in result
        assert "GHIKL" in result
