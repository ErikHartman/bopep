"""
Tests for the diffusion module (BoRF pipeline).
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from bopep.diffusion.borf import BoRF


@pytest.fixture
def mock_rfdiffusion_path(temp_dir):
    """Create a mock RFDiffusion installation directory."""
    rf_path = Path(temp_dir) / "RFdiffusion"
    rf_path.mkdir()
    
    # Create expected subdirectories and files
    scripts_dir = rf_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_inference.py").touch()
    
    return str(rf_path)


@pytest.fixture
def mock_protein_mpnn_path(temp_dir):
    """Create a mock ProteinMPNN installation directory."""
    mpnn_path = Path(temp_dir) / "ProteinMPNN"
    mpnn_path.mkdir()
    (mpnn_path / "protein_mpnn_run.py").touch()
    
    return str(mpnn_path)


@pytest.fixture
def mock_pdb_file(temp_dir):
    """Create a mock PDB file."""
    pdb_path = Path(temp_dir) / "test_protein.pdb"
    pdb_content = """HEADER    TEST PROTEIN                            01-JAN-00   TEST
ATOM      1  N   ALA A   1      20.154  16.967  12.931  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.715  12.025  1.00 20.00           C  
ATOM      3  C   ALA A   1      18.573  15.277  12.028  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.943  14.836  12.975  1.00 20.00           O  
END
"""
    pdb_path.write_text(pdb_content)
    return str(pdb_path)


@pytest.fixture
def mock_mpnn_env(temp_dir):
    """Create a mock MPNN environment executable."""
    env_path = Path(temp_dir) / "python"
    env_path.touch()
    env_path.chmod(0o755)
    return str(env_path)


@pytest.fixture
def sample_csv_path():
    """Path to the sample CSV file in the data directory."""
    return "./data/sequence_samples.csv"


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return pd.DataFrame({
        'sequence': ['ACDEF', 'GHIKL', 'MNPQR'],
        'length': [5, 5, 5],
        'target_length': [10, 10, 10]
    })


@pytest.fixture
def borf_config(temp_dir, mock_rfdiffusion_path, mock_protein_mpnn_path, mock_pdb_file, mock_mpnn_env):
    """Standard BoRF configuration for testing."""
    return {
        'output_dir': temp_dir,
        'rfdiffusion_path': mock_rfdiffusion_path,
        'protein_mpnn_path': mock_protein_mpnn_path,
        'pdb_path': mock_pdb_file,
        'mpnn_env': mock_mpnn_env
    }


class TestBoRF:
    """Test the main BoRF class."""

    def test_init_valid_config(self, borf_config):
        """Test successful initialization with valid configuration."""
        borf = BoRF(**borf_config)
        
        assert borf.output_dir == Path(borf_config['output_dir'])
        assert borf.config['rfdiffusion_path'] == borf_config['rfdiffusion_path']
        assert borf.config['protein_mpnn_path'] == borf_config['protein_mpnn_path']
        assert borf.config['pdb_path'] == borf_config['pdb_path']
        assert borf.output_dir.exists()

    def test_init_missing_pdb_path(self, temp_dir, mock_rfdiffusion_path, mock_protein_mpnn_path, mock_mpnn_env):
        """Test initialization fails without pdb_path."""
        with pytest.raises(ValueError, match="pdb_path is a mandatory argument"):
            BoRF(
                output_dir=temp_dir,
                rfdiffusion_path=mock_rfdiffusion_path,
                protein_mpnn_path=mock_protein_mpnn_path,
                mpnn_env=mock_mpnn_env,
                pdb_path=None
            )

    def test_init_invalid_rfdiffusion_path(self, temp_dir, mock_protein_mpnn_path, mock_pdb_file, mock_mpnn_env):
        """Test initialization fails with invalid RFDiffusion path."""
        with pytest.raises(ValueError, match="RFDiffusion path does not exist"):
            BoRF(
                output_dir=temp_dir,
                rfdiffusion_path="/nonexistent/path",
                protein_mpnn_path=mock_protein_mpnn_path,
                pdb_path=mock_pdb_file,
                mpnn_env=mock_mpnn_env
            )

    def test_init_invalid_mpnn_env(self, temp_dir, mock_rfdiffusion_path, mock_protein_mpnn_path, mock_pdb_file):
        """Test initialization fails with invalid MPNN environment."""
        with pytest.raises(ValueError, match="MPNN environment path does not exist"):
            BoRF(
                output_dir=temp_dir,
                rfdiffusion_path=mock_rfdiffusion_path,
                protein_mpnn_path=mock_protein_mpnn_path,
                pdb_path=mock_pdb_file,
                mpnn_env="/nonexistent/python"
            )

    def test__validate_configuration_valid(self, borf_config):
        """Test configuration validation with valid setup."""
        borf = BoRF(**borf_config)
        validation = borf._validate_configuration()
        
        assert validation['rfdiffusion_configured'] is True
        assert validation['protein_mpnn_configured'] is True
        assert validation['pdb_file_exists'] is True
        assert validation['output_dir_writable'] is True
        assert validation['rfdiffusion_exists'] is True
        assert validation['rfdiffusion_script_exists'] is True
        assert validation['protein_mpnn_exists'] is True
        assert validation['protein_mpnn_script_exists'] is True

    def test__validate_configuration_missing_components(self, temp_dir, mock_pdb_file, mock_mpnn_env):
        """Test configuration validation with missing components."""
        borf = BoRF(
            output_dir=temp_dir,
            rfdiffusion_path="",
            protein_mpnn_path="", 
            pdb_path=mock_pdb_file,
            mpnn_env=mock_mpnn_env
        )
        validation = borf._validate_configuration()
        
        assert validation['rfdiffusion_configured'] is False
        assert validation['protein_mpnn_configured'] is False
        assert validation['rfdiffusion_exists'] is False
        assert validation['protein_mpnn_exists'] is False

    @patch('bopep.diffusion.borf.RFDiffusion')
    def test_rfdiffusion_property_lazy_initialization(self, mock_rfdiffusion_class, borf_config):
        """Test that RFDiffusion is lazily initialized."""
        mock_instance = Mock()
        mock_rfdiffusion_class.return_value = mock_instance
        
        borf = BoRF(**borf_config)
        
        # Should not be initialized yet
        assert borf._rfdiffusion is None
        
        # Access the property to trigger initialization
        rf_instance = borf.rfdiffusion
        
        # Should now be initialized
        assert borf._rfdiffusion is mock_instance
        assert rf_instance is mock_instance
        mock_rfdiffusion_class.assert_called_once()

    @patch('bopep.diffusion.borf.MPNNFastRelax')
    def test_mpnn_fastrelax_property_lazy_initialization(self, mock_mpnn_class, borf_config):
        """Test that MPNNFastRelax is lazily initialized."""
        mock_instance = Mock()
        mock_mpnn_class.return_value = mock_instance
        
        borf = BoRF(**borf_config)
        
        # Should not be initialized yet
        assert borf._mpnn_fastrelax is None
        
        # Access the property to trigger initialization
        mpnn_instance = borf.mpnn_fastrelax
        
        # Should now be initialized
        assert borf._mpnn_fastrelax is mock_instance
        assert mpnn_instance is mock_instance
        mock_mpnn_class.assert_called_once()


    @patch('bopep.diffusion.borf.RFDiffusion')
    def test_run_rfdiffusion_only(self, mock_rfdiffusion_class, borf_config, sample_csv_path):
        """Test running only the RFDiffusion step."""
        # Setup mocks
        mock_rf_instance = Mock()
        mock_rf_instance.run.return_value = {
            'successful_runs': 5,
            'failed_runs': 0
        }
        mock_rfdiffusion_class.return_value = mock_rf_instance
        
        borf = BoRF(**borf_config)
        
        # Test with provided samples CSV
        results = borf._run_rfdiffusion_only(
            samples_csv=sample_csv_path,
            dry_run=True, 
            skip_existing=False
        )
        
        assert results['successful_runs'] == 5
        assert results['failed_runs'] == 0
        mock_rf_instance.run.assert_called_once_with(
            samples_csv=sample_csv_path,
            dry_run=True,
            skip_existing=False
        )

    @patch('bopep.diffusion.borf.MPNNFastRelax')
    def test_run_mpnn_fastrelax_only(self, mock_mpnn_class, borf_config):
        """Test running only the MPNN + FastRelax step."""
        # Setup mocks
        mock_mpnn_instance = Mock()
        mock_mpnn_instance.run.return_value = {
            'processed_pdbs': 10,
            'output_csv': 'output.csv'
        }
        mock_mpnn_class.return_value = mock_mpnn_instance
        
        borf = BoRF(**borf_config)
        
        results = borf._run_mpnn_fastrelax_only(
            designs_dir="test_diffusions",
            temperature=0.2,
            relax_cycles=2,
            threads=8,
            limited_run=5,
            output_csv="custom_output.csv"
        )
        
        assert results['processed_pdbs'] == 10
        mock_mpnn_instance.run.assert_called_once_with(
            designs_dir="test_diffusions",
            temperature=0.2,
            relax_cycles=2,
            threads=8,
            limited_run=5,
            output_csv="custom_output.csv"
        )

    @patch('bopep.diffusion.borf.RFDiffusion')
    @patch('bopep.diffusion.borf.MPNNFastRelax')
    def test_run_complete_pipeline(self, mock_mpnn_class, mock_rf_class, borf_config, sample_csv_path):
        """Test running the complete pipeline."""
        # Setup mocks
        mock_rf_instance = Mock()
        mock_rf_instance.run.return_value = {
            'successful_runs': 3,
            'failed_runs': 1
        }
        mock_rf_class.return_value = mock_rf_instance
        
        mock_mpnn_instance = Mock()
        mock_mpnn_instance.run.return_value = {
            'processed_pdbs': 3,
            'output_csv': 'final_output.csv',
            'interface_dg_scores': 5
        }
        mock_mpnn_class.return_value = mock_mpnn_instance
        
        borf = BoRF(**borf_config)
        
        results = borf.run(
            samples_csv=sample_csv_path,
            rf_dry_run=True,
            temperature=0.15,
            relax_cycles=2
        )
        
        # Check combined results
        assert results['pipeline_success'] is True
        assert results['total_designs_generated'] == 3
        assert results['total_sequences_optimized'] == 3
        assert results['final_output_csv'] == 'final_output.csv'
        assert 'elapsed_time' in results
        assert 'rfdiffusion_results' in results
        assert 'mpnn_fastrelax_results' in results

    @patch('bopep.diffusion.borf.RFDiffusion')
    @patch('bopep.diffusion.borf.MPNNFastRelax')
    def test_run_pipeline_validation_failure(self, mock_mpnn_class, mock_rf_class, temp_dir, mock_pdb_file, mock_mpnn_env):
        """Test pipeline fails with invalid configuration."""
        # Create BoRF with incomplete configuration
        borf = BoRF(
            output_dir=temp_dir,
            rfdiffusion_path="",  # Invalid path
            protein_mpnn_path="",  # Invalid path
            pdb_path=mock_pdb_file,
            mpnn_env=mock_mpnn_env
        )
        
        with pytest.raises(ValueError, match="Pipeline not properly configured"):
            borf.run()

    @patch('bopep.diffusion.borf.RFDiffusion')
    def test_run_rfdiffusion_with_custom_samples(self, mock_rf_class, borf_config, temp_dir):
        """Test RFDiffusion step with custom samples CSV."""
        mock_rf_instance = Mock()
        mock_rf_instance.run.return_value = {'successful_runs': 2, 'failed_runs': 0}
        mock_rf_class.return_value = mock_rf_instance
        
        custom_csv = str(Path(temp_dir) / "custom_samples.csv")
        
        borf = BoRF(**borf_config)
        results = borf._run_rfdiffusion_only(samples_csv=custom_csv)
        
        mock_rf_instance.run.assert_called_once_with(
            samples_csv=custom_csv,
            dry_run=False,
            skip_existing=True
        )

    def test_rfdiffusion_property_no_path(self, temp_dir, mock_pdb_file, mock_mpnn_env):
        """Test RFDiffusion property raises error when path not configured."""
        borf = BoRF(
            output_dir=temp_dir,
            rfdiffusion_path="",  # Empty path
            protein_mpnn_path="/some/path",
            pdb_path=mock_pdb_file,
            mpnn_env=mock_mpnn_env
        )
        
        with pytest.raises(ValueError, match="RFDiffusion path not specified"):
            _ = borf.rfdiffusion


class TestBoRFIntegration:
    """Integration tests for BoRF that test component interactions."""

    @patch('subprocess.run')
    def test_end_to_end_mock_pipeline(self, mock_subprocess, borf_config, temp_dir, sample_csv_path):
        """Test end-to-end pipeline with mocked external calls."""
        # Mock subprocess calls to succeed
        mock_subprocess.return_value = Mock(returncode=0)
        temp_dir = Path(temp_dir)
        # Create diffusions directory and sample PDB files
        designs_dir = temp_dir / "diffusions"
        designs_dir.mkdir()
        for i in range(2):
            (designs_dir / f"diffusion_{i}.pdb").touch()
        
        with patch('bopep.diffusion.diffusion.RFDiffusion.run') as mock_rf_run, \
             patch('bopep.diffusion.mpnn_fastrelax.MPNNFastRelax.run') as mock_mpnn_run:
            
            mock_rf_run.return_value = {'successful_runs': 2, 'failed_runs': 0}
            mock_mpnn_run.return_value = {
                'processed_pdbs': 2,
                'output_csv': str(temp_dir / 'final_output.csv'),
                'interface_dg_scores': 3
            }
            
            borf = BoRF(**borf_config)
            results = borf.run(samples_csv=sample_csv_path, rf_dry_run=True)
            
            assert results['pipeline_success'] is True
            assert results['total_designs_generated'] == 2
            assert results['total_sequences_optimized'] == 2


    def test_sample_csv_file_format(self, sample_csv_path):
        """Test that the sample CSV file has the expected format and content."""
        assert Path(sample_csv_path).exists(), "Sample CSV file should exist"
        
        df = pd.read_csv(sample_csv_path)
        expected_columns = ['sample_id', 'length', 'hotspots']
        assert list(df.columns) == expected_columns
        assert len(df) > 0
        
        # Check data types and content
        assert df['sample_id'].dtype in ['int64', 'object']
        assert df['length'].dtype == 'int64'
        assert df['hotspots'].dtype == 'object'
        
        # Check that all lengths are positive
        assert (df['length'] > 0).all()
        
        # Check that hotspots contain comma-separated values
        for hotspot in df['hotspots']:
            assert ',' in str(hotspot) or len(str(hotspot).strip()) > 0

    @patch('bopep.diffusion.borf.RFDiffusion')
    def test_rfdiffusion_with_real_csv_data(self, mock_rfdiffusion_class, borf_config, sample_csv_path):
        """Test RFDiffusion step with real CSV data from data directory."""
        mock_rf_instance = Mock()
        mock_rf_instance.run.return_value = {
            'successful_runs': 5,
            'failed_runs': 0
        }
        mock_rfdiffusion_class.return_value = mock_rf_instance
        
        borf = BoRF(**borf_config)
        
        # Test with the real CSV file
        results = borf._run_rfdiffusion_only(samples_csv=sample_csv_path)
        
        assert results['successful_runs'] == 5
        assert results['failed_runs'] == 0
        mock_rf_instance.run.assert_called_once_with(
            samples_csv=sample_csv_path,
            dry_run=False,
            skip_existing=True
        )


if __name__ == "__main__":
    pytest.main([__file__])
