from unittest.mock import patch
import pytest
import os
import tempfile
import shutil
import json
import yaml
from typing import List, Tuple
from bopep.docking.docker import Docker
from bopep.docking.boltz_docker import BoltzDocker
from bopep.docking.base_docking_model import BaseDockingModel
from bopep.structure.parser import extract_sequence_from_structure


class TestDocker:
    """Test the Docker class for peptide docking"""

    def test_init_default(self):
        """Test Docker initialization with default parameters"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        assert docker.output_dir == "/tmp"
        assert docker.models == ["alphafold"]

    def test_init_custom(self):
        """Test Docker initialization with custom parameters"""
        docker_kwargs = {
            "models": ["alphafold", "boltz"],
            "num_models": 3,
            "num_recycles": 5,
            "amber": False,
            "output_dir": "/tmp",
            "gpu_ids": [0, 1],
            "overwrite_results": True
        }
        docker = Docker(docker_kwargs)

        assert docker.models == ["alphafold", "boltz"]
        assert docker.output_dir == "/tmp"

    

    def test_set_target_structure_nonexistent_file(self):
        """Test setting target structure with nonexistent file"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(FileNotFoundError):
            docker.set_target_structure("nonexistent.pdb")

    def test_dock_sequences_empty_list(self):
        """Test docking with empty peptide list but no target set"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(ValueError, match="Target structure not set"):
            docker.dock_sequences([])


class TestDockingUtils:
    """Test docking utility functions"""

    def test_extract_sequence_from_structure_nonexistent_file(self):
        """Test sequence extraction from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            extract_sequence_from_structure("nonexistent.pdb")


class MockDockingModel(BaseDockingModel):
    """Mock docking model for testing parallel processing logic."""
    
    def __init__(self, **kwargs):
        self.method_name = "mock"
        super().__init__(**kwargs)
    
    def dock(self, peptide_sequences: List[str], target_structure: str, 
             target_sequence: str, target_name: str) -> List[str]:
        return self._dock_with_common_logic(peptide_sequences, target_structure, 
                                          target_sequence, target_name)
    
    def _dock_single_peptide(self, peptide_sequence: str, target_structure: str,
                           target_sequence: str, target_name: str, gpu_id: str = "0") -> str:
        """Mock single peptide docking - creates a fake raw directory."""
        raw_dir = self._create_raw_peptide_dir(target_name, peptide_sequence)
        # Create a fake output file to indicate successful "docking"
        with open(os.path.join(raw_dir, f"mock_output_{peptide_sequence}.txt"), "w") as f:
            f.write(f"Mock docking output for {peptide_sequence} on GPU {gpu_id}")
        return raw_dir
    
    def process_raw_output(self, raw_peptide_dir: str, peptide_sequence: str, 
                          target_name: str) -> str:
        """Mock raw output processing."""
        processed_dir = self._create_processed_peptide_dir(target_name, peptide_sequence)
        
        # Verify the raw directory contains the expected peptide's data
        expected_file = os.path.join(raw_peptide_dir, f"mock_output_{peptide_sequence}.txt")
        if not os.path.exists(expected_file):
            raise ValueError(f"MAPPING ERROR: Raw directory {raw_peptide_dir} does not contain "
                           f"expected output for peptide {peptide_sequence}!")
        
        # Copy mock data to processed directory
        shutil.copy2(expected_file, os.path.join(processed_dir, f"processed_{peptide_sequence}.txt"))
        
        # Create metrics file
        metrics = {
            "peptide_sequence": peptide_sequence,
            "target_name": target_name,
            "docking_method": "mock",
            "test_metric": f"value_for_{peptide_sequence}"
        }
        self._save_metrics_json(metrics, processed_dir, prefix="mock_metrics")
        
        return processed_dir
    
    def _get_method_parameters(self) -> dict:
        return {"mock_param": "test_value"}
    
    @staticmethod
    def _dock_sequences_for_gpu(sequences: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              method_params: dict) -> List[Tuple[str, str]]:
        """Mock GPU docking that returns (peptide, raw_dir) tuples."""
        output_dir = os.path.dirname(os.path.dirname(raw_output_dir))
        
        temp_docker = MockDockingModel(
            output_dir=output_dir,
            gpu_ids=[gpu_id],
            **method_params
        )
        
        docked_results = []
        for peptide in sequences:
            raw_dir = temp_docker._dock_single_peptide(
                peptide, target_structure, target_sequence, target_name, gpu_id
            )
            if raw_dir:
                docked_results.append((peptide, raw_dir))
        
        return docked_results


class TestParallelProcessing:
    """Test parallel processing peptide-to-directory mapping."""
    
    def test_parallel_peptide_mapping(self):
        """
        Test that parallel processing maintains correct peptide-to-directory mapping.
        
        This is a regression test for a bug where sequences distributed across multiple GPUs
        would get their results mixed up due to incorrect ordering when results are collected.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test sequences - use a mix that would expose ordering bugs
            sequences = ["PEPTIDE_A", "PEPTIDE_B", "PEPTIDE_C", "PEPTIDE_D", "PEPTIDE_E", "PEPTIDE_F"]
            target_name = "test_target"
            
            # Create mock docker with multiple GPUs to trigger parallel processing
            docker = MockDockingModel(
                output_dir=temp_dir,
                gpu_ids=["0", "1"],  # 2 GPUs will trigger parallel processing
                overwrite_results=True
            )
            
            # Mock the target structure and sequence
            mock_target = os.path.join(temp_dir, "mock_target.pdb")
            with open(mock_target, "w") as f:
                f.write("MOCK PDB CONTENT")
            
            mock_sequence = "MOCKSEQUENCE"
            
            # Run the docking simulation
            processed_dirs = docker.dock(sequences, mock_target, mock_sequence, target_name)
            
            # Verify we got results for all sequences
            assert len(processed_dirs) == len(sequences)
            
            # Verify each processed directory contains the correct peptide's data
            for processed_dir in processed_dirs:
                dir_name = os.path.basename(processed_dir)
                expected_peptide = dir_name.replace(f"{target_name}_", "")
                
                # Check that the processed file exists and contains the right peptide
                processed_file = os.path.join(processed_dir, f"processed_{expected_peptide}.txt")
                assert os.path.exists(processed_file), f"Directory {dir_name} missing file for {expected_peptide}"
                
                # Check metrics file contains correct peptide
                metrics_file = os.path.join(processed_dir, "mock_metrics.json")
                assert os.path.exists(metrics_file), f"Metrics file missing in {dir_name}"
                
                with open(metrics_file) as f:
                    metrics = json.load(f)
                assert metrics["peptide_sequence"] == expected_peptide, \
                    f"Metrics in {dir_name} for wrong peptide: expected {expected_peptide}, got {metrics['peptide_sequence']}"
    
    def test_sequential_peptide_mapping(self):
        """Test that sequential processing (single GPU) still works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sequences = ["SEQ_A", "SEQ_B", "SEQ_C"]
            target_name = "test_target"
            
            # Create mock docker with single GPU (sequential processing)
            docker = MockDockingModel(
                output_dir=temp_dir,
                gpu_ids=["0"],  # Single GPU triggers sequential processing
                overwrite_results=True
            )
            
            mock_target = os.path.join(temp_dir, "mock_target.pdb")
            with open(mock_target, "w") as f:
                f.write("MOCK PDB CONTENT")
            
            mock_sequence = "MOCKSEQUENCE"
            
            # Run the docking simulation
            processed_dirs = docker.dock(sequences, mock_target, mock_sequence, target_name)
            
            # Verify we got results for all sequences
            assert len(processed_dirs) == len(sequences)
            
            # Verify mapping is correct (should work the same as parallel)
            for processed_dir in processed_dirs:
                dir_name = os.path.basename(processed_dir)
                expected_peptide = dir_name.replace(f"{target_name}_", "")
                
                processed_file = os.path.join(processed_dir, f"processed_{expected_peptide}.txt")
                assert os.path.exists(processed_file), f"Directory {dir_name} missing file for {expected_peptide}"
    
    def test_empty_peptide_list_parallel(self):
        """Test parallel processing with empty peptide list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            docker = MockDockingModel(
                output_dir=temp_dir,
                gpu_ids=["0", "1"],
                overwrite_results=True
            )
            
            mock_target = os.path.join(temp_dir, "mock_target.pdb")
            with open(mock_target, "w") as f:
                f.write("MOCK PDB CONTENT")
            
            processed_dirs = docker.dock([], mock_target, "MOCK", "test")
            assert processed_dirs == []


class TestBoltzDocker:
    """
    Test the BoltzDocker class for Boltz-based docking.
    """

    def test_init_default(self):
        """Test BoltzDocker initialization with default parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            assert boltz.method_name == "boltz"
            assert boltz.recycling_steps == 3
            assert boltz.diffusion_samples == 1
            assert boltz.output_format == "pdb"
            assert boltz.sampling_steps == 200
            assert boltz.step_scale == 1.638

    def test_init_custom(self):
        """Test BoltzDocker initialization with custom parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(
                output_dir=temp_dir,
                recycling_steps=5,
                diffusion_samples=3,
                output_format="mmcif",
                sampling_steps=100,
                step_scale=2.0
            )
            
            assert boltz.recycling_steps == 5
            assert boltz.diffusion_samples == 3
            assert boltz.output_format == "mmcif"
            assert boltz.sampling_steps == 100
            assert boltz.step_scale == 2.0

    def test_detect_protein_chain_and_sequence(self):
        """Test protein chain detection and sequence extraction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            # Test with the actual 5CR6.cif file
            cif_path = os.path.join("data", "5CR6.cif")
            if os.path.exists(cif_path):
                chain_id, sequence = boltz._detect_protein_chain_and_sequence(cif_path)
                
                # Verify chain D is detected (the actual protein chain in 5CR6.cif)
                assert chain_id == "D"
                
                # Verify sequence is extracted correctly
                assert len(sequence) > 0
                assert sequence.startswith("MANKAVNDFILAMNYDKKKLLTHQGESIENRFIK")
                assert len(sequence) == 467  # Known length of 5CR6 protein
            else:
                pytest.skip("5CR6.cif test file not found")

    def test_detect_protein_chain_no_chains(self):
        """Test protein chain detection with no protein chains"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            # Mock the get_chain_sequences function to return empty dict
            with patch('bopep.docking.boltz_docker.get_chain_sequences', return_value={}):
                with pytest.raises(ValueError, match="No protein chains found"):
                    boltz._detect_protein_chain_and_sequence("mock_file.cif")

    def test_create_yaml_config_with_auto_detection(self):
        """Test YAML configuration creation with automatic chain detection"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            # Test with the actual 5CR6.cif file
            cif_path = os.path.join("data", "5CR6.cif")
            if os.path.exists(cif_path):
                yaml_path = boltz._create_yaml_config(
                    peptide_sequence="ARNPIYDGLCVFY",
                    target_sequence="",  # Empty to test auto-detection
                    target_name="5CR6",
                    output_dir=temp_dir,
                    target_structure=cif_path
                )
                
                # Verify YAML file was created
                assert os.path.exists(yaml_path)
                
                # Verify YAML content
                with open(yaml_path, "r") as f:
                    config = yaml.safe_load(f)
                
                # Check structure
                assert "sequences" in config
                assert "templates" in config
                assert len(config["sequences"]) == 2
                
                # Check protein A (target)
                protein_a = config["sequences"][0]["protein"]
                assert protein_a["id"] == "A"
                assert protein_a["msa"] == "empty"
                assert len(protein_a["sequence"]) == 467  # 5CR6 protein length
                assert protein_a["sequence"].startswith("MANKAVNDFILAMNYDKKKLLTHQGESIENRFIK")
                
                # Check protein B (peptide)
                protein_b = config["sequences"][1]["protein"]
                assert protein_b["id"] == "B"
                assert protein_b["sequence"] == "ARNPIYDGLCVFY"
                assert protein_b["msa"] == "empty"
                
                # Check template configuration
                template = config["templates"][0]
                assert template["chain_id"] == "A"  # Should use "A" to match protein sequence ID
                assert template["cif"].endswith("5CR6.cif")
            else:
                pytest.skip("5CR6.cif test file not found")

    def test_create_yaml_config_with_provided_sequence(self):
        """Test YAML configuration creation with provided target sequence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            cif_path = os.path.join("data", "5CR6.cif")
            if os.path.exists(cif_path):
                # Provide a custom sequence (different from the one in the file)
                custom_sequence = "TESTSEQUENCE"
                
                yaml_path = boltz._create_yaml_config(
                    peptide_sequence="ARNPIYDGLCVFY",
                    target_sequence=custom_sequence,
                    target_name="5CR6",
                    output_dir=temp_dir,
                    target_structure=cif_path
                )
                
                # Verify YAML content uses detected sequence, not provided one
                with open(yaml_path, "r") as f:
                    config = yaml.safe_load(f)
                
                # Should use the detected sequence since it doesn't match
                protein_a = config["sequences"][0]["protein"]
                assert protein_a["sequence"] != custom_sequence
                assert len(protein_a["sequence"]) == 467  # Should be the detected 5CR6 sequence
            else:
                pytest.skip("5CR6.cif test file not found")

    def test_prepare_template_file_cif(self):
        """Test template file preparation for CIF files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            cif_path = os.path.join("data", "5CR6.cif")
            if os.path.exists(cif_path):
                template_path = boltz._prepare_template_file(cif_path, temp_dir)
                
                # Verify file was copied
                assert os.path.exists(template_path)
                assert template_path.endswith("5CR6.cif")
                assert os.path.dirname(template_path) == temp_dir
                
                # Verify it's a copy, not the original
                assert template_path != cif_path
            else:
                pytest.skip("5CR6.cif test file not found")

    def test_prepare_template_file_pdb_error(self):
        """Test template file preparation raises error for PDB files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            # Create a mock PDB file
            pdb_path = os.path.join(temp_dir, "test.pdb")
            with open(pdb_path, "w") as f:
                f.write("MOCK PDB CONTENT")
            
            with pytest.raises(ValueError, match="PDB files are not supported"):
                boltz._prepare_template_file(pdb_path, temp_dir)

    def test_prepare_template_file_unsupported_format(self):
        """Test template file preparation raises error for unsupported formats"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz = BoltzDocker(output_dir=temp_dir)
            
            # Create a mock file with unsupported extension
            unsupported_path = os.path.join(temp_dir, "test.xyz")
            with open(unsupported_path, "w") as f:
                f.write("MOCK CONTENT")
            
            with pytest.raises(ValueError, match="Unsupported file format"):
                boltz._prepare_template_file(unsupported_path, temp_dir)
