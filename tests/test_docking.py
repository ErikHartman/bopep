import pytest
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb


class TestDocker:
    """Test the Docker class for peptide docking"""

    def test_init_default(self):
        """Test Docker initialization with default parameters"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        assert docker.base_output_dir == "/tmp"
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
        assert docker.base_output_dir == "/tmp"

    

    def test_set_target_structure_nonexistent_file(self):
        """Test setting target structure with nonexistent file"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(FileNotFoundError):
            docker.set_target_structure("nonexistent.pdb")

    def test_dock_peptides_empty_list(self):
        """Test docking with empty peptide list but no target set"""
        docker_kwargs = {"output_dir": "/tmp", "models": ["alphafold"]}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(ValueError, match="Target structure not set"):
            docker.dock_peptides([])


class TestDockingUtils:
    """Test docking utility functions"""

    def test_extract_sequence_from_pdb_nonexistent_file(self):
        """Test sequence extraction from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            extract_sequence_from_pdb("nonexistent.pdb")
