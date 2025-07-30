import pytest
import tempfile

from bopep.docking.docker import Docker
from bopep.docking.utils import docking_folder_exists, extract_sequence_from_pdb


class TestDocker:
    """Test the Docker class for peptide docking"""

    def test_init_default(self):
        """Test Docker initialization with default parameters"""
        docker_kwargs = {"output_dir": "/tmp"}
        docker = Docker(docker_kwargs)
        
        assert docker.num_models == 5
        assert docker.num_recycles == 10
        assert docker.recycle_early_stop_tolerance == 0.01
        assert docker.amber
        assert docker.num_relax == 1
        assert docker.output_dir == "/tmp"
        assert docker.gpu_ids == [0]
        assert not docker.overwrite_results

    def test_init_custom(self):
        """Test Docker initialization with custom parameters"""
        docker_kwargs = {
            "num_models": 3,
            "num_recycles": 5,
            "amber": False,
            "output_dir": "/custom",
            "gpu_ids": [0, 1],
            "overwrite_results": True
        }
        docker = Docker(docker_kwargs)
        
        assert docker.num_models == 3
        assert docker.num_recycles == 5
        assert not docker.amber
        assert docker.output_dir == "/custom"
        assert docker.gpu_ids == [0, 1]
        assert docker.overwrite_results

    def test_init_missing_output_dir(self):
        """Test Docker initialization fails without output_dir"""
        docker_kwargs = {"num_models": 5}
        
        with pytest.raises(KeyError):
            Docker(docker_kwargs)

    def test_set_target_structure_basic(self):
        """Test setting target structure with valid file"""
        docker_kwargs = {"output_dir": "/tmp"}
        docker = Docker(docker_kwargs)
        
        # Just test that the attribute is set (don't actually read file)
        docker.target_structure_path = "test.pdb"
        assert docker.target_structure_path == "test.pdb"

    def test_set_target_structure_nonexistent_file(self):
        """Test setting target structure with nonexistent file"""
        docker_kwargs = {"output_dir": "/tmp"}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(FileNotFoundError):
            docker.set_target_structure("nonexistent.pdb")

    def test_dock_peptides_empty_list(self):
        """Test docking with empty peptide list but no target set"""
        docker_kwargs = {"output_dir": "/tmp"}
        docker = Docker(docker_kwargs)
        
        with pytest.raises(ValueError, match="Target structure not set"):
            docker.dock_peptides([])


class TestDockingUtils:
    """Test docking utility functions"""

    def test_extract_sequence_from_pdb_nonexistent_file(self):
        """Test sequence extraction from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            extract_sequence_from_pdb("nonexistent.pdb")

    def test_docking_folder_exists_false(self):
        """Test docking folder existence check when folder doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_name = "test_target"
            peptide = "ACDEF"
            
            result, path = docking_folder_exists(temp_dir, target_name, peptide)
            
            assert not result
            assert temp_dir in path
