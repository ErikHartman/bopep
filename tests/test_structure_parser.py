"""
Tests for the StructureParser class.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from Bio.PDB.Structure import Structure

from bopep.structure.parser import StructureParser, parse_structure


class TestStructureParser:
    """Test cases for StructureParser class."""
    
    def test_init_default_params(self):
        """Test StructureParser initialization with default parameters."""
        parser = StructureParser()
        assert parser.quiet is True
        assert parser.auth_residues is False
    
    def test_init_custom_params(self):
        """Test StructureParser initialization with custom parameters."""
        parser = StructureParser(quiet=False, auth_residues=True)
        assert parser.quiet is False
        assert parser.auth_residues is True
    
    def test_get_supported_formats(self):
        """Test that supported formats are correctly returned."""
        formats = StructureParser.get_supported_formats()
        expected = ['.pdb', '.cif', '.pdbx', '.mmcif']
        assert formats == expected
    
    @pytest.mark.parametrize("filepath,expected", [
        ("test.pdb", True),
        ("test.cif", True),
        ("test.pdbx", True),
        ("test.mmcif", True),
        ("TEST.PDB", True),  # Case insensitive
        ("TEST.CIF", True),
        ("test.xyz", False),
        ("test.txt", False),
        ("test", False),
    ])
    def test_is_supported_format(self, filepath, expected):
        """Test format support detection."""
        assert StructureParser.is_supported_format(filepath) == expected
    
    @patch('bopep.structure.parser.PDBParser')
    def test_get_parser_pdb(self, mock_pdb_parser):
        """Test that PDBParser is used for .pdb files."""
        parser = StructureParser(quiet=False)
        parser._get_parser("test.pdb")
        mock_pdb_parser.assert_called_once_with(QUIET=False)
    
    @patch('bopep.structure.parser.MMCIFParser')
    def test_get_parser_cif(self, mock_mmcif_parser):
        """Test that MMCIFParser is used for .cif files."""
        parser = StructureParser(quiet=False, auth_residues=True)
        parser._get_parser("test.cif")
        mock_mmcif_parser.assert_called_once_with(QUIET=False, auth_residues=True)
    
    @patch('bopep.structure.parser.MMCIFParser')
    def test_get_parser_pdbx(self, mock_mmcif_parser):
        """Test that MMCIFParser is used for .pdbx files."""
        parser = StructureParser()
        parser._get_parser("test.pdbx")
        mock_mmcif_parser.assert_called_once_with(QUIET=True, auth_residues=False)
    
    @patch('bopep.structure.parser.MMCIFParser')
    def test_get_parser_mmcif(self, mock_mmcif_parser):
        """Test that MMCIFParser is used for .mmcif files."""
        parser = StructureParser()
        parser._get_parser("test.mmcif")
        mock_mmcif_parser.assert_called_once_with(QUIET=True, auth_residues=False)
    
    def test_get_parser_unsupported_format(self):
        """Test that unsupported formats raise ValueError."""
        parser = StructureParser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser._get_parser("test.xyz")
    
    @patch('bopep.structure.parser.os.path.exists')
    def test_parse_file_not_found(self, mock_exists):
        """Test that FileNotFoundError is raised for missing files."""
        mock_exists.return_value = False
        parser = StructureParser()
        
        with pytest.raises(FileNotFoundError, match="Structure file not found"):
            parser.parse("nonexistent.pdb")
    
    def test_parse_structure_id_from_filename(self):
        """Test that structure_id is derived from filename when not provided."""
        parser = StructureParser()
        # Use actual data file
        test_file = "data/1ssc.pdb"
        result = parser.parse(test_file)
        
        # Check that the structure_id was set correctly
        assert result.id == "1ssc"
        assert hasattr(result, 'get_models')

    def test_parse_custom_structure_id(self):
        """Test parsing with custom structure_id."""
        parser = StructureParser()
        # Use actual data file
        test_file = "data/4glf.pdb"
        result = parser.parse(test_file, structure_id="custom_id")
        
        # Check that the custom structure_id was used
        assert result.id == "custom_id"
        assert hasattr(result, 'get_models')
    
    @patch('bopep.structure.parser.StructureParser.parse')
    def test_parse_structure_static_method(self, mock_parse):
        """Test the static parse_structure method."""
        mock_structure = MagicMock()
        mock_parse.return_value = mock_structure
        
        result = StructureParser.parse_structure(
            "test.pdb", 
            structure_id="test", 
            quiet=False, 
            auth_residues=True
        )
        
        # Verify that a StructureParser was created with correct params
        # and parse was called
        assert result == mock_structure
        mock_parse.assert_called_once_with("test.pdb", "test")


class TestConvenienceFunction:
    """Test cases for the convenience parse_structure function."""
    
    @patch('bopep.structure.parser.StructureParser.parse_structure')
    def test_parse_structure_function(self, mock_static_method):
        """Test the convenience parse_structure function."""
        mock_structure = MagicMock()
        mock_static_method.return_value = mock_structure
        
        result = parse_structure(
            "test.pdb", 
            structure_id="test", 
            quiet=False, 
            auth_residues=True
        )
        
        mock_static_method.assert_called_once_with(
            "test.pdb", "test", False, True
        )
        assert result == mock_structure


class TestIntegration:
    """Integration tests using real sample data."""
    
    def create_sample_pdb_content(self):
        """Create minimal valid PDB content for testing."""
        return """HEADER    TEST PROTEIN                            01-JAN-00   TEST
ATOM      1  N   ALA A   1      20.154  16.967  12.931  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.715  12.025  1.00 20.00           C  
ATOM      3  C   ALA A   1      18.573  15.277  12.028  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.943  14.836  12.975  1.00 20.00           O  
ATOM      5  CB  ALA A   1      17.857  17.609  12.380  1.00 20.00           C  
END
"""
    
    def test_parse_real_pdb_file(self):
        """Test parsing a real PDB file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(self.create_sample_pdb_content())
            temp_path = f.name
        
        try:
            parser = StructureParser()
            structure = parser.parse(temp_path)
            
            # Verify basic structure properties
            assert isinstance(structure, Structure)
            assert len(list(structure.get_models())) == 1
            
            model = structure[0]
            chains = list(model.get_chains())
            assert len(chains) == 1
            assert chains[0].id == 'A'
            
            residues = list(chains[0].get_residues())
            assert len(residues) == 1
            assert residues[0].get_resname() == 'ALA'
            
        finally:
            os.unlink(temp_path)
    
    def test_parse_structure_with_different_extensions(self):
        """Test that case insensitive extensions work."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.PDB', delete=False) as f:
            f.write(self.create_sample_pdb_content())
            temp_path = f.name
        
        try:
            structure = parse_structure(temp_path)
            assert isinstance(structure, Structure)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])