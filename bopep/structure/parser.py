from typing import Optional
from Bio.PDB import PDBParser, MMCIFParser, Structure
import os


class StructureParser:
    """
    Unified structure parser for PDB and CIF files.
    
    This class provides a consistent interface for parsing structural data
    from both PDB and CIF file formats, automatically selecting the appropriate
    BioPython parser based on the file extension.
    """
    
    def __init__(self, quiet: bool = True, auth_residues: bool = False):
        """
        Initialize the StructureParser.
        """
        self.quiet = quiet
        self.auth_residues = auth_residues
    
    def parse(self, filepath: str, structure_id: Optional[str] = None) -> Structure:
        """
        Parse a structure file (PDB or CIF format).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Structure file not found: {filepath}")
        
        if structure_id is None:
            structure_id = os.path.splitext(os.path.basename(filepath))[0]
        
        parser = self._get_parser(filepath)
        return parser.get_structure(structure_id, filepath)
    
    def _get_parser(self, filepath: str):
        """
        Get the appropriate BioPython parser based on file extension.
        """
        filepath_lower = filepath.lower()
        
        if filepath_lower.endswith('.cif') or filepath_lower.endswith('.pdbx') or filepath_lower.endswith('.mmcif'):
            return MMCIFParser(QUIET=self.quiet, auth_residues=self.auth_residues)
        elif filepath_lower.endswith('.pdb'):
            return PDBParser(QUIET=self.quiet)
        else:
            raise ValueError(
                f"Unsupported file format: {filepath}. "
                "Supported formats are .pdb and .cif"
            )
    
    @staticmethod
    def parse_structure(filepath: str, structure_id: Optional[str] = None, 
                       quiet: bool = True, auth_residues: bool = False) -> Structure:
        """
        Convenience static method for parsing structure files.
        """
        parser = StructureParser(quiet=quiet, auth_residues=auth_residues)
        return parser.parse(filepath, structure_id)
    
    @staticmethod
    def get_supported_formats():
        """
        Get list of supported file formats.
        """
        return ['.pdb', '.cif', '.pdbx', '.mmcif']
    
    @staticmethod
    def is_supported_format(filepath: str) -> bool:
        """
        Check if a file format is supported.
        """
        filepath_lower = filepath.lower()
        return any(filepath_lower.endswith(ext) for ext in StructureParser.get_supported_formats())


def parse_structure(filepath: str, structure_id: Optional[str] = None, 
                   quiet: bool = True, auth_residues: bool = False) -> Structure:
    """
    Convenience function for parsing structure files.
    
    This is a shorthand for StructureParser.parse_structure() that can be imported
    directly for quick usage.
    
    Parameters
    ----------
    filepath : str
        Path to the structure file (.pdb or .cif)
    structure_id : str, optional
        Identifier for the structure. If None, uses the filename without extension
    quiet : bool, default True
        If True, suppress BioPython parser warnings
    auth_residues : bool, default False
        For CIF files, whether to use author residue numbering
    
    Returns
    -------
    Bio.PDB.Structure.Structure
        Parsed structure object
        
    Examples
    --------
    >>> from bopep.structure.parser import parse_structure
    >>> structure = parse_structure("protein.pdb")
    >>> structure = parse_structure("protein.cif", structure_id="my_protein")
    """
    return StructureParser.parse_structure(filepath, structure_id, quiet, auth_residues)