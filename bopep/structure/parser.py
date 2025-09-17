from typing import Optional
from Bio.PDB import PDBParser, MMCIFParser, Structure
import os
from Bio.Data.IUPACData import protein_letters_3to1

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


def extract_sequence_from_structure(structure_file: str, chain_id: str = "A") -> str:
    """
    Extract the amino acid sequence from a structure file for a specific chain.
    
    This function uses BioPython's amino acid mapping for robust sequence extraction
    from both PDB and CIF format files.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
    chain_id : str, default "A"
        Chain identifier to extract sequence from
        
    Returns
    -------
    str
        Single-letter amino acid sequence for the specified chain
        
    Examples
    --------
    >>> seq = extract_sequence_from_structure("protein.pdb", "A")
    >>> seq = extract_sequence_from_structure("protein.cif", "B")
    """

    
    structure = parse_structure(structure_file, structure_id="sequence_extraction")
    
    # Use only the first model to avoid repeated sequences for NMR/multi-model structures
    model = structure[0]
    
    sequence = ""
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                # Only consider standard amino acid residues
                if residue.id[0] == ' ':
                    resname = residue.get_resname()
                    # Use BioPython's standard mapping with fallback to 'X' for unknown
                    aa_letter = protein_letters_3to1.get(resname, 'X')
                    sequence += aa_letter
            break
    
    return sequence


def get_chain_sequences(structure_file: str) -> dict:
    """
    Extract amino acid sequences for all chains in a structure file.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
        
    Returns
    -------
    dict
        Dictionary mapping chain IDs to their amino acid sequences
        
    Examples
    --------
    >>> sequences = get_chain_sequences("protein.pdb")
    >>> print(sequences)  # {'A': 'MKLAVF...', 'B': 'EILVGD...'}
    """
    from Bio.Data.IUPACData import protein_letters_3to1
    
    structure = parse_structure(structure_file, structure_id="chain_sequences")
    
    chain_sequences = {}
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                # Only consider standard amino acid residues
                if residue.id[0] == ' ':
                    resname = residue.get_resname()
                    # Use BioPython's standard mapping with fallback to 'X' for unknown
                    aa_letter = protein_letters_3to1.get(resname, 'X')
                    sequence += aa_letter
            
            if sequence:  # Only add chains that have amino acid residues
                chain_sequences[chain.id] = sequence
    
    return chain_sequences


def check_starting_index_in_structure(structure_file: str) -> int:
    """
    Check the starting residue index in a structure file.
    
    Some structure files are not 0-indexed, this function helps identify
    the actual starting residue number.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
        
    Returns
    -------
    int or None
        The starting residue index, or None if no valid residue is found
        
    Examples
    --------
    >>> start_idx = check_starting_index_in_structure("protein.pdb")
    >>> print(f"Structure starts at residue {start_idx}")
    """
    try:
        structure = parse_structure(structure_file, structure_id="check_index")
        model = structure[0]
        
        for chain in model:
            for residue in chain:
                # Skip non-amino acid residues
                if residue.id[0] == ' ':
                    return residue.id[1]  # Return the residue sequence number
        
        return None
        
    except FileNotFoundError:
        print(f"Error: structure file {structure_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading structure file: {e}")
        return None


def get_structure_residues(structure_file: str) -> list:
    """
    Get a list of (chain_id, residue_number) tuples for all residues in a structure.
    
    This replaces manual PDB parsing with BioPython-based extraction that works
    with all supported structure formats (PDB, CIF, PDBX, MMCIF).
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
        
    Returns
    -------
    list of tuple
        List of (chain_id, residue_number) tuples in structure order
        
    Examples
    --------
    >>> residues = get_structure_residues("protein.pdb")
    >>> print(residues)  # [('A', 1), ('A', 2), ..., ('B', 1), ('B', 2), ...]
    """
    residue_chain_list = []
    
    try:
        structure = parse_structure(structure_file, structure_id="residue_list")
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Only consider standard amino acid residues
                    if residue.id[0] == ' ':
                        chain_id = chain.id
                        residue_num = residue.id[1]  # Residue sequence number
                        residue_chain_list.append((chain_id, str(residue_num)))
        
    except Exception as e:
        print(f"Error reading structure file: {e}")
        
    return residue_chain_list


def get_chain_coordinates(structure_file: str, chain_id: str, atom_type: str = "CA") -> list:
    """
    Extract coordinates for a specific atom type from a specific chain.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
    chain_id : str
        Chain identifier to extract coordinates from
    atom_type : str, default "CA"
        Atom type to extract (e.g., "CA", "CB", "N", "C", "O")
        
    Returns
    -------
    list
        List of coordinate arrays (x, y, z) for the specified atom type
        
    Examples
    --------
    >>> coords = get_chain_coordinates("protein.pdb", "A", "CA")
    >>> coords = get_chain_coordinates("protein.cif", "B", "CB")
    """
    structure = parse_structure(structure_file, structure_id="coordinate_extraction")
    
    coordinates = []
    model = structure[0]  # Use first model
    
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                # Only consider standard amino acid residues
                if residue.id[0] == ' ':
                    if atom_type in residue:
                        atom = residue[atom_type]
                        coordinates.append(atom.get_coord())
            break
    
    return coordinates


def get_chain_list(structure_file: str) -> list:
    """
    Get a list of all chain IDs in a structure file.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
        
    Returns
    -------
    list
        List of chain IDs present in the structure
        
    Examples
    --------
    >>> chains = get_chain_list("protein.pdb")
    >>> print(chains)  # ['A', 'B', 'C']
    """
    structure = parse_structure(structure_file, structure_id="chain_list")
    
    chain_ids = []
    for model in structure:
        for chain in model:
            if chain.id not in chain_ids:
                chain_ids.append(chain.id)
    
    return sorted(chain_ids)


def get_all_atom_coordinates(structure_file: str, chain_id: str) -> dict:
    """
    Get coordinates for all atoms in a specific chain, organized by residue.
    
    Parameters
    ----------
    structure_file : str
        Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
    chain_id : str
        Chain identifier to extract coordinates from
        
    Returns
    -------
    dict
        Dictionary mapping residue numbers to atom coordinate dictionaries
        Format: {residue_num: {atom_name: coordinates}}
        
    Examples
    --------
    >>> all_coords = get_all_atom_coordinates("protein.pdb", "A")
    >>> ca_coord = all_coords[1]["CA"]  # CA coordinates for residue 1
    """
    structure = parse_structure(structure_file, structure_id="all_coordinates")
    
    residue_coords = {}
    model = structure[0]  # Use first model
    
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                # Only consider standard amino acid residues
                if residue.id[0] == ' ':
                    residue_num = residue.id[1]
                    atom_coords = {}
                    
                    for atom in residue:
                        atom_coords[atom.name] = atom.get_coord()
                    
                    residue_coords[residue_num] = atom_coords
            break
    
    return residue_coords


def get_residue_coordinates(structure_file: str, chain_id: str, residue_indices: list[int], atom_type: str = None) -> list:
    """
    Extract coordinates for specific residues by their zero-based indices.
    
    Parameters
    ----------
    structure_file : str
        Path to structure file (PDB/CIF/PDBX/MMCIF)
    chain_id : str
        Chain identifier
    residue_indices : list[int]
        Zero-based indices of residues to extract (0 = first residue in chain)
    atom_type : str, optional
        Specific atom type to extract (e.g., 'CA', 'CB'). If None, all atoms are extracted.
        
    Returns
    -------
    list
        List of coordinate arrays (x, y, z) for the specified residues
        
    Examples
    --------
    >>> coords = get_residue_coordinates("protein.pdb", "A", [0, 1, 2], "CA")
    >>> coords = get_residue_coordinates("protein.pdb", "B", [5, 10, 15])  # All atoms
    """
    structure = parse_structure(structure_file)
    model = structure[0]
    
    if chain_id not in model:
        return []
        
    chain = model[chain_id]
    
    # Get only standard amino acid residues
    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    
    coordinates = []
    for idx in residue_indices:
        if 0 <= idx < len(residues):
            residue = residues[idx]
            if atom_type:
                # Extract specific atom type
                if atom_type in residue:
                    coordinates.append(residue[atom_type].get_coord())
            else:
                # Extract all atoms
                for atom in residue.get_atoms():
                    coordinates.append(atom.get_coord())
                    
    return coordinates


def get_all_chain_atoms(structure_file: str, chain_id: str, atom_type: str = None, return_atoms: bool = False):
    """
    Get all atoms or coordinates from a chain with optional filtering.
    
    Parameters
    ----------
    structure_file : str
        Path to structure file (PDB/CIF/PDBX/MMCIF)
    chain_id : str
        Chain identifier
    atom_type : str, optional
        Specific atom type to extract (e.g., 'CA', 'CB'). If None, all atoms are extracted.
    return_atoms : bool, optional
        If True, return Bio.PDB.Atom objects. If False, return coordinates.
        
    Returns
    -------
    list
        List of atom coordinates (x, y, z) if return_atoms=False, or atom objects if return_atoms=True
        
    Examples
    --------
    >>> coords = get_all_chain_atoms("protein.pdb", "A", "CA")
    >>> atoms = get_all_chain_atoms("protein.pdb", "B", return_atoms=True)
    """
    structure = parse_structure(structure_file)
    model = structure[0]
    
    if chain_id not in model:
        return []
        
    chain = model[chain_id]
    
    # Get only standard amino acid residues
    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    
    results = []
    for residue in residues:
        if atom_type:
            # Extract specific atom type
            if atom_type in residue:
                atom = residue[atom_type]
                results.append(atom if return_atoms else atom.get_coord())
        else:
            # Extract all atoms
            for atom in residue.get_atoms():
                results.append(atom if return_atoms else atom.get_coord())
                
    return results