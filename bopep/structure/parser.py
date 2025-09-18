from typing import List, Optional, Tuple
from Bio.PDB import PDBParser, MMCIFParser, Structure
import os
from Bio.SeqUtils import seq1
from .cache import get_from_cache, store_in_cache

class StructureParser:
    """
    Unified structure parser for PDB and CIF files.
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
        
        # Check cache first for parsed structure
        cached_structure = get_from_cache(filepath, 'structure')
        if cached_structure is not None:
            # Create a copy with the requested structure_id to avoid modifying the cached object
            import copy
            structure_copy = copy.copy(cached_structure)
            structure_copy.id = structure_id
            return structure_copy
        
        parser = self._get_parser(filepath)
        structure = parser.get_structure(structure_id, filepath)
        
        # Cache the parsed structure
        store_in_cache(filepath, 'structure', structure)
        return structure
    
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
                f"Supported formats are: {', '.join(self.get_supported_formats())}"
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
    """Convenience function for parsing structure files with caching."""
    return StructureParser.parse_structure(filepath, structure_id, quiet, auth_residues)


def extract_sequence_from_structure(structure_file: str, chain_id: str = "A") -> str:
    """Extract amino acid sequence from a structure file for a specific chain."""
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
                    aa_letter = seq1(resname)
                    sequence += aa_letter
            break
    
    return sequence


def get_chain_sequences(structure_file: str) -> dict:
    """Extract amino acid sequences for all chains in a structure file."""
    # Check cache first
    cached_sequences = get_from_cache(structure_file, 'sequences')
    if cached_sequences is not None:
        return cached_sequences
    
    structure = parse_structure(structure_file, structure_id="chain_sequences")
    
    chain_sequences = {}
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                if residue.id[0] == ' ':
                    resname = residue.get_resname()
                    aa_letter = seq1(resname)
                    sequence += aa_letter
            
            if sequence:  # Only add chains that have amino acid residues
                chain_sequences[chain.id] = sequence
    
    # Store in cache
    store_in_cache(structure_file, 'sequences', chain_sequences)
    return chain_sequences


def check_starting_index_in_structure(structure_file: str) -> Optional[int]:
    """Check the starting residue index in a structure file."""
    structure = parse_structure(structure_file, structure_id="check_index")
    model = structure[0]
    
    for chain in model:
        for residue in chain:
            # Skip non-amino acid residues
            if residue.id[0] == ' ':
                return residue.id[1]  # Return the residue sequence number
    
    return None
    



def get_structure_residues(structure_file: str) -> List[Tuple[str, str]] :
    """Get a list of (chain_id, residue_number) tuples for all residues in a structure."""

    cached_residues = get_from_cache(structure_file, 'structure_residues')
    if cached_residues is not None:
        return cached_residues
    
    residue_chain_list = []
    
    structure = parse_structure(structure_file, structure_id="residue_list")
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only consider standard amino acid residues
                if residue.id[0] == ' ':
                    chain_id = chain.id
                    residue_num = residue.id[1]  # Residue sequence number
                    residue_chain_list.append((chain_id, str(residue_num)))
    

    store_in_cache(structure_file, 'structure_residues', residue_chain_list)
    return residue_chain_list


def get_chain_coordinates(structure_file: str, chain_id: str, atom_type: str = "CA") -> list:
    """Extract coordinates for a specific atom type from a specific chain."""
    coord_key = f'{chain_id}_{atom_type}'
    
    cached_coords = get_from_cache(structure_file, f'coordinates.{coord_key}')
    if cached_coords is not None:
        return cached_coords
    
    structure = parse_structure(structure_file, structure_id="coordinate_extraction")
    
    coordinates = []
    model = structure[0]
    
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                if residue.id[0] == ' ':
                    if atom_type in residue:
                        atom = residue[atom_type]
                        coordinates.append(atom.get_coord())
            break
    
    store_in_cache(structure_file, f'coordinates.{coord_key}', coordinates)
    return coordinates


def get_chain_list(structure_file: str) -> list:
    """Get a list of all chain IDs in a structure file."""
    cached_chains = get_from_cache(structure_file, 'chains')
    if cached_chains is not None:
        return cached_chains
    
    structure = parse_structure(structure_file, structure_id="chain_list")
    
    chain_ids = []
    for model in structure:
        for chain in model:
            if chain.id not in chain_ids:
                chain_ids.append(chain.id)
    
    chain_ids = sorted(chain_ids)
    
    store_in_cache(structure_file, 'chains', chain_ids)
    return chain_ids


def get_all_atom_coordinates(structure_file: str, chain_id: str) -> dict:
    """Get coordinates for all atoms in a specific chain, organized by residue."""
    cache_key = f'all_atom_coords_{chain_id}'
    
    cached_coords = get_from_cache(structure_file, cache_key)
    if cached_coords is not None:
        return cached_coords
    
    structure = parse_structure(structure_file, structure_id="all_coordinates")
    
    residue_coords = {}
    model = structure[0]
    
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                if residue.id[0] == ' ':
                    residue_num = residue.id[1]
                    atom_coords = {}
                    
                    for atom in residue:
                        atom_coords[atom.name] = atom.get_coord()
                    
                    residue_coords[residue_num] = atom_coords
            break
    
    store_in_cache(structure_file, cache_key, residue_coords)
    return residue_coords


def get_residue_coordinates(structure_file: str, chain_id: str, residue_indices: List[int], atom_type: str = None) -> list:
    """Extract coordinates for specific residues by their zero-based indices."""
    indices_str = '_'.join(map(str, residue_indices))
    atom_str = atom_type if atom_type else 'all'
    cache_key = f'residue_coords_{chain_id}_{indices_str}_{atom_str}'
    
    # Check cache first
    cached_coords = get_from_cache(structure_file, cache_key)
    if cached_coords is not None:
        return cached_coords
    
    structure = parse_structure(structure_file)
    model = structure[0]
    
    if chain_id not in model:
        coordinates = []
    else:
        chain = model[chain_id]
        
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        
        coordinates = []
        for idx in residue_indices:
            if 0 <= idx < len(residues):
                residue = residues[idx]
                if atom_type:
                    if atom_type in residue:
                        coordinates.append(residue[atom_type].get_coord())
                else:
                    for atom in residue.get_atoms():
                        coordinates.append(atom.get_coord())
    
    store_in_cache(structure_file, cache_key, coordinates)
    return coordinates


def get_all_chain_atoms(structure_file: str, chain_id: str, atom_type: str = None, return_atoms: bool = False):
    """Get all atoms or coordinates from a chain with optional filtering."""
    atom_str = atom_type if atom_type else 'all'
    return_type = 'atoms' if return_atoms else 'coords'
    cache_key = f'all_chain_{return_type}_{chain_id}_{atom_str}'
    
    cached_results = get_from_cache(structure_file, cache_key)
    if cached_results is not None:
        return cached_results
    
    structure = parse_structure(structure_file)
    model = structure[0]
    
    if chain_id not in model:
        results = []
    else:
        chain = model[chain_id]
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        
        results = []
        for residue in residues:
            if atom_type:
                if atom_type in residue:
                    atom = residue[atom_type]
                    results.append(atom if return_atoms else atom.get_coord())
            else:
                for atom in residue.get_atoms():
                    results.append(atom if return_atoms else atom.get_coord())
    
    store_in_cache(structure_file, cache_key, results)
    return results