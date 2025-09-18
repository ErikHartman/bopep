"""
Structure parsing utilities for bopep.

This module provides unified interfaces for parsing structural data
from various file formats (PDB, CIF) commonly used in structural biology.
"""

from .parser import (
    StructureParser, 
    parse_structure,
    extract_sequence_from_structure,
    get_chain_sequences,
    check_starting_index_in_structure,
    get_structure_residues,
    get_chain_coordinates,
    get_chain_list,
    get_all_atom_coordinates,
    get_residue_coordinates,
    get_all_chain_atoms,
)

from .cache import (
    clear_structure_cache,
    get_cache_stats
)

__all__ = [
    'StructureParser',
    'parse_structure',
    'extract_sequence_from_structure', 
    'get_chain_sequences',
    'check_starting_index_in_structure',
    'get_structure_residues',
    'get_chain_coordinates',
    'get_chain_list',
    'get_all_atom_coordinates',
    'get_residue_coordinates',
    'get_all_chain_atoms',
    'clear_structure_cache',
    'get_cache_stats'
]