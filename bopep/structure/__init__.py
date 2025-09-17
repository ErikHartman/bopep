"""
Structure parsing utilities for bopep.

This module provides unified interfaces for parsing structural data
from various file formats (PDB, CIF) commonly used in structural biology.
"""

from .parser import StructureParser, parse_structure

__all__ = [
    'StructureParser',
    'parse_structure'
]