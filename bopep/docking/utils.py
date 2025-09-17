from bopep.structure.parser import parse_structure
import os
import glob


def extract_sequence_from_pdb(pdb_file: str, chain_id: str = "A"):
    """
    Extracts the sequence from a PDB or CIF file for a given chain.

    Parameters:
    - pdb_file: Path to the PDB or CIF file.
    - chain_id: The chain ID to extract the sequence from (default is 'A').

    Returns:
    - Extracted sequence as a string.
    """
    structure = parse_structure(pdb_file, structure_id="target")
    aa_dict = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        "SEC": "U",
        "PYL": "O",
    }
    # Use only the first model to avoid repeated sequences for NMR/multi-model structures
    model = structure[0]
    sequence = "".join(
        aa_dict.get(residue.get_resname(), "X")
        for chain in model
        if chain.id == chain_id
        for residue in chain
        if residue.id[0] == " "
    )
    return sequence