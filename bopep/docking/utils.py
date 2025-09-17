from bopep.structure.parser import parse_structure
import os
import glob


def extract_sequence_from_structure(structure_file: str, chain_id: str = "A"):
    """
    Extracts the sequence from a structure file (PDB or CIF format) for a given chain.
    """
    structure = parse_structure(structure_file, structure_id="target")
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
        aa_dict.get(res.get_resname(), "X")
        for chain in model
        if chain.id == chain_id
        for res in chain
        if res.get_id()[0] == " "  # Standard amino acids only
    )
    return sequence


def get_pdb_files_in_dir(directory_path: str) -> list:
    """
    Get all PDB files in a directory.

    Parameters:
    - directory_path: Path to directory to search.

    Returns:
    - List of PDB file paths.
    """
    return glob.glob(os.path.join(directory_path, "*.pdb"))