
from Bio.PDB import PDBParser

def extract_sequence_from_pdb(pdb_file, chain_id="A"):
    """
    Extracts the sequence from a PDB file for a given chain.

    Parameters:
    - pdb_file: Path to the PDB file.
    - chain_id: The chain ID to extract the sequence from (default is 'A').

    Returns:
    - Extracted sequence as a string.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_file)
    aa_dict = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
        "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
        "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
        "TRP": "W", "TYR": "Y", "SEC": "U", "PYL": "O"
    }
    sequence = "".join(
        aa_dict.get(residue.get_resname(), "X")
        for model in structure
        for chain in model
        if chain.id == chain_id
        for residue in chain
        if residue.id[0] == " "
    )
    return sequence