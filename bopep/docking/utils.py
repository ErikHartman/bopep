from Bio.PDB import PDBParser
import os


def extract_sequence_from_pdb(pdb_file: str, chain_id: str = "A"):
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
    sequence = "".join(
        aa_dict.get(residue.get_resname(), "X")
        for model in structure
        for chain in model
        if chain.id == chain_id
        for residue in chain
        if residue.id[0] == " "
    )
    return sequence


def clean_up_files(
    peptide_output_dir: str, target_structure_copy: str, peptide_name: str
):
    """
    Cleans up temporary files created during docking.

    Parameters:
    - peptide_output_dir: Directory containing docking results.
    - target_structure_copy: Path to the copied target structure file.
    - peptide_name: Name of the peptide for logging purposes.
    """
    try:
        # Remove copied PDB file
        if os.path.exists(target_structure_copy):
            os.remove(target_structure_copy)

        # Remove unnecessary files generated during docking
        for file in os.listdir(peptide_output_dir):
            if (
                file.startswith("pdb70")
                or file.endswith(".cif")
                or file == "cite.bibtex"
                or file.startswith("combined_input")
            ):
                os.remove(os.path.join(peptide_output_dir, file))
    except OSError as e:
        print(f"Error deleting temporary files for {peptide_name}: {e}")
