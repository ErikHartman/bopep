from Bio.PDB import PDBParser, MMCIFParser
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
    # Choose parser based on file extension
    if pdb_file.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
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
    docking_dir: str, target_structure_copy: str, peptide_name: str
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
        for file in os.listdir(docking_dir):
            if (
                file.startswith("pdb70")
                or file.endswith(".cif")
                or file == "cite.bibtex"
                or file.startswith("combined_input")
                or file.endswith(".png")
                or file.endswith(".jpg")
            ):
                os.remove(os.path.join(docking_dir, file))
    except OSError as e:
        print(f"Error deleting temporary files for {peptide_name}: {e}")

def docking_folder_exists(base_docking_dir : str, peptide : str, target_structure : str) -> bool:
    """
    Checks if a docking result exists for a given target+peptide,
    and also checks if the folder contains a relaxed PDB result.

    Parameters:
    - base_docking_dir: Base directory for docking results
    - peptide: Peptide sequence
    - target_structure: Path to target structure file

    Returns:
    - (exists, peptide_dir): Tuple with boolean if folder exists and has finished, and the path to that directory
    """
    target_name = os.path.basename(target_structure).replace(".pdb", "")
    peptide_dir = os.path.join(base_docking_dir, f"{target_name}_{peptide}")
    exists = os.path.exists(peptide_dir) and os.path.isdir(peptide_dir)
    
    # Check either for finished.txt or target_peptide.done.txt
    contains_done_txt = os.path.exists(os.path.join(peptide_dir, f"{target_name}_{peptide}.done.txt"))
    contains_finished_txt = os.path.exists(os.path.join(peptide_dir, "finished.txt"))
    # find _relaxed_ using regex
    contains_relaxed_structure = bool(
        glob.glob(os.path.join(peptide_dir, "*_relaxed_*.pdb"))
    )
    
    if exists and (contains_done_txt or contains_finished_txt) and contains_relaxed_structure:
        print(f"Docking result for {peptide} already exists in {peptide_dir}. Skipping...")
        return True, peptide_dir
    else:
        return False, peptide_dir
