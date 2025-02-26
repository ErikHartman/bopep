from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import cKDTree
import os


def is_peptide_in_binding_site(
    pdb_file: str, binding_site_residue_indices: list, threshold: float = 5.0
) -> bool:
    """
    Determines if the peptide in the given PDB content is within the threshold distance
    to the receptor's binding site.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("docked", pdb_file)
    except Exception as e:
        print(f"Error parsing PDB content: {e}")
        return False

    try:
        model = structure[0]
        receptor_chain = model["A"]
        peptide_chain = model["B"]

        receptor_binding_site_atoms = [
            atom
            for residue in receptor_chain
            if residue.id[1] in binding_site_residue_indices
            for atom in residue.get_atoms()
        ]

        if not receptor_binding_site_atoms:
            print(f"No atoms found in specified receptor binding site residues.")
            return False

        peptide_atoms = list(peptide_chain.get_atoms())
        if not peptide_atoms:
            print(f"No atoms found in peptide chain.")
            return False

        # Find atoms within distance threshold
        receptor_coords = np.array([atom.coord for atom in receptor_binding_site_atoms])
        peptide_coords = np.array([atom.coord for atom in peptide_atoms])

        tree = cKDTree(peptide_coords)
        distances, _ = tree.query(receptor_coords, distance_upper_bound=threshold)
        return np.any(distances != float("inf"))

    except KeyError:
        print(f"Chains A or B not found in structure.")
        return False

def n_peptides_in_binding_site_colab_dir(
    colab_dir : str, binding_site_residue_indices : list, threshold=5.0
):
    """
    Evaluates if the docked peptide is within a given proximity to the receptor binding site
    across multiple models.
    """
    matches_within_threshold = 0

    colab_files = os.listdir(colab_dir)
    pdb_files = [f for f in colab_files if f.endswith(".pdb")] # is this enough?

    for pdb_file in pdb_files:
        if is_peptide_in_binding_site(pdb_file, binding_site_residue_indices, threshold):
            matches_within_threshold += 1

    return matches_within_threshold 
