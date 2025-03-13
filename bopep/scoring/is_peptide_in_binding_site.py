from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import cKDTree
import os
import re


def is_peptide_in_binding_site_pdb_file(
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
    colab_dir: str, binding_site_residue_indices: list, threshold=5.0
) -> tuple:
    """
    Evaluates if the docked peptide is within a given proximity to the receptor binding site
    across multiple models. Only considers files with pattern 'rank_00X' in their name.
    """
    matches_within_threshold = 0

    colab_files = os.listdir(colab_dir)

    top_pdb_is_in_binding_site = False

    # Regex search for pdb files with rank_00X pattern in the directory
    pdb_files = [
        os.path.join(colab_dir, file)
        for file in colab_files
        if re.search(r"unrelaxed_rank_00\d+.*\.pdb$", file, re.IGNORECASE)
    ]

    for pdb_file in pdb_files:
        if is_peptide_in_binding_site_pdb_file(
            pdb_file, binding_site_residue_indices, threshold
        ):
            matches_within_threshold += 1
            if "rank_001" in pdb_file:
                top_pdb_is_in_binding_site = True

    return top_pdb_is_in_binding_site, matches_within_threshold / len(pdb_files)
