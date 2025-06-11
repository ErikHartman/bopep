from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import cKDTree
import os
import re
import math


def get_binding_site(
    pdb_file: str,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    threshold: float = 5.0,
) -> tuple:
    """
    Identifies interacting residues in the binding site for both receptor and peptide chains.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file
    receptor_chain : str, optional
        Chain ID for the receptor (default "A")
    peptide_chain : str, optional
        Chain ID for the peptide (default "B")
    threshold : float, optional
        Distance threshold (in Å) to consider residues as interacting (default 5.0)

    Returns
    -------
    tuple
        (receptor_binding_site_atoms, receptor_binding_site_residue_indices,
         peptide_binding_site_residue_indices, peptide_atoms)
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("docked", pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return [], [], [], []

    try:
        model = structure[0]

        receptor_chain = model[receptor_chain]
        peptide_chain = model[peptide_chain]

        min_receptor_residue_id = min(
            residue.id[1]
            for residue in receptor_chain.get_residues()
            if residue.id[0] == " "
        )
        min_peptide_residue_id = min(
            residue.id[1]
            for residue in peptide_chain.get_residues()
            if residue.id[0] == " "
        )

        # Get all atoms from both chains
        receptor_atoms = list(receptor_chain.get_atoms())
        peptide_atoms = list(peptide_chain.get_atoms())

        if not receptor_atoms or not peptide_atoms:
            return [], [], [], []

        # Build KD-trees for efficient distance calculations
        peptide_coords = np.array([atom.coord for atom in peptide_atoms])
        peptide_tree = cKDTree(peptide_coords)

        # Find interacting residues
        receptor_binding_site_residue_indices = set()
        peptide_binding_site_residue_indices = set()

        # For each receptor atom, find peptide atoms within threshold
        for i, atom in enumerate(receptor_atoms):
            indices = peptide_tree.query_ball_point(atom.coord, threshold)
            if indices:
                receptor_binding_site_residue_indices.add(
                    atom.get_parent().id[1] - min_receptor_residue_id
                )
                for idx in indices:
                    peptide_binding_site_residue_indices.add(
                        peptide_atoms[idx].get_parent().id[1] - min_peptide_residue_id
                    )

        # Get the atoms from the binding site residues
        receptor_binding_site_atoms = [
            atom
            for residue in receptor_chain
            if residue.id[1] in receptor_binding_site_residue_indices
            for atom in residue.get_atoms()
        ]

        return (
            receptor_binding_site_atoms,
            list(receptor_binding_site_residue_indices),
            list(peptide_binding_site_residue_indices),
            peptide_atoms,
        )

    except KeyError as e:
        print(f"Chain not found in structure: {e}")
        return [], [], [], []


def is_peptide_in_binding_site_pdb_file(
    pdb_file: str, binding_site_residue_indices: list = None, threshold: float = 5.0
) -> bool:
    """
    Determines if the peptide in the given PDB content is within the threshold distance
    to the receptor's binding site.
    """
    _, receptor_binding_site_indices, _, _ = get_binding_site(
        pdb_file, threshold=threshold
    )

    if binding_site_residue_indices is not None:
        # if any receptor_binding_site_indices is in the binding_site_residue_indices, return True
        for receptor_index in receptor_binding_site_indices:
            if receptor_index in binding_site_residue_indices:
                return True

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
            pdb_file, binding_site_residue_indices, threshold=threshold
        ):
            matches_within_threshold += 1
            if "rank_001" in pdb_file:
                top_pdb_is_in_binding_site = True

    return top_pdb_is_in_binding_site, matches_within_threshold / len(pdb_files)


def smooth_peptide_binding_site_score(
    pdb_file: str,
    binding_site_residue_indices: list,
    threshold: float = 10.0,
    alpha: float = 0.5,
) -> float:
    """
    Computes a continuous 'in-pocket' score for a peptide in chain B relative to
    a receptor in chain A, focusing on interaction distances.

    The score ranges roughly from 0 (completely out of pocket)
    to 1 (fully in pocket), computed by averaging a smooth distance-based
    logistic function across binding site residues.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to parse.
    binding_site_residue_indices : list, optional
        List of residue indices in the receptor chain that are considered part of the binding site.
    threshold : float, optional
        Distance (Å) around which the logistic penalty transitions (default 10.0).
        Lower distances yield higher "in-pocket" contribution.
    alpha : float, optional
        Steepness of the logistic function (default 0.5).

    Returns
    -------
    float
        A score between 0 and 1 indicating how well the peptide is positioned
        in the binding site. Higher is better (more in-pocket).
    """
    # Get the binding site - use a larger threshold for determining binding site
    binding_site_threshold = max(
        threshold * 2, 10.0
    )  # Use a larger threshold to identify potential binding site
    _, _, _, peptide_atoms = get_binding_site(
        pdb_file, threshold=binding_site_threshold
    )

    # Now calculate the score using the identified binding site
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("docked", pdb_file)
        model = structure[0]
        receptor_chain = model["A"]

        peptide_coords = np.array([atom.coord for atom in peptide_atoms])
        tree = cKDTree(peptide_coords)

        # For each binding-site residue, find min distance to any peptide atom
        residue_scores = []
        for res_idx in binding_site_residue_indices:
            residue = receptor_chain[(" ", res_idx, " ")]
            residue_atom_coords = np.array([atom.coord for atom in residue.get_atoms()])
            if residue_atom_coords.size == 0:
                continue

            # Query the KD-tree for distances from this residue's atoms
            distances, _ = tree.query(residue_atom_coords)
            min_dist = np.min(distances)

            # Apply logistic function: 1 / (1 + exp(alpha * (d - threshold)))
            residue_score = 1.0 / (1.0 + math.exp(alpha * (min_dist - threshold)))
            residue_scores.append(residue_score)

        if not residue_scores:
            return 0.0

        in_pocket_score = float(np.mean(residue_scores))
        return in_pocket_score

    except (KeyError, Exception) as e:
        print(f"Error calculating binding site score: {e}")
        return 0.0

if __name__ == "__main__":
    original_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/0_complexes/pdbs/1d6w.pdb"
    docked_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/2_docked/pdbs/1d6w_DFEEIPEEL/1d6w_DFEEIPEEL_relaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb"

    receptor_binding_site_atoms, receptor_binding_site_residue_indices, peptide_binding_site_residue_indices, peptide_atoms = get_binding_site(original_pdb)
    print(receptor_binding_site_residue_indices)
    receptor_binding_site_atoms, receptor_binding_site_residue_indices, peptide_binding_site_residue_indices, peptide_atoms = get_binding_site(docked_pdb)
    print(receptor_binding_site_residue_indices)