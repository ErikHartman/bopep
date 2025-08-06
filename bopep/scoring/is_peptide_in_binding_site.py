from typing import Tuple
from Bio.PDB import PDBParser
import numpy as np
from scipy.spatial import cKDTree
import os
import re
import math


def centroid(coords: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean of an (N×3) array of points."""
    return coords.mean(axis=0)

def is_peptide_near_binding_site_by_centroid(
    pdb_file: str,
    binding_site_residue_indices: list[int],
    receptor_chain: str = "A",
    peptide_chain: str   = "B",
    cutoff: float        = 10.0,
) -> Tuple[float, bool]:
    """
    Compute centroids of the binding-site atoms and the peptide atoms,
    and return True if their separation is <= cutoff (Å).

    binding_site_residue_indices: zero-based positions in the chain
      (0→first residue, 1→second, …)
    """
    parser = PDBParser(QUIET=True)
    model  = parser.get_structure("docked", pdb_file)[0]

    R = model[receptor_chain]
    P = model[peptide_chain]

    # find what absolute PDB number the first residue has
    all_ids = sorted(r.id[1] for r in R if r.id[0] == " ")
    min_id  = all_ids[0]

    # debug: what residues are these, in actual PDB numbering?
    abs_ids = [rel + min_id for rel in binding_site_residue_indices]
    names   = []
    for idx in abs_ids:
        key = (" ", idx, " ")
        names.append(R[key].get_resname() if key in R else None)

    # collect binding-site atom coords
    bs_coords = []
    for rel in binding_site_residue_indices:
        pdb_idx = rel + min_id
        key     = (" ", pdb_idx, " ")
        if key in R:
            bs_coords.extend(atom.coord for atom in R[key].get_atoms())

    if not bs_coords:
        raise ValueError(f"No binding-site atoms found for indices {binding_site_residue_indices}")

    bs_coords  = np.vstack(bs_coords)
    pep_coords = np.vstack([a.coord for a in P.get_atoms()])

    bs_cent  = centroid(bs_coords)
    pep_cent = centroid(pep_coords)
    dist     = np.linalg.norm(bs_cent - pep_cent)

    return dist, dist <= cutoff

def get_binding_site(
    pdb_file: str,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    threshold: float = 5.0,
) -> Tuple[list, list, list, list]:
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
        for i, atom in enumerate(receptor_atoms): # This will be 1-indexed
            indices = peptide_tree.query_ball_point(atom.coord, threshold)
            if indices:
                receptor_binding_site_residue_indices.add(
                    atom.get_parent().id[1]
                ) # E4
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
            sorted(list(receptor_binding_site_residue_indices)),
            sorted(list(peptide_binding_site_residue_indices)),
            peptide_atoms,
        )

    except KeyError as e:
        print(f"Chain not found in structure: {e}")
        return [], [], [], []


def is_peptide_in_binding_site_pdb_file(
    pdb_file: str, binding_site_residue_indices: list = None, threshold: float = 5.0, required_n_contact_residues: int = 2
) -> Tuple[int, bool]:
    """
    Determines if the peptide in the given PDB content is within the threshold distance
    to the receptor's binding site.
    """
    _, receptor_binding_site_indices, _, _ = get_binding_site(
        pdb_file, threshold=threshold
    )

    if binding_site_residue_indices is not None:
        # if any receptor_binding_site_indices is in the binding_site_residue_indices, return True
        nr_contact_residues = 0
        for receptor_index in receptor_binding_site_indices:
            if receptor_index in binding_site_residue_indices:
                nr_contact_residues += 1

    if nr_contact_residues >= required_n_contact_residues:
        return nr_contact_residues, True

    else:
        return nr_contact_residues, False


def n_peptides_in_binding_site_processed_dir(
    processed_dir: str, binding_site_residue_indices: list, threshold=5.0, required_n_contact_residues: int = 2
) -> Tuple[float, bool, int]:
    """
    Evaluates if the docked peptide is within a given proximity to the receptor binding site
    across multiple models in a processed directory. Works with standardized model naming.
    """
    import glob
    
    matches_within_threshold = 0
    top_model_is_in_binding_site = False
    n_contacts = 0

    # Find all model files in the processed directory
    pdb_patterns = ["*_model_*.pdb", "*_model_*.cif", "model_*.pdb", "model_*.cif"]
    pdb_files = []
    
    for pattern in pdb_patterns:
        files = glob.glob(os.path.join(processed_dir, pattern))
        pdb_files.extend(files)
    
    # Remove duplicates and sort
    pdb_files = sorted(list(set(pdb_files)))
    
    if not pdb_files:
        print(f"WARNING: No model files found in {processed_dir}")
        return 0.0, False, 0

    for pdb_file in pdb_files:
        n_contacts_temp, is_in_binding_site = is_peptide_in_binding_site_pdb_file(
            pdb_file, binding_site_residue_indices, threshold=threshold, 
            required_n_contact_residues=required_n_contact_residues
        )
        if is_in_binding_site:
            matches_within_threshold += 1
            # Check if this is the top model (model_1 or method_model_1)
            filename = os.path.basename(pdb_file)
            if "model_1" in filename:
                top_model_is_in_binding_site = True
                n_contacts = n_contacts_temp

    return matches_within_threshold / len(pdb_files), top_model_is_in_binding_site, n_contacts



def smooth_peptide_binding_site_score(
    pdb_file: str,
    binding_site_residue_indices: list[int],
    threshold: float = 10.0,
    alpha: float = 0.5,
) -> float:
    """
    binding_site_residue_indices: zero-based positions in the receptor chain
    (0→first residue, 1→second, …)
    """
    # get the peptide atoms and identify binding site by contact
    binding_site_threshold = max(threshold * 2, 10.0)
    _, _, _, peptide_atoms = get_binding_site(
        pdb_file, threshold=binding_site_threshold
    )

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("docked", pdb_file)[0]
    receptor_chain = model["A"]

    # figure out absolute numbering for the chain
    min_receptor_residue_id = min(
        res.id[1] for res in receptor_chain.get_residues() if res.id[0] == " "
    )
    # convert zero-based → absolute
    abs_indices = [rel + min_receptor_residue_id
                   for rel in binding_site_residue_indices]

    # build KD-tree of peptide atoms
    peptide_coords = np.array([atom.coord for atom in peptide_atoms])
    tree = cKDTree(peptide_coords)

    # compute per-residue score
    residue_scores = []
    for pdb_idx in abs_indices:
        key = (" ", pdb_idx, " ")
        residue = receptor_chain[key]
        coords = np.array([atom.coord for atom in residue.get_atoms()])
        if coords.size == 0:
            continue

        dists, _ = tree.query(coords)
        min_dist = dists.min()
        residue_scores.append(1.0 / (1.0 + math.exp(alpha * (min_dist - threshold))))

    if not residue_scores:
        return 0.0

    return float(np.mean(residue_scores))



if __name__ == "__main__":
    docked_pdb = "/home/er8813ha/bopep/examples/docking/processed/4glf_NYLSELSEHV/alphafold_model_1.pdb"
    
    bsri = [22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 69, 70, 71, 72,
                73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                88, 89, 90, 104, 105, 106, 107, 108, 109, 110] 
    
    bsri = [residue - 19 for residue in bsri]  # convert to zero-based indices
    
    print("Testing with single PDB file:")
    print(is_peptide_in_binding_site_pdb_file(docked_pdb, binding_site_residue_indices=bsri, threshold=5.0, required_n_contact_residues=5))

    centroid_in_binding_site = is_peptide_near_binding_site_by_centroid(
        docked_pdb,
        binding_site_residue_indices=bsri,
        receptor_chain="A",
        peptide_chain="B",
        cutoff=20.0,
    )
    print(f"Centroid is in binding site: {centroid_in_binding_site}")
    
    # Test with processed directory
    processed_dir = "/home/er8813ha/bopep/examples/docking/processed/4glf_NYLSELSEHV"
    if os.path.exists(processed_dir):
        print("\nTesting with processed directory:")
        result = n_peptides_in_binding_site_processed_dir(
            processed_dir, binding_site_residue_indices=bsri, threshold=5.0, required_n_contact_residues=5
        )
        print(f"Processed directory result: {result}")