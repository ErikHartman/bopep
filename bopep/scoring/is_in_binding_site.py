from typing import Tuple
from bopep.structure.parser import parse_structure
import numpy as np
from scipy.spatial import cKDTree
import os
import math


def centroid(coords: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean of an (N×3) array of points."""
    return coords.mean(axis=0)

def is_sequence_near_binding_site_by_centroid(
    structure_file: str,
    binding_site_residue_indices: list[int],
    receptor_chain: str = "A",
    sequence_chain: str   = "B",
    cutoff: float        = 10.0,
) -> Tuple[float, bool]:
    """
    Compute centroids of the binding-site atoms and the sequence atoms,
    and return True if their separation is <= cutoff (Å).

    binding_site_residue_indices: zero-based positions in the chain
      (0→first residue, 1→second, …)
    """
    structure = parse_structure(structure_file, structure_id="docked")
    model = structure[0]

    R = model[receptor_chain]
    P = model[sequence_chain]

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
    seq_coords = np.vstack([a.coord for a in P.get_atoms()])

    bs_cent  = centroid(bs_coords)
    seq_cent = centroid(seq_coords)
    dist     = np.linalg.norm(bs_cent - seq_cent)

    return dist, dist <= cutoff

def get_binding_site(
    structure_file: str,
    receptor_chain: str = "A",
    sequence_chain: str = "B",
    threshold: float = 5.0,
) -> Tuple[list, list, list, list]:
    """
    Identifies interacting residues in the binding site for both receptor and sequence chains.
    """
    try:
        structure = parse_structure(structure_file, structure_id="docked", auth_residues=False)
    except Exception as e:
        print(f"Error parsing structure file: {e}")
        return [], [], [], []

    try:
        model = structure[0]

        receptor_chain_obj = model[receptor_chain]
        sequence_chain_obj = model[sequence_chain]
        
        # Filter to only standard amino acid residues immediately after parsing
        receptor_residues = [res for res in receptor_chain_obj.get_residues() if res.id[0] == " "]
        sequence_residues = [res for res in sequence_chain_obj.get_residues() if res.id[0] == " "]
        
        if not receptor_residues or not sequence_residues:
            return [], [], [], []

        receptor_residue_map = {i: res for i, res in enumerate(receptor_residues)}
        sequence_residue_map = {i: res for i, res in enumerate(sequence_residues)}
        
        # Create reverse mapping from PDB residue ID to zero-based index
        receptor_pdb_to_zero = {res.id[1]: i for i, res in enumerate(receptor_residues)}
        sequence_pdb_to_zero = {res.id[1]: i for i, res in enumerate(sequence_residues)}

        # Get all atoms from filtered residues
        receptor_atoms = [atom for res in receptor_residues for atom in res.get_atoms()]
        sequence_atoms = [atom for res in sequence_residues for atom in res.get_atoms()]

        if not receptor_atoms or not sequence_atoms:
            return [], [], [], []

        # Build KD-trees for efficient distance calculations
        sequence_coords = np.array([atom.coord for atom in sequence_atoms])
        sequence_tree = cKDTree(sequence_coords)

        # Find interacting residues (using zero-based indices)
        receptor_binding_site_residue_indices = set()
        sequence_binding_site_residue_indices = set()

        # For each receptor atom, find sequence atoms within threshold
        for atom in receptor_atoms:
            indices = sequence_tree.query_ball_point(atom.coord, threshold)
            if indices:
                # Get zero-based index for receptor residue
                receptor_pdb_id = atom.get_parent().id[1]
                if receptor_pdb_id in receptor_pdb_to_zero:
                    receptor_binding_site_residue_indices.add(receptor_pdb_to_zero[receptor_pdb_id])
                
                # Get zero-based indices for sequence residues
                for idx in indices:
                    sequence_pdb_id = sequence_atoms[idx].get_parent().id[1]
                    if sequence_pdb_id in sequence_pdb_to_zero:
                        sequence_binding_site_residue_indices.add(sequence_pdb_to_zero[sequence_pdb_id])

        # Get the atoms from the binding site residues (using zero-based indices)
        receptor_binding_site_atoms = [
            atom
            for zero_idx in receptor_binding_site_residue_indices
            for atom in receptor_residue_map[zero_idx].get_atoms()
        ]            


        return (
            receptor_binding_site_atoms,
            sorted(list(receptor_binding_site_residue_indices)),
            sorted(list(sequence_binding_site_residue_indices)),
            sequence_atoms,
        )

    except KeyError as e:
        print(f"Chain not found in structure: {e}")
        return [], [], [], []

def get_receptor_contacts(structure_file: str,
    receptor_chain: str = "A",
    sequence_chain: str = "B",
    threshold: float = 5.0,):
    return get_binding_site(structure_file, receptor_chain, sequence_chain, threshold)[1]

def is_sequence_in_binding_site_pdb_file(
    structure_file: str, binding_site_residue_indices: list = None, threshold: float = 5.0, required_n_contact_residues: int = 2, receptor_chain: str = "A", sequence_chain: str = "B"
) -> Tuple[int, bool]:
    """
    Determines if the sequence in the given structure file is within the threshold distance
    to the receptor's binding site.
    """
    _, receptor_binding_site_indices, _, _ = get_binding_site(
        structure_file, receptor_chain, sequence_chain, threshold
    )

    nr_contact_residues = 0
    if binding_site_residue_indices is not None:
        # if any receptor_binding_site_indices is in the binding_site_residue_indices, return True
        for receptor_index in receptor_binding_site_indices:
            if receptor_index in binding_site_residue_indices:
                nr_contact_residues += 1

    if nr_contact_residues >= required_n_contact_residues:
        return nr_contact_residues, True
    else:
        return nr_contact_residues, False


def n_sequences_in_binding_site_processed_dir(
    processed_dir: str, binding_site_residue_indices: list, threshold=5.0, required_n_contact_residues: int = 2
) -> Tuple[float, bool, int]:
    """
    Evaluates if the docked sequence is within a given proximity to the receptor binding site
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
        n_contacts_temp, is_in_binding_site = is_sequence_in_binding_site_pdb_file(
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



def smooth_sequence_binding_site_score(
    structure_file: str,
    binding_site_residue_indices: list[int],
    threshold: float = 10.0,
    alpha: float = 0.5,
    receptor_chain: str = "A",
    sequence_chain: str = "B",
) -> float:
    """
    binding_site_residue_indices: zero-based positions in the receptor chain
    (0→first residue, 1→second, …)
    """
    # get the sequence atoms and identify binding site by contact
    binding_site_threshold = max(threshold * 2, 10.0)
    _, _, _, sequence_atoms = get_binding_site(
        structure_file, receptor_chain, sequence_chain, threshold=binding_site_threshold
    )

    structure = parse_structure(structure_file, structure_id="docked")
    model = structure[0]
    receptor_chain_obj = model[receptor_chain]

    # figure out absolute numbering for the chain
    min_receptor_residue_id = min(
        res.id[1] for res in receptor_chain_obj.get_residues() if res.id[0] == " "
    )
    # convert zero-based → absolute
    abs_indices = [rel + min_receptor_residue_id
                   for rel in binding_site_residue_indices]

    # build KD-tree of sequence atoms
    sequence_coords = np.array([atom.coord for atom in sequence_atoms])
    tree = cKDTree(sequence_coords)

    # compute per-residue score
    residue_scores = []
    for pdb_idx in abs_indices:
        key = (" ", pdb_idx, " ")
        residue = receptor_chain_obj[key]
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
    docked_pdb = "/srv/data1/er8813ha/bopep/docked/cd14_processed/processed/4glf_NENARQQLERQNK/boltz_model_1.pdb"

    docked_test_ = "/srv/data1/er8813ha/bopep/docked/cd14_processed/processed/4glf_NENARQQLERQNK/boltz_model_1.pdb"
    
    bsri = [22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 69, 70, 71, 72,
                73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                88, 89, 90, 104, 105, 106, 107, 108, 109, 110] 
    
    bsri = [residue - 19 for residue in bsri]  # convert to zero-based indices
    
    print("Testing with single PDB file:")
    print(is_sequence_in_binding_site_pdb_file(docked_pdb, binding_site_residue_indices=bsri, threshold=5.0, required_n_contact_residues=5))

    centroid_in_binding_site = is_sequence_near_binding_site_by_centroid(
        docked_pdb,
        binding_site_residue_indices=bsri,
        receptor_chain="A",
        sequence_chain="B",
        cutoff=20.0,
    )
    print(f"Centroid is in binding site: {centroid_in_binding_site}")
    
    # Test with processed directory
    processed_dir = "/home/er8813ha/bopep/examples/docking/processed/4glf_NYLSELSEHV"
    if os.path.exists(processed_dir):
        print("\nTesting with processed directory:")
        result = n_sequences_in_binding_site_processed_dir(
            processed_dir, binding_site_residue_indices=bsri, threshold=5.0, required_n_contact_residues=5
        )
        print(f"Processed directory result: {result}")


    print(get_receptor_contacts(docked_test_, "A","B", 5.0))