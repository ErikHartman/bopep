import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from bopep.structure.parser import get_chain_sequences, get_chain_coordinates
import os
import glob


def match_and_truncate(ref_seq :  str, ref_coords : list, target_seq : str, target_coords : list):
    if ref_seq in target_seq:
        i = target_seq.index(ref_seq)
        return ref_coords, target_coords[i:i+len(ref_seq)]
    elif target_seq in ref_seq:
        i = ref_seq.index(target_seq)
        return ref_coords[i:i+len(target_seq)], target_coords
    else:
        raise ValueError(f"Could not match reference and target receptor sequences for alignment. Reference sequence: {ref_seq}, Target sequence: {target_seq}")

def rmsd(coords1 : np.ndarray, coords2 : np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

def align_and_compute_rmsd(
    ref_structure_file : str, structure_file : str, peptide_sequence :str,
):
    """
    Given multiple structure files (PDB/CIF), align each structure's receptor (chain A by default)
    and compute a distance matrix of peptide RMSDs between all pairs.

    Each structure's receptor is aligned to the first structure's receptor, then
    peptide RMSD is computed between all pairs of aligned structures.
    """
    try:
        ref_chain_seqs = get_chain_sequences(ref_structure_file)
        new_chain_seqs = get_chain_sequences(structure_file)
        if len(ref_chain_seqs) != 2:
            raise ValueError("Reference structure file must have exactly two chains.")
        ref_keys = list(ref_chain_seqs.keys())

        # Assign peptide/receptor chains by sequence match
        
        pairs = [
            (ref_keys[0], ref_keys[1]),
            (ref_keys[1], ref_keys[0])
        ]
        for pep_id, rec_id in pairs:
            if ref_chain_seqs.get(pep_id, "") == peptide_sequence:
                ref_pep_chain_id = pep_id
                ref_rec_chain_id, ref_rec_seq = rec_id, ref_chain_seqs[rec_id]
                break
        else:
            raise ValueError("The peptide sequence found in reference chains does not match the new peptide sequence.")

        ref_rec_coords = np.array(get_chain_coordinates(ref_structure_file, ref_rec_chain_id))
        ref_pep_coords = np.array(get_chain_coordinates(ref_structure_file, ref_pep_chain_id))
        new_rec_coords = np.array(get_chain_coordinates(structure_file, "A"))
        new_pep_coords = np.array(get_chain_coordinates(structure_file, "B"))
        new_rec_seq = new_chain_seqs.get("A", "")

        ref_rec_coords_trunc, new_rec_coords_trunc = match_and_truncate(ref_rec_seq, ref_rec_coords, new_rec_seq, new_rec_coords)
        sup = SVDSuperimposer()
        sup.set(ref_rec_coords_trunc, new_rec_coords_trunc)
        sup.run()
        rot, tran = sup.get_rotran()
        new_pep_coords_aligned = np.dot(new_pep_coords, rot) + tran
        return rmsd(ref_pep_coords, new_pep_coords_aligned)
    
    except Exception as e:
        print(f"Warning: Template RMSD calculation failed: {e}")
        print(f"  Reference file: {ref_structure_file}")
        print(f"  Target file: {structure_file}")
        print(f"  Peptide sequence: {peptide_sequence}")
    return None


def compute_intra_model_rmsd(processed_dir: str, peptide_sequence: str):
    """
    Compute intra-model RMSD metrics for peptide chains within and across docking methods.
    """
    results = {}

    # Find all model files by method
    alphafold_models = sorted(glob.glob(os.path.join(processed_dir, "alphafold_model_*.pdb")))
    boltz_models = sorted(glob.glob(os.path.join(processed_dir, "boltz_model_*.pdb")))

    def _prep_ref_mapping(pdb_path: str, pep_seq: str):
        """Return (ref_pep_chain_id, ref_rec_chain_id) for a file, or None if not found."""
        seqs = get_chain_sequences(pdb_path)  # Now uses caching automatically
        keys = list(seqs.keys())
        if len(keys) != 2:
            return None
        # Try to find which chain matches the peptide
        if seqs.get(keys[0], "") == pep_seq:
            return keys[0], keys[1]
        if seqs.get(keys[1], "") == pep_seq:
            return keys[1], keys[0]
        # Fallback to common convention (B is peptide)
        if seqs.get("B", "") == pep_seq:
            return "B", "A"
        return None

    def _compute_pairwise_rmsd(model_files):
        if len(model_files) < 2:
            return None

        # Choose a reference model that we can map (peptide/receptor chain IDs)
        ref_file = None
        ref_mapping = None
        for f in model_files:
            m = _prep_ref_mapping(f, peptide_sequence)
            if m is not None:
                ref_file = f
                ref_mapping = m
                break
        if ref_file is None or ref_mapping is None:
            return None

        # Prepare reference receptor sequence and coordinates
        ref_pep_id, ref_rec_id = ref_mapping
        ref_seqs = get_chain_sequences(ref_file)  # Uses caching
        ref_rec_seq = ref_seqs.get(ref_rec_id, "")
        ref_rec_coords = np.array(get_chain_coordinates(ref_file, ref_rec_id))  # Uses caching

        # For each model, align its receptor (assumed A) to the reference receptor once,
        # then store its peptide coords transformed into the reference frame
        aligned_pep_coords = {}
        for f in model_files:
            try:
                comp_rec_coords = np.array(get_chain_coordinates(f, "A"))  # Uses caching
                comp_pep_coords = np.array(get_chain_coordinates(f, "B"))  # Uses caching
                comp_seqs = get_chain_sequences(f)  # Uses caching
                comp_rec_seq = comp_seqs.get("A", "")
                
                # Match-and-truncate receptor coords to ensure SVD sees same length/order
                ref_rec_coords_trunc, comp_rec_coords_trunc = match_and_truncate(
                    ref_rec_seq, ref_rec_coords, comp_rec_seq, comp_rec_coords
                )
                sup = SVDSuperimposer()
                sup.set(ref_rec_coords_trunc, comp_rec_coords_trunc)
                sup.run()
                rot, tran = sup.get_rotran()
                aligned_pep_coords[f] = np.dot(comp_pep_coords, rot) + tran
            except Exception:
                # Skip files that fail alignment
                continue

        # Need at least two successfully aligned models
        if len(aligned_pep_coords) < 2:
            return None

        # Compute pairwise RMSDs among pre-aligned peptide coordinates
        files = list(aligned_pep_coords.keys())
        rmsd_values = []
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                try:
                    rmsd_values.append(rmsd(aligned_pep_coords[files[i]], aligned_pep_coords[files[j]]))
                except Exception:
                    continue

        return float(np.mean(rmsd_values)) if rmsd_values else None

    # Compute intra-method RMSDs 
    if len(alphafold_models) >= 2:
        results['intra_alphafold_mean_rmsd'] = _compute_pairwise_rmsd(alphafold_models)

    if len(boltz_models) >= 2:
        results['intra_boltz_mean_rmsd'] = _compute_pairwise_rmsd(boltz_models)

    # Compute cross-method RMSD (all models together)
    all_models = alphafold_models + boltz_models
    if len(all_models) >= 2:
        results['intra_all_mean_rmsd'] = _compute_pairwise_rmsd(all_models)

    return results


if __name__ == "__main__":
    structure_file_path = "./data/1ssc.pdb"

    seq_dict = get_chain_sequences(structure_file_path)

    print(seq_dict["B"], seq_dict["A"])
    
    distance_matrix = align_and_compute_rmsd(ref_structure_file=structure_file_path, structure_file=structure_file_path, peptide_sequence=seq_dict["B"])
    print(f"Distance matrix for {structure_file_path}:")
    print(distance_matrix)