import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from bopep.structure.parser import get_chain_sequences, get_chain_coordinates
import os
import glob
from itertools import combinations


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
            if ref_chain_seqs[pep_id] == peptide_sequence:
                ref_pep_chain_id =  pep_id
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


def compute_intra_model_rmsd(processed_dir : str, peptide_sequence : str):
    """
    Compute intra-model RMSD metrics for peptide chains within and across docking methods.
    """
    results = {}
    
    # Find all model files by method
    alphafold_models = sorted(glob.glob(os.path.join(processed_dir, "alphafold_model_*.pdb")))
    boltz_models = sorted(glob.glob(os.path.join(processed_dir, "boltz_model_*.pdb")))
    
    # Function to compute pairwise RMSDs within a set of models
    def compute_pairwise_rmsd(model_files):
        if len(model_files) < 2:
            return None
        
        rmsd_values = []
        for ref_file, comp_file in combinations(model_files, 2):
            try:
                rmsd_val = align_and_compute_rmsd(ref_file, comp_file, peptide_sequence)
                rmsd_values.append(rmsd_val)
            except Exception as e:
                continue
        
        return np.mean(rmsd_values) if rmsd_values else None
    
    # Compute intra-method RMSDs
    if len(alphafold_models) >= 2:
        results['intra_alphafold_mean_rmsd'] = compute_pairwise_rmsd(alphafold_models)
    
    if len(boltz_models) >= 2:
        results['intra_boltz_mean_rmsd'] = compute_pairwise_rmsd(boltz_models)
    
    # Compute cross-method RMSD (all models together)
    all_models = alphafold_models + boltz_models
    if len(all_models) >= 2:
        results['intra_all_mean_rmsd'] = compute_pairwise_rmsd(all_models)
    
    return results


if __name__ == "__main__":
    structure_file_path = "./data/1ssc.pdb"

    seq_dict = get_chain_sequences(structure_file_path)

    print(seq_dict["B"], seq_dict["A"])
    
    distance_matrix = align_and_compute_rmsd(ref_structure_file=structure_file_path, structure_file=structure_file_path, peptide_sequence=seq_dict["B"])
    print(f"Distance matrix for {structure_file_path}:")
    print(distance_matrix)

    print(compute_intra_model_rmsd(processed_dir="/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV", peptide_sequence="NYLSELSEHV"))
