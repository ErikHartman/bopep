import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from bopep.scoring.utils import parse_pdb, get_chain_sequences, match_and_truncate

def rmsd(coords1, coords2):
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

def align_and_compute_rmsd(
    ref_pdb_file, pdb_file, peptide_sequence,
):
    """
    Given multiple PDBs, align each structure's receptor (chain A by default)
    and compute a distance matrix of peptide RMSDs between all pairs.

    Each structure's receptor is aligned to the first PDB's receptor, then
    peptide RMSD is computed between all pairs of aligned structures.
    """
 
    ref_chain_seqs = get_chain_sequences(ref_pdb_file)
    new_chain_seqs = get_chain_sequences(pdb_file)
    if len(ref_chain_seqs) != 2:
        raise ValueError("Reference PDB must have exactly two chains.")
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

    ref_rec_coords, ref_pep_coords = map(np.array, parse_pdb(ref_pdb_file, ref_rec_chain_id, ref_pep_chain_id))
    new_rec_coords, new_pep_coords = map(np.array, parse_pdb(pdb_file, "A", "B"))
    new_rec_seq = new_chain_seqs.get("A", "")

    ref_rec_coords_trunc, new_rec_coords_trunc = match_and_truncate(ref_rec_seq, ref_rec_coords, new_rec_seq, new_rec_coords)
    sup = SVDSuperimposer()
    sup.set(ref_rec_coords_trunc, new_rec_coords_trunc)
    sup.run()
    rot, tran = sup.get_rotran()
    new_pep_coords_aligned = np.dot(new_pep_coords, rot) + tran
    return rmsd(ref_pep_coords, new_pep_coords_aligned)


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"

    seq_dict = get_chain_sequences(pdb_file_path)

    print(seq_dict["B"], seq_dict["A"])
    
    distance_matrix = align_and_compute_rmsd(ref_pdb_file=pdb_file_path, pdb_file=pdb_file_path, peptide_sequence=seq_dict["B"])
    print(f"Distance matrix for {pdb_file_path}:")
    print(distance_matrix)
