import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1
from bopep.scoring.utils import parse_pdb

def rmsd(coords1, coords2):
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

def align_and_compute_rmsd(
    ref_pdb_file, pdb_file
):
    """
    Given multiple PDBs, align each structure's receptor (chain A by default)
    and compute a distance matrix of peptide RMSDs between all pairs.

    Each structure's receptor is aligned to the first PDB's receptor, then
    peptide RMSD is computed between all pairs of aligned structures.
    """

    def get_chain_sequences(pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_file)
        return {chain.id: ''.join([
            protein_letters_3to1.get(residue.get_resname().capitalize(), 'X')
            for residue in chain if residue.id[0] == ' '
        ]) for model in structure for chain in model}

    ref_chain_seqs = get_chain_sequences(ref_pdb_file)
    if len(ref_chain_seqs) != 2:
        raise ValueError("Reference PDB must have exactly two chains.")
    new_chain_seqs = get_chain_sequences(pdb_file)
    new_pep_seq = new_chain_seqs.get("B", "")
    ref_keys = list(ref_chain_seqs.keys())

    # Here we select the reference peptide based on which chain matches the new peptide.

    if new_pep_seq and ref_chain_seqs[ref_keys[0]] == new_pep_seq:
        ref_pep_chain_id, ref_pep_seq = ref_keys[0], ref_chain_seqs[ref_keys[0]]
        ref_rec_chain_id, ref_rec_seq = ref_keys[1], ref_chain_seqs[ref_keys[1]]
    elif new_pep_seq and ref_chain_seqs[ref_keys[1]] == new_pep_seq:
        ref_pep_chain_id, ref_pep_seq = ref_keys[1], ref_chain_seqs[ref_keys[1]]
        ref_rec_chain_id, ref_rec_seq = ref_keys[0], ref_chain_seqs[ref_keys[0]]
    else:
        sorted_chains = sorted(ref_chain_seqs.items(), key=lambda x: len(x[1]))
        ref_pep_chain_id, ref_pep_seq = sorted_chains[0]
        ref_rec_chain_id, ref_rec_seq = sorted_chains[1]


    ref_rec_coords, ref_pep_coords = parse_pdb(ref_pdb_file, ref_rec_chain_id, ref_pep_chain_id)
    ref_rec_coords = np.array(ref_rec_coords)
    ref_pep_coords = np.array(ref_pep_coords)

    def get_matching_region_coords(full_seq, coords, ref_seq):
        if ref_seq in full_seq:
            i = full_seq.index(ref_seq)
            return coords[i:i+len(ref_seq)]
        elif full_seq in ref_seq:
            return coords
        else:
            raise ValueError("No matching region found between chain and reference receptor sequence.")

    ref_rec_coords_trunc = get_matching_region_coords(ref_rec_seq, ref_rec_coords, ref_rec_seq)
    sup = SVDSuperimposer()
    new_rec_coords, new_pep_coords = parse_pdb(pdb_file, "A", "B")
    new_rec_coords = np.array(new_rec_coords)
    
    new_pep_coords = np.array(new_pep_coords)
    new_chain_seqs = get_chain_sequences(pdb_file)

    new_rec_seq = new_chain_seqs.get("A", "")
    new_pep_seq = new_chain_seqs.get("B", "")
    
    def find_matching_region(ref_seq, target_seq, target_coords):
        if ref_seq in target_seq:
            i = target_seq.index(ref_seq)
            return target_coords[i:i+len(ref_seq)]
        else:
            raise ValueError("Reference receptor sequence not found in target receptor sequence. Ensure that the PDB file has the correct protein sequence.")
    new_rec_coords_trunc = find_matching_region(ref_rec_seq, new_rec_seq, new_rec_coords)

    sup.set(ref_rec_coords_trunc, new_rec_coords_trunc)
    sup.run()
    rot, tran = sup.get_rotran()
    new_pep_coords_aligned = np.dot(new_pep_coords, rot) + tran
    distance = rmsd(ref_pep_coords, new_pep_coords_aligned)
    return distance


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    distance_matrix = align_and_compute_rmsd(ref_pdb_file=pdb_file_path, pdb_file=pdb_file_path)
    print(f"Distance matrix for {pdb_file_path}:")
    print(distance_matrix)
