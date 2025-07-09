import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from bopep.scoring.utils import parse_pdb

def rmsd(coords1, coords2):
    return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

def align_and_compute_rmsd(
    ref_pdb_file, pdb_file, alignment_chain="A", peptide_chain="B"
):
    """
    Given multiple PDBs, align each structure's receptor (chain A by default)
    and compute a distance matrix of peptide RMSDs between all pairs.

    Each structure's receptor is aligned to the first PDB's receptor, then
    peptide RMSD is computed between all pairs of aligned structures.
    """
    (ref_rec_coords, ref_pep_coords) = parse_pdb(
        ref_pdb_file, alignment_chain, peptide_chain
    )
    ref_rec_coords = np.array(ref_rec_coords)
    ref_pep_coords = np.array(ref_pep_coords)

    aligned_peptide_coords = [ref_pep_coords]
    
    sup = SVDSuperimposer()


    (new_rec_coords, new_pep_coords) = parse_pdb(
        pdb_file, alignment_chain, peptide_chain
    )
    new_rec_coords = np.array(new_rec_coords)
    new_pep_coords = np.array(new_pep_coords)
    sup.set(ref_rec_coords, new_rec_coords)
    sup.run()
    rot, tran = sup.get_rotran()

    new_pep_coords_aligned = np.dot(new_pep_coords, rot) + tran
    aligned_peptide_coords.append(new_pep_coords_aligned)

    distance = rmsd(ref_pep_coords, new_pep_coords_aligned)
    return distance


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    distance_matrix = align_and_compute_rmsd(ref_pdb_file=pdb_file_path, pdb_file=pdb_file_path)
    print(f"Distance matrix for {pdb_file_path}:")
    print(distance_matrix)
