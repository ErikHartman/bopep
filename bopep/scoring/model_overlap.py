import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

from util import parse_pdb


def count_agreeing_pdbs(
    pdb_files, receptor_chain="A", peptide_chain="B", rmsd_threshold: float = 3.0
):
    """
    Given multiple PDBs, align each structure's receptor (chain A by default)
    to the receptor of the *first* PDB, then check if their peptide (chain B by default)
    is in the 'same site' as the reference's peptide by measuring peptide RMSD.

    :param pdb_files: list of paths to PDB files
    :param receptor_chain: chain ID for the receptor (defaults to 'A')
    :param peptide_chain: chain ID for the peptide (defaults to 'B')
    :param rmsd_threshold: maximum RMSD (Å) for considering them the same binding site
    :return: (int) number of PDBs whose peptides bind the same site as the reference
    """
    if not pdb_files:
        print("No PDB files provided.")
        return 0

    ref_pdb = pdb_files[0]
    (ref_rec_coords, _, ref_pep_coords, _) = parse_pdb(
        ref_pdb, receptor_chain, peptide_chain
    )
    ref_rec_coords = np.array(ref_rec_coords)  # shape (N, 3)
    ref_pep_coords = np.array(ref_pep_coords)  # shape (M, 3)

    def rmsd(coords1, coords2):
        return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

    agreeing_count = 0
    sup = SVDSuperimposer()

    for pdb_file in pdb_files:
        # Parse
        (new_rec_coords, _, new_pep_coords, _) = parse_pdb(
            pdb_file, receptor_chain, peptide_chain
        )
        new_rec_coords = np.array(new_rec_coords)
        new_pep_coords = np.array(new_pep_coords)

        sup.set(ref_rec_coords, new_rec_coords)
        sup.run()
        rot, tran = sup.get_rotran()

        # Apply transform to new peptide coords
        new_pep_coords_aligned = np.dot(new_pep_coords, rot) + tran

        # Compute peptide RMSD vs. reference
        current_rmsd = rmsd(ref_pep_coords, new_pep_coords_aligned)

        if current_rmsd <= rmsd_threshold:
            agreeing_count += 1

    return agreeing_count


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    agreeing_count = count_agreeing_pdbs([pdb_file_path, pdb_file_path])
    print(f"Nr overlapping models: {pdb_file_path}: {agreeing_count}")
