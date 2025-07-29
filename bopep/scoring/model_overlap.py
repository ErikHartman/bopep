import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
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

    # Logging: Print AA sequence and chain IDs for reference PDB
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1
    def get_chain_sequence(pdb_file, chain_id):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('struct', pdb_file)
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    seq = []
                    for residue in chain:
                        if residue.id[0] == ' ':
                            resname = residue.get_resname()
                            try:
                                seq.append(protein_letters_3to1[resname.capitalize()])
                            except KeyError:
                                seq.append('X')
                    return ''.join(seq)
        return None

    ref_rec_coords, ref_pep_coords = parse_pdb(ref_pdb_file, "B", "A")
    ref_rec_coords = np.array(ref_rec_coords)
    ref_pep_coords = np.array(ref_pep_coords)

    ref_pep_seq = get_chain_sequence(ref_pdb_file, "A")
    ref_rec_seq = get_chain_sequence(ref_pdb_file, "B")
    print(f"Reference PDB: Peptide chain A sequence: {ref_pep_seq}")
    print(f"Reference PDB: Binder chain B sequence: {ref_rec_seq}")
    print(f"Reference PDB: Peptide chain ID: A, Binder chain ID: B")

    # Truncate to matching region if receptor is a substring
    # (Assume we want to align the region of the receptor chain that matches the reference receptor)
    def get_matching_region_coords(full_seq, coords, ref_seq):
        if ref_seq in full_seq:
            start = full_seq.index(ref_seq)
            end = start + len(ref_seq)
            return coords[start:end]
        elif full_seq in ref_seq:
            # If the chain is a substring of the reference, use all
            return coords
        else:
            raise ValueError("No matching region found between chain and reference receptor sequence.")

    # For reference
    ref_rec_seq_full = ref_rec_seq
    ref_rec_seq_ref = ref_rec_seq  # For reference, align to itself
    ref_rec_coords_trunc = get_matching_region_coords(ref_rec_seq_full, ref_rec_coords, ref_rec_seq_ref)

    # Truncate reference peptide to itself (no-op, but for symmetry)
    def find_matching_peptide_region(ref_seq, target_seq, target_coords):
        if ref_seq in target_seq:
            start = target_seq.index(ref_seq)
            end = start + len(ref_seq)
            return target_coords[start:end]
        else:
            raise ValueError("Reference peptide sequence not found in target peptide sequence.")

    ref_pep_coords_trunc = ref_pep_coords
    aligned_peptide_coords = [ref_pep_coords_trunc]
    sup = SVDSuperimposer()

    new_rec_coords, new_pep_coords = parse_pdb(
        pdb_file, "A", "B"
    )
    new_rec_coords = np.array(new_rec_coords)
    new_pep_coords = np.array(new_pep_coords)

    new_pep_seq = get_chain_sequence(pdb_file, "B")
    new_rec_seq = get_chain_sequence(pdb_file, "A")
    print(f"Target PDB: Peptide chain B sequence: {new_pep_seq}")
    print(f"Target PDB: Binder chain A sequence: {new_rec_seq}")
    print(f"Target PDB: Peptide chain ID: B, Binder chain ID: A")

    # Truncate target receptor to region matching reference receptor
    def find_matching_region(ref_seq, target_seq, target_coords):
        if ref_seq in target_seq:
            start = target_seq.index(ref_seq)
            end = start + len(ref_seq)
            return target_coords[start:end]
        else:
            raise ValueError("Reference receptor sequence not found in target receptor sequence.")

    new_rec_coords_trunc = find_matching_region(ref_rec_seq, new_rec_seq, new_rec_coords)

    # Truncate target peptide to region matching reference peptide
    new_pep_coords_trunc = find_matching_peptide_region(ref_pep_seq, new_pep_seq, new_pep_coords)

    sup.set(ref_rec_coords_trunc, new_rec_coords_trunc)
    sup.run()
    rot, tran = sup.get_rotran()

    new_pep_coords_aligned = np.dot(new_pep_coords_trunc, rot) + tran
    aligned_peptide_coords.append(new_pep_coords_aligned)

    distance = rmsd(ref_pep_coords_trunc, new_pep_coords_aligned)
    return distance


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    distance_matrix = align_and_compute_rmsd(ref_pdb_file=pdb_file_path, pdb_file=pdb_file_path)
    print(f"Distance matrix for {pdb_file_path}:")
    print(distance_matrix)
