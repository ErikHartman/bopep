import math
import numpy as np
from bopep.scoring.util import parse_pdb

def evobind_loss_from_pdb(pdb_file_path,
                          receptor_chain='A',
                          peptide_chain='B'):
    """
    Convenience function that does:
      1) Parse the PDB to get coords/bfactors
      2) Compute the EvoBind score

    :param pdb_file_path: Path to the PDB file
    :param receptor_chain: The chain ID(s) for the receptor
    :param peptide_chain: The chain ID(s) for the peptide
    :return: EvoBind score (float)
    """
    (rec_coords, rec_bfactors,
     pep_coords, pep_bfactors) = parse_pdb(
                                    pdb_file_path,
                                    receptor_chain=receptor_chain,
                                    peptide_chain=peptide_chain
                                 )
    score = compute_evobind_score(rec_coords, rec_bfactors,
                                  pep_coords, pep_bfactors)
    return score

def compute_evobind_score(
    receptor_coords, receptor_bfactors,
    peptide_coords, peptide_bfactors
):
    """
    Computes the (modified) EvoBind loss:
        score = (sum_d_pep_rec + sum_d_rec_pep) * pLDDT
    where:
        - sum_d_pep_rec is the sum of distances between each peptide
          atom and its closest receptor atom
        - sum_d_rec_pep is the sum of distances between each receptor
          atom and its closest peptide atom
        - pLDDT is taken as the average of B-factors (assuming pLDDT stored in B-factor)

    :param receptor_coords: list of (x, y, z) for the receptor
    :param receptor_bfactors: list of floats for the receptor
    :param peptide_coords: list of (x, y, z) for the peptide
    :param peptide_bfactors: list of floats for the peptide
    :return: EvoBind score (float)
    """
    
    sum_d_pep_rec = 0.0
    for p_atom in peptide_coords:
        min_dist = float('inf')
        for r_atom in receptor_coords:
            dist = math.dist(p_atom, r_atom)
            if dist < min_dist:
                min_dist = dist
        sum_d_pep_rec += min_dist
    mean_d_pep_rec = sum_d_pep_rec / len(peptide_coords) if peptide_coords else 0.0

    sum_d_rec_pep = 0.0
    for r_atom in receptor_coords:
        min_dist = float('inf')
        for p_atom in peptide_coords:
            dist = math.dist(r_atom, p_atom)
            if dist < min_dist:
                min_dist = dist
        sum_d_rec_pep += min_dist
    mean_d_rec_pep = sum_d_rec_pep / len(receptor_coords) if receptor_coords else 0.0

    all_bfactors = receptor_bfactors + peptide_bfactors
    plddt = np.mean(all_bfactors) if len(all_bfactors) > 0 else 1
    evobind_score = (mean_d_pep_rec + mean_d_rec_pep) * (1/plddt)

    return evobind_score

if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    score = evobind_loss_from_pdb(pdb_file_path)
    print(f"EvoBind score for {pdb_file_path}: {score}")