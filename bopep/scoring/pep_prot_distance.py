import math
from bopep.scoring.util import parse_pdb
from bopep.scoring.util import get_plDDT_from_dir

"""
Distance-based loss function for protein-peptide interactions.

Inspired by EvoBind.
"""


def distance_score_from_pdb(pdb_file_path, receptor_chain="A", peptide_chain="B"):
    """
    Convenience function that:
      1) Parses the PDB to get coordinates/bfactors
      2) Computes the distance-based score
      3) Normalizes by plDDT

    :param pdb_file_path: Path to the PDB file
    :param receptor_chain: The chain ID(s) for the receptor
    :param peptide_chain: The chain ID(s) for the peptide
    :return: Distance-based score (float)
    """
    (rec_coords, rec_bfactors, pep_coords, pep_bfactors) = parse_pdb(
        pdb_file_path, receptor_chain=receptor_chain, peptide_chain=peptide_chain
    )
    distance_score = compute_distance_score(rec_coords, pep_coords)
    plDDT = get_plDDT_from_dir(pdb_file_path)
    score = distance_score * (1 / plDDT)
    return score


def compute_distance_score(receptor_coords, peptide_coords):
    """
    Computes a distance-based score between receptor and peptide:
        score = mean_d_pep_rec + mean_d_rec_pep
    where:
        - mean_d_pep_rec is the average distance between each peptide
          atom and its closest receptor atom
        - mean_d_rec_pep is the average distance between each receptor
          atom and its closest peptide atom
    """
    # Calculate peptide-to-receptor distances
    sum_d_pep_rec = 0.0
    for p_atom in peptide_coords:
        min_dist = float("inf")
        for r_atom in receptor_coords:
            dist = math.dist(p_atom, r_atom)
            if dist < min_dist:
                min_dist = dist
        sum_d_pep_rec += min_dist
    mean_d_pep_rec = sum_d_pep_rec / len(peptide_coords) if peptide_coords else 0.0

    # Calculate receptor-to-peptide distances
    sum_d_rec_pep = 0.0
    for r_atom in receptor_coords:
        min_dist = float("inf")
        for p_atom in peptide_coords:
            dist = math.dist(r_atom, p_atom)
            if dist < min_dist:
                min_dist = dist
        sum_d_rec_pep += min_dist
    mean_d_rec_pep = sum_d_rec_pep / len(receptor_coords) if receptor_coords else 0.0

    distance_score = mean_d_pep_rec + mean_d_rec_pep

    return distance_score


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    score = distance_score_from_pdb(pdb_file_path)
    print(f"Distance score for {pdb_file_path}: {score}")
