import numpy as np
from src.scoring.util import ChainData, parse_pdb_to_chains
from .scorer import Scorer

"""
TODO:

Take some of the utility functions that can be used
by other scripts and put them in their own module.

Then, refactor the code below to use those utility functions.

This file should just contain a class for the EvoBind loss function.
"""


def find_interface_residues_by_cb(
    receptor: ChainData, peptide: ChainData, dist_cutoff: float = 8.0
) -> np.ndarray:
    """
    Identify receptor residue numbers (in 'receptor.resnums') that are
    within 'dist_cutoff' Å of any peptide Cβ.

    Returns
    -------
    interface_resnums : np.ndarray of unique receptor residue IDs
    """
    rec_cb_idx = np.where(receptor.atoms == "CB")[0]
    pep_cb_idx = np.where(peptide.atoms == "CB")[0]

    if rec_cb_idx.size == 0 or pep_cb_idx.size == 0:
        return np.array([], dtype=int)  # No Cβ atoms to compare

    rec_cb_coords = receptor.coords[rec_cb_idx]
    pep_cb_coords = peptide.coords[pep_cb_idx]

    # Build distance matrix between rec_CB and pep_CB
    diff = rec_cb_coords[:, None, :] - pep_cb_coords[None, :, :]
    dmat = np.sqrt(np.sum(diff**2, axis=-1))  # shape (num_rec_CB, num_pep_CB)

    # For each receptor CB, check if there's any peptide CB < dist_cutoff
    interface_mask = np.any(dmat < dist_cutoff, axis=1)
    interface_cb_indices = rec_cb_idx[interface_mask]

    # Map CB indices -> actual residue numbers
    interface_resnums = np.unique(receptor.resnums[interface_cb_indices])
    return interface_resnums


def atom_distance_matrix(coordsA: np.ndarray, coordsB: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise distances between coordsA and coordsB.
    coordsA shape = (N, 3), coordsB shape = (M, 3)
    Returns dmat shape = (N, M).
    """
    diff = coordsA[:, None, :] - coordsB[None, :, :]
    dmat = np.sqrt(np.sum(diff**2, axis=-1))
    return dmat


def compute_min_dist_stats(
    receptor: ChainData, peptide: ChainData, interface_resnums: np.ndarray
):
    """
    Given a list of receptor residue IDs that define the "interface",
    build a distance matrix vs. ALL peptide atoms to get:
      - rec_if_dist: average minimal distance from each receptor interface residue to peptide
      - pep_if_dist: average minimal distance from each peptide residue to receptor interface

    If no interface residues exist, returns (20.0, 20.0) as a fallback.
    """
    if interface_resnums.size == 0:
        # No interface found
        return (20.0, 20.0)

    # For each interface residue, gather all atoms from that residue
    # (just like the original code did).
    iface_atom_indices = [
        i for i, rno in enumerate(receptor.resnums) if rno in interface_resnums
    ]
    if len(iface_atom_indices) == 0:
        return (20.0, 20.0)

    rec_iface_coords = receptor.coords[iface_atom_indices]
    pep_coords = peptide.coords  # all peptide atoms

    dmat = atom_distance_matrix(rec_iface_coords, pep_coords)

    # rec_if_dist = average of (min distance from each interface residue "row" to any pep atom)
    rec_if_dist = np.mean(np.min(dmat, axis=1))

    pep_if_dist = np.mean(np.min(dmat, axis=0))

    return (rec_if_dist, pep_if_dist)


class EvoBindLoss(Scorer):
    """
    Class that loads a PDB with two chains (receptor, peptide)
    and computes the "loss" function:
        loss = ((rec_if_dist + pep_if_dist) / 2) * (1 / pep_pLDDT)
    """

    def __init__(self, dist_cutoff: float = 8.0):
        self.dist_cutoff = dist_cutoff 
        self.receptor = None
        self.peptide = None
        self.pep_plddt = 0.0
        pass

    def compute_loss(self) -> float:
        """
        1) Identify interface residues in the receptor via Cβ.
        2) Compute rec_if_dist and pep_if_dist for those residues vs. the entire peptide.
        3) Compute pLDDT (from B-factors).
        4) Return the final loss.

        loss = ((rec_if_dist + pep_if_dist)/2) * (1 / pep_pLDDT)
        """
        interface_resnums = find_interface_residues_by_cb(
            self.receptor, self.peptide, self.dist_cutoff
        )
        rec_if_dist, pep_if_dist = compute_min_dist_stats(
            self.receptor, self.peptide, interface_resnums
        )
        loss_value = ((rec_if_dist + pep_if_dist) / 2.0) * (1.0 / self.pep_plddt)
        return loss_value
    
    def score(self, pdb_path: str):
        pdb_path = pdb_path
        self.receptor, self.peptide = parse_pdb_to_chains(pdb_path)
        self.pep_plddt = (
            np.mean(self.peptide.b_factors) if len(self.peptide.b_factors) > 0 else 0.0
        )
        return self.compute_loss()
