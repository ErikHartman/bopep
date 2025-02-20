from Bio.PDB import PDBParser
from typing import NamedTuple

import numpy as np


class ChainData(NamedTuple):
    """
    Simple container for chain-specific data:
      - coords: (N, 3) float ndarray
      - atoms: (N,) array of strings (atom names)
      - b_factors: (N,) float array
      - resnums: (N,) int array (residue numbers)
    """

    coords: np.ndarray
    atoms: np.ndarray
    b_factors: np.ndarray
    resnums: np.ndarray


def parse_pdb_to_chains(pdb_path: str):
    """
    Parses a PDB with at least two chains (assume FIRST is receptor, SECOND is peptide).
    Returns (receptor_data, peptide_data).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("single_complex", pdb_path)

    chain_data_list = []
    for chain in structure.get_chains():
        coords = []
        atoms = []
        b_factors = []
        resnums = []
        for residue in chain:
            # Skip non-standard or hetero residues
            if residue.id[0].strip() not in ("", " "):
                continue
            res_id = residue.id[1]
            for atom in residue:
                coords.append(atom.coord)
                atoms.append(atom.name)
                b_factors.append(atom.bfactor)
                resnums.append(res_id)

        chain_data_list.append(
            ChainData(
                coords=np.array(coords, dtype=float),
                atoms=np.array(atoms),
                b_factors=np.array(b_factors, dtype=float),
                resnums=np.array(resnums, dtype=int),
            )
        )

    if len(chain_data_list) < 2:
        raise ValueError(
            f"PDB should have at least 2 chains, found {len(chain_data_list)}"
        )

    # Return the first two chains as (receptor, peptide)
    return chain_data_list[0], chain_data_list[1]