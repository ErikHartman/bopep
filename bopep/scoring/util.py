from io import StringIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np
from scipy.spatial import cKDTree

def parse_pdb(pdb_file_path,
                             receptor_chain='A',
                             peptide_chain='B'):
    """
    Parses a PDB file using BioPython and returns coordinates & B-factors
    for receptor and peptide atoms separately.

    :param pdb_file_path: Path to the PDB file
    :param receptor_chain: Chain ID for the receptor
    :param peptide_chain: Chain ID for the peptide

    :return: (
        receptor_coords: list of (x, y, z),
        receptor_bfactors: list of float,
        peptide_coords: list of (x, y, z),
        peptide_bfactors: list of float
    )
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(id='complex', file=pdb_file_path)

    receptor_coords = []
    receptor_bfactors = []
    peptide_coords = []
    peptide_bfactors = []

    model = structure[0]
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            for atom in residue:
                x, y, z = atom.coord
                bfactor = atom.bfactor

                if chain_id == receptor_chain:
                    receptor_coords.append((x, y, z))
                    receptor_bfactors.append(bfactor)
                elif chain_id == peptide_chain:
                    peptide_coords.append((x, y, z))
                    peptide_bfactors.append(bfactor)

    return receptor_coords, receptor_bfactors, peptide_coords, peptide_bfactors


def is_peptide_in_binding_site(
    pdb_file, binding_site_residue_indices, threshold=5.0
):
    """
    TODO: Update this to make it more general.
    """
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure("temp_struc", pdb_file)
    except Exception as e:
        print(f"Error parsing PDB content: {e}")
        return False

    try:
        model = structure[0]
        receptor_chain = model["A"]
        peptide_chain = model["B"]

        receptor_binding_site_atoms = [
            atom
            for residue in receptor_chain
            if residue.id[1] in binding_site_residue_indices
            for atom in residue.get_atoms()
        ]

        if not receptor_binding_site_atoms:
            print(f"No atoms found in specified receptor binding site residues.")
            return False

        peptide_atoms = list(peptide_chain.get_atoms())
        if not peptide_atoms:
            print(f"No atoms found in peptide chain.")
            return False

        # Find atoms within distance threshold
        receptor_coords = np.array([atom.coord for atom in receptor_binding_site_atoms])
        peptide_coords = np.array([atom.coord for atom in peptide_atoms])

        tree = cKDTree(peptide_coords)
        distances, _ = tree.query(receptor_coords, distance_upper_bound=threshold)
        return np.any(distances != float("inf")) # any or all?

    except KeyError:
        print(f"Chains A or B not found in structure.")
        return False