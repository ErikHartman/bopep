from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial import cKDTree
import os
import json
import re


def parse_pdb(pdb_file_path, receptor_chain="A", peptide_chain="B"):
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
    structure = parser.get_structure(id="complex", file=pdb_file_path)

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


def get_plDDT_from_dir(colab_dir, rank_num : int = 1):
    """
    Extracts the plDDT score from an unzipped docking result directory.
    """
    if not os.path.isdir(colab_dir):
        print(f"Directory {colab_dir} does not exist.")
        return None

    json_pattern = re.compile(fr".*_scores_rank_00{rank_num}_.*\.json")
    json_files = []

    for root, _, files in os.walk(colab_dir):
        json_files.extend(
            [os.path.join(root, f) for f in files if json_pattern.search(f)]
        )

    if not json_files:
        print(f"No matching JSON file found in {colab_dir}")
        return None

    try:
        with open(json_files[0], "r") as f:
            return json.load(f).get("plDDT")
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return None
