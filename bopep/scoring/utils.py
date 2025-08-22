from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import os
import json
import re

def find_relaxed_pdb_file(colab_dir):
    """
    Finds the relaxed rank_001 PDB file in the colab directory.
    
    :param colab_dir: Path to the colab directory containing PDB files.
    
    :return: Path to the first matching PDB file or None if not found.
    """
    pdb_pattern = re.compile(r".*_relaxed_rank_001_.*\.pdb")
    for root, _, files in os.walk(colab_dir):
        for file in files:
            if pdb_pattern.search(file):
                return os.path.join(root, file)
    return None


def parse_pdb(pdb_file_path, receptor_chain="A", peptide_chain="B"):
    """
    Parses a PDB file using BioPython and returns coordinates & B-factors
    for receptor and peptide atoms separately.

    :param pdb_file_path: Path to the PDB file
    :param receptor_chain: Chain ID for the receptor
    :param peptide_chain: Chain ID for the peptide

    :return: (
        receptor_coords: list of (x, y, z),
        peptide_coords: list of (x, y, z),
    )
    """
    if pdb_file_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True, auth_residues=False)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file_path)

    receptor_coords = []
    peptide_coords = []

    model = structure[0]
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            for atom in residue:
                # if atom == alpha carbon
                atom_type = atom.get_id()
                if atom_type != "CA":
                    continue    
                x, y, z = atom.coord
                if chain_id == receptor_chain:
                    receptor_coords.append((x, y, z))
                elif chain_id == peptide_chain:
                    peptide_coords.append((x, y, z))

    return receptor_coords, peptide_coords


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

def get_chain_sequences(pdb_file):
     if pdb_file.endswith('.cif'):
         parser = MMCIFParser(QUIET=True)
     else:
         parser = PDBParser(QUIET=True)
     structure = parser.get_structure('struct', pdb_file)
     return {chain.id: ''.join([
         protein_letters_3to1.get(residue.get_resname().capitalize(), 'X')
         for residue in chain if residue.id[0] == ' '
     ]) for model in structure for chain in model}

def match_and_truncate(ref_seq, ref_coords, target_seq, target_coords):
    if ref_seq in target_seq:
        i = target_seq.index(ref_seq)
        return ref_coords, target_coords[i:i+len(ref_seq)]
    elif target_seq in ref_seq:
        i = ref_seq.index(target_seq)
        return ref_coords[i:i+len(target_seq)], target_coords
    else:
        raise ValueError("Could not match reference and target receptor sequences for alignment.")