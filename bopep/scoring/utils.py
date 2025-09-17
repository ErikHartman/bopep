from Bio.Data.IUPACData import protein_letters_3to1
from bopep.structure.parser import parse_structure as _parse_structure

def get_receptor_peptide_coords(structure_file_path : str, receptor_chain : str="A", peptide_chain : str="B"):
    """
    Extracts alpha carbon coordinates for receptor and peptide chains from a structure file.

    :param structure_file_path: Path to the structure file (.pdb, .cif, .pdbx, .mmcif)
    :param receptor_chain: Chain ID for the receptor
    :param peptide_chain: Chain ID for the peptide
    :return: tuple (
        receptor_coords: list of (x, y, z),
        peptide_coords: list of (x, y, z),
    )
    """
    structure = _parse_structure(structure_file_path, structure_id="complex", auth_residues=False)

    receptor_coords = []
    peptide_coords = []

    model = structure[0]
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            # Only consider amino acid residues (skip ligands, waters, etc.)
            if residue.id[0] != ' ':
                continue
            
            # Get alpha carbon coordinates
            if 'CA' in residue:
                ca_atom = residue['CA']
                coords = ca_atom.get_coord()
                
                if chain_id == receptor_chain:
                    receptor_coords.append(coords)
                elif chain_id == peptide_chain:
                    peptide_coords.append(coords)

    return receptor_coords, peptide_coords

def get_chain_sequences(structure_file : str):
     structure = _parse_structure(structure_file, structure_id='struct')
     return {chain.id: ''.join([
         protein_letters_3to1.get(residue.get_resname().capitalize(), 'X')
         for residue in chain if residue.id[0] == ' '
     ]) for model in structure for chain in model}

def match_and_truncate(ref_seq :  str, ref_coords : list, target_seq : str, target_coords : list):
    if ref_seq in target_seq:
        i = target_seq.index(ref_seq)
        return ref_coords, target_coords[i:i+len(ref_seq)]
    elif target_seq in ref_seq:
        i = ref_seq.index(target_seq)
        return ref_coords[i:i+len(target_seq)], target_coords
    else:
        raise ValueError(f"Could not match reference and target receptor sequences for alignment. Reference sequence: {ref_seq}, Target sequence: {target_seq}")