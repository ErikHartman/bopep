from typing import Dict, Any, Optional, Tuple, List
from bopep.structure.parser import parse_structure, get_structure_residues
from Bio.PDB import Selection, NeighborSearch

def get_sequence_plddt(
    plddt_vector: List[float], 
    structure_file: str, 
    sequence_chain: str = "B"
) -> Optional[float]:
    """
    Calculate sequence pLDDT from vector using residue mapping.
    """

    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for sequence chain
    sequence_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == sequence_chain.upper()
    ]
    
    if not sequence_indices:
        return None
    
    # Extract pLDDT values for sequence residues
    sequence_plddt_values = [plddt_vector[i] for i in sequence_indices]
    
    return sum(sequence_plddt_values) / len(sequence_plddt_values)
    



def get_weighted_sequence_plddt(
    plddt_vector: List[float],
    structure_file: str,
    sequence_chain: str = "B",
    protein_chain: str = "A",
    distance_threshold: float = 5.0
) -> Optional[float]:
    """
    Calculate interface sequence pLDDT from confidence vector based on contacts.
    """

    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Parse structure for contact analysis
    structure = parse_structure(structure_file, structure_id='structure')
    
    # Find chains
    chain_a = None
    chain_b = None
    for model in structure:
        try:
            chain_a = model[protein_chain]
            chain_b = model[sequence_chain]
            break
        except KeyError:
            continue
    
    if not chain_a or not chain_b:
        return None
    
    # Identify contact residues
    atoms_in_chain_a = Selection.unfold_entities(chain_a, 'A')
    atoms_in_chain_b = Selection.unfold_entities(chain_b, 'A')
    
    ns = NeighborSearch(atoms_in_chain_a)
    contact_residues_in_chain_b = set()
    
    for atom in atoms_in_chain_b:
        neighbors = ns.search(atom.coord, distance_threshold, 'R')
        if neighbors:
            contact_residues_in_chain_b.add(atom.get_parent().get_id())
    
    # Map sequence residues to pLDDT values with contact weighting
    sequence_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == sequence_chain.upper()
    ]
    
    if not sequence_indices:
        return None
    
    # Get residue IDs for sequence chain
    sequence_residue_ids = [
        residue.get_id() for residue in chain_b
    ]
    
    # Apply contact-based weighting
    interface_weighted_values = []
    
    for i, plddt_idx in enumerate(sequence_indices):
        if i < len(sequence_residue_ids):
            residue_id = sequence_residue_ids[i]
            plddt_value = plddt_vector[plddt_idx]
            
            if residue_id in contact_residues_in_chain_b:
                # Contact residue - use actual pLDDT
                interface_weighted_values.append(plddt_value)
            else:
                # Non-contact residue - weight as zero
                interface_weighted_values.append(0.0)
    
    # Calculate interface sequence pLDDT average
    interface_sequence_plddt = sum(interface_weighted_values) / len(interface_weighted_values) if interface_weighted_values else None
    
    return interface_sequence_plddt



def get_sequence_pae(
    pae_matrix: List[List[float]],
    structure_file: str,
    sequence_chain: str = "B"
) -> Optional[float]:
    """
    Calculate sequence PAE from confidence matrix using residue mapping.
"""
    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for sequence chain
    sequence_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == sequence_chain.upper()
    ]
    
    if not sequence_indices:
        return None
    
    # Extract PAE values involving sequence residues
    all_pae_values = []
    for i in sequence_indices:
        for j in range(len(residue_chain_list)):
            all_pae_values.append(pae_matrix[i][j])
    
    return sum(all_pae_values) / len(all_pae_values) if all_pae_values else None
    

def get_sequence_pde(
    pde_matrix: List[List[float]],
    structure_file: str,
    sequence_chain: str = "B"
) -> Optional[float]:
    """
    Calculate sequence PDE (Protein Distance Error) from confidence matrix.
    """

    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for sequence chain
    sequence_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == sequence_chain.upper()
    ]
    
    if not sequence_indices:
        return None
    
    # Extract PDE values involving sequence residues
    all_pde_values = []
    for i in sequence_indices:
        for j in range(len(residue_chain_list)):
            all_pde_values.append(pde_matrix[i][j])
    
    return sum(all_pde_values) / len(all_pde_values) if all_pde_values else None
    


if __name__ == "__main__":
    # Example usage
    import json

    test_dir = "/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV"
    metrics = json.load(open(test_dir + "/alphafold_metrics.json"))

    print(metrics.keys())


    plddt = get_sequence_plddt(metrics["plddt"], test_dir + "/alphafold_model_1.pdb", sequence_chain="B")
    print(plddt)
    print(len(plddt))
    print(len(plddt[0]))
    iptm = metrics["iptm"]
    print(iptm)