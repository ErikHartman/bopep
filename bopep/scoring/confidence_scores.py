"""
Confidence scoring functions for calculating peptide-specific metrics from confidence data.

This module provides functions to calculate peptide-specific confidence scores
from matrices/vectors extracted from AlphaFold and Boltz predictions.
"""

from typing import Dict, Any, Optional, Tuple, List
from bopep.structure.parser import parse_structure, get_structure_residues
from Bio.PDB import Selection, NeighborSearch

def get_peptide_plddt(
    plddt_vector: List[float], 
    structure_file: str, 
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide pLDDT from vector using residue mapping.
    
    Args:
        plddt_vector: Vector of pLDDT values per residue
        structure_file: Path to structure file (PDB/CIF) for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average pLDDT for peptide chain or None if error
    """

    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for peptide chain
    peptide_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == peptide_chain.upper()
    ]
    
    if not peptide_indices:
        return None
    
    # Extract pLDDT values for peptide residues
    peptide_plddt_values = [plddt_vector[i] for i in peptide_indices]
    
    return sum(peptide_plddt_values) / len(peptide_plddt_values)
    



def get_weighted_peptide_plddt(
    plddt_vector: List[float],
    structure_file: str,
    peptide_chain: str = "B",
    protein_chain: str = "A",
    distance_threshold: float = 5.0
) -> Optional[float]:
    """
    Calculate interface peptide pLDDT from confidence vector based on contacts.
    
    This function weights peptide residues based on their contact with the protein,
    setting non-contact residues to 0 and averaging over all peptide residues.
    
    Args:
        plddt_vector: Vector of pLDDT values per residue
        structure_file: Path to structure file (PDB/CIF) for contact analysis
        peptide_chain: Chain ID of the peptide (default: "B")
        protein_chain: Chain ID of the protein (default: "A")
        distance_threshold: Distance threshold for defining contacts (default: 5.0 Å)
        
    Returns:
        Interface peptide pLDDT (contact-weighted average) or None if error
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
            chain_b = model[peptide_chain]
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
    
    # Map peptide residues to pLDDT values with contact weighting
    peptide_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == peptide_chain.upper()
    ]
    
    if not peptide_indices:
        return None
    
    # Get residue IDs for peptide chain
    peptide_residue_ids = [
        residue.get_id() for residue in chain_b
    ]
    
    # Apply contact-based weighting
    interface_weighted_values = []
    
    for i, plddt_idx in enumerate(peptide_indices):
        if i < len(peptide_residue_ids):
            residue_id = peptide_residue_ids[i]
            plddt_value = plddt_vector[plddt_idx]
            
            if residue_id in contact_residues_in_chain_b:
                # Contact residue - use actual pLDDT
                interface_weighted_values.append(plddt_value)
            else:
                # Non-contact residue - weight as zero
                interface_weighted_values.append(0.0)
    
    # Calculate interface peptide pLDDT average
    interface_peptide_plddt = sum(interface_weighted_values) / len(interface_weighted_values) if interface_weighted_values else None
    
    return interface_peptide_plddt



def get_peptide_pae(
    pae_matrix: List[List[float]],
    structure_file: str,
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide PAE from confidence matrix using residue mapping.
    
    Args:
        pae_matrix: 2D matrix of PAE values (residue x residue)
        structure_file: Path to structure file (PDB/CIF) for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average PAE for peptide chain or None if error
"""
    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for peptide chain
    peptide_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == peptide_chain.upper()
    ]
    
    if not peptide_indices:
        return None
    
    # Extract PAE values involving peptide residues
    all_pae_values = []
    for i in peptide_indices:
        for j in range(len(residue_chain_list)):
            all_pae_values.append(pae_matrix[i][j])
    
    return sum(all_pae_values) / len(all_pae_values) if all_pae_values else None
    

def get_peptide_pde(
    pde_matrix: List[List[float]],
    structure_file: str,
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide PDE (Protein Distance Error) from confidence matrix.
    
    Args:
        pde_matrix: 2D matrix of PDE values (residue x residue)
        structure_file: Path to structure file (PDB/CIF) for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average PDE for peptide chain or None if error
    """

    # Get residue chain mapping from structure file
    residue_chain_list = get_structure_residues(structure_file)
    if not residue_chain_list:
        return None
    
    # Find indices for peptide chain
    peptide_indices = [
        i for i, (chain, _) in enumerate(residue_chain_list) 
        if chain.upper() == peptide_chain.upper()
    ]
    
    if not peptide_indices:
        return None
    
    # Extract PDE values involving peptide residues
    all_pde_values = []
    for i in peptide_indices:
        for j in range(len(residue_chain_list)):
            all_pde_values.append(pde_matrix[i][j])
    
    return sum(all_pde_values) / len(all_pde_values) if all_pde_values else None
    


if __name__ == "__main__":
    # Example usage


    test_dir = "/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV"
    
    
    
    import json

    metrics = json.load(open(test_dir + "/alphafold_metrics.json"))

    print(metrics.keys())


    plddt = get_peptide_plddt(metrics["plddt"], test_dir + "/alphafold_model_1.pdb", peptide_chain="B")
    print(plddt)
    print(len(plddt))
    print(len(plddt[0]))
    iptm = metrics["iptm"]
    print(iptm)