"""
Confidence scoring functions for calculating peptide-specific metrics from confidence data.

This module provides functions to calculate peptide-specific confidence scores
from matrices/vectors extracted from AlphaFold and Boltz predictions.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from Bio.PDB import PDBParser
import re


def get_peptide_plddt_from_vector(
    plddt_vector: List[float], 
    pdb_file: str, 
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide pLDDT from confidence vector using residue mapping.
    
    Args:
        plddt_vector: Vector of pLDDT values per residue
        pdb_file: Path to PDB file for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average pLDDT for peptide chain or None if error
    """
    try:
        # Get residue chain mapping from PDB
        residue_chain_list = _parse_pdb_residues(pdb_file)
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
        
    except Exception as e:
        print(f"Error calculating peptide pLDDT from vector: {e}")
        return None


def get_weighted_peptide_plddt_from_vector(
    plddt_vector: List[float],
    pdb_file: str,
    peptide_chain: str = "B",
    protein_chain: str = "A",
    distance_threshold: float = 5.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate weighted peptide pLDDT from confidence vector based on contacts.
    
    Args:
        plddt_vector: Vector of pLDDT values per residue
        pdb_file: Path to PDB file for contact analysis
        peptide_chain: Chain ID of the peptide (default: "B")
        protein_chain: Chain ID of the protein (default: "A")
        distance_threshold: Distance threshold for defining contacts (default: 5.0 Å)
        
    Returns:
        Tuple of (overall_weighted_avg, residue_weighted_avg) or (None, None) if error
    """
    try:
        from Bio.PDB import Selection, NeighborSearch
        
        # Get residue chain mapping from PDB
        residue_chain_list = _parse_pdb_residues(pdb_file)
        if not residue_chain_list:
            return None, None
        
        # Parse PDB structure for contact analysis
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
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
            return None, None
        
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
            return None, None
        
        # Get residue IDs for peptide chain
        peptide_residue_ids = [
            residue.get_id() for residue in chain_b
        ]
        
        # Apply contact-based weighting
        weighted_values = []
        residue_weighted_values = []
        
        for i, plddt_idx in enumerate(peptide_indices):
            if i < len(peptide_residue_ids):
                residue_id = peptide_residue_ids[i]
                plddt_value = plddt_vector[plddt_idx]
                
                if residue_id in contact_residues_in_chain_b:
                    # Contact residue - use actual pLDDT
                    weighted_values.append(plddt_value)
                    residue_weighted_values.append(plddt_value)
                else:
                    # Non-contact residue - weight as zero
                    weighted_values.append(0.0)
                    residue_weighted_values.append(0.0)
        
        # Calculate averages
        overall_avg = sum(weighted_values) / len(weighted_values) if weighted_values else None
        residue_avg = sum(residue_weighted_values) / len(residue_weighted_values) if residue_weighted_values else None
        
        return overall_avg, residue_avg
        
    except Exception as e:
        print(f"Error calculating weighted peptide pLDDT from vector: {e}")
        return None, None


def get_peptide_pae_from_matrix(
    pae_matrix: List[List[float]],
    pdb_file: str,
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide PAE from confidence matrix using residue mapping.
    
    Args:
        pae_matrix: 2D matrix of PAE values (residue x residue)
        pdb_file: Path to PDB file for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average PAE for peptide chain or None if error
    """
    try:
        # Get residue chain mapping from PDB
        residue_chain_list = _parse_pdb_residues(pdb_file)
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
        
    except Exception as e:
        print(f"Error calculating peptide PAE from matrix: {e}")
        return None


def get_peptide_pde_from_vector(
    pde_vector: List[float],
    pdb_file: str,
    peptide_chain: str = "B"
) -> Optional[float]:
    """
    Calculate peptide PDE (Protein Distance Error) from confidence vector.
    
    Args:
        pde_vector: Vector of PDE values per residue
        pdb_file: Path to PDB file for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average PDE for peptide chain or None if error
    """
    try:
        # Get residue chain mapping from PDB
        residue_chain_list = _parse_pdb_residues(pdb_file)
        if not residue_chain_list:
            return None
        
        # Find indices for peptide chain
        peptide_indices = [
            i for i, (chain, _) in enumerate(residue_chain_list) 
            if chain.upper() == peptide_chain.upper()
        ]
        
        if not peptide_indices:
            return None
        
        # Extract PDE values for peptide residues
        peptide_pde_values = [pde_vector[i] for i in peptide_indices]
        
        return sum(peptide_pde_values) / len(peptide_pde_values)
        
    except Exception as e:
        print(f"Error calculating peptide PDE from vector: {e}")
        return None


def calculate_peptide_confidence_scores(
    metrics_data: Dict[str, Any],
    pdb_file: str,
    method: str,
    peptide_chain: str = "B",
    protein_chain: str = "A"
) -> Dict[str, Optional[float]]:
    """
    Calculate all peptide confidence scores from parsed metrics data.
    
    Args:
        metrics_data: Parsed metrics data containing confidence matrices/vectors
        pdb_file: Path to PDB file for residue mapping
        method: Method name ('alphafold' or 'boltz')
        peptide_chain: Chain ID of the peptide (default: "B")
        protein_chain: Chain ID of the protein (default: "A")
        
    Returns:
        Dict with peptide confidence scores
    """
    scores = {}
    
    try:
        # Calculate pLDDT scores
        plddt_key = f"{method}_plddt_vector"
        if plddt_key in metrics_data:
            plddt_vector = metrics_data[plddt_key]
            scores["peptide_plddt"] = get_peptide_plddt_from_vector(
                plddt_vector, pdb_file, peptide_chain
            )
            
            # Calculate weighted pLDDT
            weighted_overall, weighted_residues = get_weighted_peptide_plddt_from_vector(
                plddt_vector, pdb_file, peptide_chain, protein_chain
            )
            scores["weighted_plddt_overall"] = weighted_overall
            scores["weighted_plddt_residues"] = weighted_residues
        
        # Calculate PAE scores
        pae_key = f"{method}_pae_matrix"
        if pae_key in metrics_data:
            pae_matrix = metrics_data[pae_key]
            scores["peptide_pae"] = get_peptide_pae_from_matrix(
                pae_matrix, pdb_file, peptide_chain
            )
        
        # Calculate PDE scores (Boltz only)
        if method == "boltz":
            pde_key = f"{method}_pde_vector"
            if pde_key in metrics_data:
                pde_vector = metrics_data[pde_key]
                scores["peptide_pde"] = get_peptide_pde_from_vector(
                    pde_vector, pdb_file, peptide_chain
                )
        
    except Exception as e:
        print(f"Error calculating peptide confidence scores: {e}")
    
    return scores


def _parse_pdb_residues(pdb_file: str) -> List[Tuple[str, str]]:
    """
    Parse PDB file to build residue chain list in order.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        List of (chain_id, residue_num) tuples in order
    """
    residue_chain_list = []
    
    try:
        with open(pdb_file, "r") as f:
            last_chain_resid = None
            for line in f:
                if line.startswith("ATOM"):
                    chain_id = line[21]
                    residue_num = line[22:26].strip()
                    
                    chain_resid = (chain_id, residue_num)
                    if chain_resid != last_chain_resid:
                        residue_chain_list.append(chain_resid)
                        last_chain_resid = chain_resid
                        
    except IOError as e:
        print(f"Error reading PDB file: {e}")
        
    return residue_chain_list
