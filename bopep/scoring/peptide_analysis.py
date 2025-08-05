"""
Peptide analysis functions for extracting peptide-specific metrics from structure predictions.

This module provides functions to calculate peptide-specific scores from PDB files
and metrics data, supporting both AlphaFold and Boltz outputs.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple, List
from Bio.PDB import PDBParser, Selection, NeighborSearch
from collections import defaultdict


def get_peptide_plddt_from_metrics(metrics_data: Dict[str, Any], method: str) -> Optional[float]:
    """
    Extract peptide pLDDT from parsed metrics data.
    
    Args:
        metrics_data: Parsed metrics data from MetricsParser
        method: Method name ('alphafold' or 'boltz')
        
    Returns:
        Average peptide pLDDT or None if not available
    """
    plddt_key = f"{method}_plddt"
    return metrics_data.get(plddt_key)


def get_peptide_pae_from_metrics(metrics_data: Dict[str, Any], method: str) -> Optional[float]:
    """
    Extract peptide PAE from parsed metrics data.
    
    Args:
        metrics_data: Parsed metrics data from MetricsParser
        method: Method name ('alphafold' or 'boltz')
        
    Returns:
        Average peptide PAE or None if not available
    """
    pae_key = f"{method}_pae"
    return metrics_data.get(pae_key)


def get_peptide_plddt_from_pdb(pdb_file: str, peptide_chain: str = "B") -> Optional[float]:
    """
    Calculate peptide pLDDT from PDB file using B-factors.
    
    Args:
        pdb_file: Path to PDB file
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average pLDDT for peptide chain or None if error
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        for model in structure:
            if peptide_chain in model:
                chain = model[peptide_chain]
                bfactors = []
                
                for residue in chain:
                    for atom in residue:
                        bfactors.append(atom.get_bfactor())
                
                if bfactors:
                    return sum(bfactors) / len(bfactors)
        
        return None
        
    except Exception as e:
        print(f"Error calculating peptide pLDDT from PDB: {e}")
        return None


def get_weighted_peptide_plddt_from_pdb(
    pdb_file: str, 
    peptide_chain: str = "B", 
    protein_chain: str = "A", 
    distance_threshold: float = 5.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate weighted peptide pLDDT based on contact residues with protein.
    
    Args:
        pdb_file: Path to PDB file
        peptide_chain: Chain ID of the peptide (default: "B")
        protein_chain: Chain ID of the protein (default: "A")
        distance_threshold: Distance threshold for defining contacts (default: 5.0 Å)
        
    Returns:
        Tuple of (overall_weighted_avg, residue_weighted_avg) or (None, None) if error
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_file)
        
        # Extract chains
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
            print(f"Could not find chains {protein_chain} and/or {peptide_chain}")
            return None, None
        
        # Identify contact residues in peptide chain
        atoms_in_chain_a = Selection.unfold_entities(chain_a, 'A')
        atoms_in_chain_b = Selection.unfold_entities(chain_b, 'A')

        # Find contacts using NeighborSearch
        ns = NeighborSearch(atoms_in_chain_a)
        contact_residues_in_chain_b = set()
        
        for atom in atoms_in_chain_b:
            neighbors = ns.search(atom.coord, distance_threshold, 'R')
            if neighbors:
                contact_residues_in_chain_b.add(atom.get_parent().get_id())

        # Collect B-factors for each residue in peptide chain
        residue_bfactors = defaultdict(list)
        
        for residue in chain_b:
            residue_id = residue.get_id()
            if residue_id in contact_residues_in_chain_b:
                # Contact residue - use actual B-factors
                for atom in residue:
                    residue_bfactors[residue_id].append(atom.get_bfactor())
            else:
                # Non-contact residue - weight as zero
                residue_bfactors[residue_id].append(0)

        # Calculate averages
        avg_bfactors_per_residue = {
            res_id: sum(bfactors)/len(bfactors) 
            for res_id, bfactors in residue_bfactors.items()
        }
        
        # Overall average (all atoms)
        all_bfactors = [
            bfactor for bfactors in residue_bfactors.values() 
            for bfactor in bfactors
        ]
        overall_avg = sum(all_bfactors) / len(all_bfactors) if all_bfactors else 0
        
        # Residue average
        residue_avg = (
            sum(avg_bfactors_per_residue.values()) / len(avg_bfactors_per_residue) 
            if avg_bfactors_per_residue else 0
        )
        
        return overall_avg, residue_avg
        
    except Exception as e:
        print(f"Error calculating weighted peptide pLDDT: {e}")
        return None, None


def get_peptide_pae_from_json(json_file: str, pdb_file: str, peptide_chain: str = "B") -> Optional[float]:
    """
    Calculate peptide PAE from JSON metrics file and PDB structure.
    
    Args:
        json_file: Path to JSON metrics file
        pdb_file: Path to PDB file for residue mapping
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Average PAE for peptide chain or None if error
    """
    try:
        # Load metrics data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get PAE matrix
        pae_array = data.get("pae")
        if not pae_array:
            return None
        
        # Parse PDB to get residue chain mapping
        residue_chain_list = _parse_pdb_residues(pdb_file)
        if not residue_chain_list:
            return None
        
        # Find indices for peptide chain
        chain_indices = [
            i for i, (chain, _) in enumerate(residue_chain_list) 
            if chain.upper() == peptide_chain.upper()
        ]
        
        if not chain_indices:
            return None
        
        # Extract PAE values involving peptide residues
        all_pae_values = []
        for i in chain_indices:
            for j in range(len(residue_chain_list)):
                all_pae_values.append(pae_array[i][j])
        
        return sum(all_pae_values) / len(all_pae_values) if all_pae_values else None
        
    except Exception as e:
        print(f"Error calculating peptide PAE: {e}")
        return None


def _parse_pdb_residues(pdb_file: str) -> List[Tuple[str, str]]:
    """
    Parse PDB file to build residue chain list.
    
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


def get_peptide_metrics_from_processed_dir(
    processed_dir: str, 
    method: str,
    peptide_chain: str = "B"
) -> Dict[str, Optional[float]]:
    """
    Get comprehensive peptide metrics from a processed directory.
    
    Args:
        processed_dir: Path to processed directory
        method: Method name ('alphafold' or 'boltz')
        peptide_chain: Chain ID of the peptide (default: "B")
        
    Returns:
        Dict with peptide metrics: plddt, pae, weighted_plddt_overall, weighted_plddt_residues
    """
    metrics = {
        "plddt": None,
        "pae": None, 
        "weighted_plddt_overall": None,
        "weighted_plddt_residues": None
    }
    
    try:
        # Get model file
        import glob
        model_pattern = os.path.join(processed_dir, f"{method}_model_1.pdb")
        model_files = glob.glob(model_pattern)
        if not model_files:
            return metrics
        
        model_file = model_files[0]
        
        # Calculate pLDDT from PDB
        metrics["plddt"] = get_peptide_plddt_from_pdb(model_file, peptide_chain)
        
        # Calculate weighted pLDDT
        weighted_overall, weighted_residues = get_weighted_peptide_plddt_from_pdb(
            model_file, peptide_chain
        )
        metrics["weighted_plddt_overall"] = weighted_overall
        metrics["weighted_plddt_residues"] = weighted_residues
        
        # Calculate PAE from JSON if available
        json_file = os.path.join(processed_dir, f"{method}_metrics.json")
        if os.path.exists(json_file):
            # For detailed PAE calculation, we need the raw data
            # This is more complex and might need method-specific logic
            pass  # TODO: Implement if needed
            
    except Exception as e:
        print(f"Error getting peptide metrics: {e}")
    
    return metrics
