"""
Metrics parser for standardizing docking method outputs.

This module provides a unified interface for parsing metrics from different
docking methods (AlphaFold, Boltz, etc.) and converting them into a standardized
format with method-prefixed keys.
"""

import os
import json
from typing import Dict, Any, List
import glob
from typing import Dict, Optional, Any
import logging


class MetricsParser:
    """
    Parser for standardizing metrics from different docking methods.
    
    Converts method-specific metric files into a unified format with
    prefixed keys (e.g., "alphafold_iptm", "boltz_confidence_score").
    """
    
    def __init__(self):
        self.supported_methods = ["alphafold", "boltz"]
    
    def parse_processed_dir(self, processed_dir: str) -> Dict[str, Any]:
        """
        Parse all available metrics files in a processed directory.
        
        Args:
            processed_dir: Path to processed directory containing method-specific metrics files
            
        Returns:
            Dict with flattened, prefixed metric keys:
            {
                "alphafold_iptm": 0.88,
                "alphafold_plddt": 0.85,
                "boltz_confidence_score": 0.75,
                "boltz_has_clash": False,
                ...
            }
        """
        all_metrics = {}
        
        # Parse each supported method
        for method in self.supported_methods:
            metrics_file = os.path.join(processed_dir, f"{method}_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    method_metrics = self._parse_method_metrics(metrics_file, method)
                    all_metrics.update(method_metrics)
                except Exception as e:
                    logging.warning(f"Failed to parse {method} metrics: {e}")
        
        return all_metrics
    
    def _parse_method_metrics(self, metrics_file: str, method: str) -> Dict[str, Any]:
        """
        Parse metrics file for a specific method and return prefixed metrics.
        
        Args:
            metrics_file: Path to the method-specific metrics JSON file
            method: Method name (alphafold, boltz, etc.)
            
        Returns:
            Dict with method-prefixed keys
        """
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        if method == "alphafold":
            return self._parse_alphafold_metrics(data)
        elif method == "boltz":
            return self._parse_boltz_metrics(data)
        else:
            logging.warning(f"Unknown method: {method}")
            return {}
    
    def get_available_methods(self, processed_dir: str) -> List[str]:
        """
        Get list of available docking methods in processed directory.
        
        Args:
            processed_dir: Path to processed directory
            
        Returns:
            List of available method names (e.g., ["alphafold", "boltz"])
        """
        available_methods = []
        
        for method in self.supported_methods:
            metrics_file = os.path.join(processed_dir, f"{method}_metrics.json")
            if os.path.exists(metrics_file):
                available_methods.append(method)
                
        return available_methods

    def _parse_alphafold_metrics(self, alphafold_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse AlphaFold-specific metrics and return with alphafold_ prefix.
        
        AlphaFold metrics structure:
        {
            "peptide_sequence": "NYLSELSEHV",
            "target_name": "4glf", 
            "model_count": 2,
            "models": {
                "alphafold_model_1": {
                    "pdb_file": "alphafold_model_1.pdb",
                    "original_rank": 1,
                    "relaxed": true,
                    "plddt": [...],
                    "pae": [...],
                    "iptm": 0.88,
                    "ptm": 0.95,
                    ...
                }
            },
            "best_model": "alphafold_model_1",
            "ranking": [...]
        }
        """
        metrics = {}
        
        # Basic metadata
        metrics["alphafold_model_count"] = alphafold_data.get("model_count")
        metrics["alphafold_best_model"] = alphafold_data.get("best_model")
        
        # Get best model metrics (typically first/best ranked model)
        models = alphafold_data.get("models", {})
        if models:
            # Get the best model (should be first in ranking or specified by best_model)
            best_model_key = alphafold_data.get("best_model")
            if best_model_key and best_model_key in models:
                best_model = models[best_model_key]
            else:
                # Fallback to first model
                best_model_key = list(models.keys())[0]
                best_model = models[best_model_key]
            
            # Extract key metrics from best model
            metrics["alphafold_iptm"] = best_model.get("iptm")
            metrics["alphafold_ptm"] = best_model.get("ptm") 
            metrics["alphafold_original_rank"] = best_model.get("original_rank")
            metrics["alphafold_relaxed"] = best_model.get("relaxed")
            
            # Calculate peptide pLDDT (average of chain B residues)
            plddt_values = best_model.get("plddt", [])
            if plddt_values:
                # For now, assume all plddt values are for peptide
                # In future, might need chain-specific logic
                metrics["alphafold_plddt"] = sum(plddt_values) / len(plddt_values)
            
            # Calculate peptide PAE if available
            pae_matrix = best_model.get("pae", [])
            if pae_matrix and isinstance(pae_matrix, list) and len(pae_matrix) > 0:
                # Calculate average PAE - this is a simplified approach
                # More sophisticated chain-specific calculation might be needed
                flat_pae = [val for row in pae_matrix for val in row if isinstance(row, list)]
                if flat_pae:
                    metrics["alphafold_pae"] = sum(flat_pae) / len(flat_pae)
            
            # Add weighted pLDDT metrics if available
            metrics["alphafold_weighted_plddt_overall"] = best_model.get("weighted_plddt_overall")
            metrics["alphafold_weighted_plddt_residues"] = best_model.get("weighted_plddt_residues")
            
            # Extract raw confidence data for confidence scoring functions
            raw_plddt = best_model.get("plddt", [])
            if raw_plddt:
                metrics["alphafold_plddt_vector"] = raw_plddt
            
            raw_pae = best_model.get("pae", [])
            if raw_pae:
                metrics["alphafold_pae_matrix"] = raw_pae
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def _parse_boltz_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Boltz-specific metrics and return with boltz_ prefix.
        
        Boltz metrics structure:
        {
            "peptide_sequence": "NYLSELSEHV",
            "docking_method": "boltz", 
            "models": [
                {
                    "model_id": 1,
                    "confidence_score": 0.75,
                    "ptm": 0.87,
                    "iptm": 0.36,
                    "complex_plddt": 0.85,
                    "complex_iplddt": 0.74,
                    "has_clash": false,
                    "fraction_plausible": 0.92,
                    ...
                }
            ],
            "best_model_id": 1,
            "best_confidence_score": 0.75,
            ...
        }
        """
        metrics = {}
        
        # Best model summary metrics (already computed in the JSON)
        metrics["boltz_best_model_id"] = data.get("best_model_id")
        metrics["boltz_confidence_score"] = data.get("best_confidence_score")
        metrics["boltz_ptm"] = data.get("best_ptm") 
        metrics["boltz_iptm"] = data.get("best_iptm")
        metrics["boltz_plddt"] = data.get("best_plddt")
        
        # Get detailed metrics from best model
        models = data.get("models", [])
        if models:
            # Find the best model (first one should be best due to sorting)
            best_model = models[0]
            
            # Additional metrics from best model
            metrics["boltz_complex_plddt"] = best_model.get("complex_plddt")
            metrics["boltz_complex_iplddt"] = best_model.get("complex_iplddt") 
            metrics["boltz_complex_pde"] = best_model.get("complex_pde")
            metrics["boltz_complex_ipde"] = best_model.get("complex_ipde")
            metrics["boltz_ligand_iptm"] = best_model.get("ligand_iptm")
            metrics["boltz_protein_iptm"] = best_model.get("protein_iptm")
            
            # Quality indicators
            metrics["boltz_has_clash"] = best_model.get("has_clash")
            metrics["boltz_fraction_plausible"] = best_model.get("fraction_plausible")
            
            # Chain-specific metrics if needed
            chains_ptm = best_model.get("chains_ptm", {})
            if chains_ptm:
                metrics["boltz_chain_0_ptm"] = chains_ptm.get("0")
                metrics["boltz_chain_1_ptm"] = chains_ptm.get("1")
        
        # Model count
        metrics["boltz_model_count"] = len(models)
        
        # Extract raw confidence data from top-level (extracted from NPZ files)
        if "plddt_vector" in data:
            metrics["boltz_plddt_vector"] = data["plddt_vector"]
        
        if "pae_matrix" in data:
            metrics["boltz_pae_matrix"] = data["pae_matrix"]
        
        if "pde_vector" in data:
            metrics["boltz_pde_vector"] = data["pde_vector"]
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def get_available_methods(self, processed_dir: str) -> list:
        """
        Get list of methods that have metrics files in the processed directory.
        
        Args:
            processed_dir: Path to processed directory
            
        Returns:
            List of method names that have metrics files
        """
        available_methods = []
        
        for method in self.supported_methods:
            metrics_file = os.path.join(processed_dir, f"{method}_metrics.json")
            if os.path.exists(metrics_file):
                available_methods.append(method)
        
        return available_methods
    
    def validate_processed_dir(self, processed_dir: str) -> Dict[str, bool]:
        """
        Validate that a processed directory contains expected files.
        
        Args:
            processed_dir: Path to processed directory
            
        Returns:
            Dict mapping method names to whether they have valid metrics files
        """
        validation = {}
        
        for method in self.supported_methods:
            metrics_file = os.path.join(processed_dir, f"{method}_metrics.json")
            is_valid = False
            
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    # Basic validation - check for required keys
                    if method == "alphafold":
                        is_valid = "models" in data and data["models"]
                    elif method == "boltz":
                        is_valid = "models" in data and data["models"]
                except Exception:
                    is_valid = False
            
            validation[method] = is_valid
        
        return validation


# Convenience function for quick parsing
def parse_metrics(processed_dir: str) -> Dict[str, Any]:
    """
    Convenience function to parse metrics from a processed directory.
    
    Args:
        processed_dir: Path to processed directory
        
    Returns:
        Dict with flattened, prefixed metric keys
    """
    parser = MetricsParser()
    return parser.parse_processed_dir(processed_dir)


if __name__ == "__main__":
    # Example usage
    test_dir = "/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV"
    
    parser = MetricsParser()
    
    print("Available methods:", parser.get_available_methods(test_dir))
    print("Validation:", parser.validate_processed_dir(test_dir))
    
    metrics = parser.parse_processed_dir(test_dir)
    print("\nParsed metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")
