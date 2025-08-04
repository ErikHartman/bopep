from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BaseDockingModel(ABC):
    """
    Abstract base class for docking models.
    
    This class defines the interface that all docking models must implement
    to ensure consistent output structure across different docking approaches.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the docking model with common parameters.
        
        Parameters:
        - output_dir: Base output directory for docking results
        - gpu_ids: List of GPU IDs to use for docking
        - overwrite_results: Whether to overwrite existing results
        """
        self.output_dir = kwargs["output_dir"]
        self.gpu_ids = kwargs.get("gpu_ids", ["0"])
        self.overwrite_results = kwargs.get("overwrite_results", False)
        
        # Create output directory structure
        self.raw_output_dir = os.path.join(self.output_dir, "raw")
        self.processed_output_dir = os.path.join(self.output_dir, "processed")
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.processed_output_dir, exist_ok=True)
    
    @abstractmethod
    def dock(self, peptide_sequences: List[str], target_structure: str, 
             target_sequence: str, target_name: str) -> List[str]:
        """
        Dock peptides to the target structure.
        
        This method should:
        1. Perform the actual docking using the model-specific approach
        2. Save raw output to self.raw_output_dir
        3. Process raw output and save to self.processed_output_dir
        4. Return list of processed output directories
        
        Parameters:
        - peptide_sequences: List of peptide sequences to dock
        - target_structure: Path to target PDB file
        - target_sequence: Target protein sequence
        - target_name: Name of the target protein
        
        Returns:
        - List of processed output directories for each peptide
        """
        pass
    
    @abstractmethod
    def process_raw_output(self, raw_peptide_dir: str, peptide_sequence: str, 
                          target_name: str) -> str:
        """
        Process raw docking output into standardized format.
        
        This method should:
        1. Extract model files (PDB/CIF) from raw output
        2. Extract all available metrics
        3. Save standardized files to processed directory
        4. Create metrics.json with all model scores
        
        Parameters:
        - raw_peptide_dir: Directory containing raw docking output for one peptide
        - peptide_sequence: The peptide sequence that was docked
        - target_name: Name of the target protein
        
        Returns:
        - Path to the processed output directory for this peptide
        """
        pass
    
    def _create_processed_peptide_dir(self, target_name: str, peptide_sequence: str) -> str:
        """
        Create standardized directory name for processed output.
        
        Format: TARGETNAME_PEPTIDESEQ
        """
        peptide_dir_name = f"{target_name}_{peptide_sequence}"
        peptide_dir_path = os.path.join(self.processed_output_dir, peptide_dir_name)
        os.makedirs(peptide_dir_path, exist_ok=True)
        return peptide_dir_path
    
    def _save_metrics_json(self, metrics: Dict[str, Any], output_dir: str):
        """
        Save metrics dictionary to metrics.json file.
        
        Parameters:
        - metrics: Dictionary containing all model metrics
        - output_dir: Directory to save the metrics.json file
        """
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
    
    def _standardize_model_filename(self, original_path: str, model_index: int) -> str:
        """
        Create standardized model filename.
        
        Parameters:
        - original_path: Original model file path
        - model_index: Index of the model (1-based)
        
        Returns:
        - Standardized filename (e.g., "model_1.pdb", "model_2.cif")
        """
        # Determine file extension
        if original_path.endswith('.pdb'):
            extension = '.pdb'
        elif original_path.endswith('.cif'):
            extension = '.cif'
        else:
            # Default to original extension
            extension = os.path.splitext(original_path)[1]
        
        return f"model_{model_index}{extension}"
    
    def check_existing_results(self, peptide_sequence: str, target_name: str) -> tuple:
        """
        Check if processed results already exist for a peptide.
        
        Parameters:
        - peptide_sequence: The peptide sequence
        - target_name: Name of the target protein
        
        Returns:
        - (exists: bool, processed_dir: str or None)
        """
        peptide_dir_name = f"{target_name}_{peptide_sequence}"
        processed_dir = os.path.join(self.processed_output_dir, peptide_dir_name)
        
        if os.path.exists(processed_dir) and os.path.exists(os.path.join(processed_dir, "metrics.json")):
            return True, processed_dir
        return False, None
