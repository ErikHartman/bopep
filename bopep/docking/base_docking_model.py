from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os
import json
import logging
from multiprocessing import get_context

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
        """
        self.output_dir = kwargs["output_dir"]
        self.gpu_ids = kwargs.get("gpu_ids", ["0"])
        self.overwrite_results = kwargs.get("overwrite_results", False)
        
        # Method name should be set by subclasses
        method_name = getattr(self, 'method_name', 'unknown')
        
        # Create method-specific raw output directory but shared processed directory
        self.raw_output_dir = os.path.join(self.output_dir, "raw", method_name)
        self.processed_output_dir = os.path.join(self.output_dir, "processed")  # Shared processed directory
        os.makedirs(self.raw_output_dir, exist_ok=True)
        os.makedirs(self.processed_output_dir, exist_ok=True)
    
    @abstractmethod
    def dock(self, peptide_sequences: List[str], target_structure: str, 
             target_sequence: str, target_name: str) -> List[str]:

        pass
    
    def _dock_with_common_logic(self, peptide_sequences: List[str], target_structure: str, 
                               target_sequence: str, target_name: str) -> List[str]:
        """
        Common docking logic that can be reused by subclasses.
        
        This method handles:
        1. Checking for existing results
        2. Filtering peptides that need docking
        3. Choosing between parallel and sequential processing
        4. Processing raw output to standardized format
        """
        method_name = getattr(self, 'method_name', 'unknown')
        logging.info(f"Starting {method_name} docking for {len(peptide_sequences)} peptides...")
        
        # Check for existing results
        previously_docked_dirs = []
        peptides_to_dock = []
        
        for peptide in peptide_sequences:
            exists, processed_dir = self.check_existing_results(peptide, target_name)
            if exists and not self.overwrite_results:
                previously_docked_dirs.append(processed_dir)
                logging.info(f"Found existing results for {peptide}, skipping docking...")
            else:
                peptides_to_dock.append(peptide)
        
        if not peptides_to_dock:
            logging.info("All peptides already docked. Returning existing results.")
            return previously_docked_dirs
        
        logging.info(f"Will dock {len(peptides_to_dock)} peptides...")
        
        # Choose between parallel and sequential processing
        if len(self.gpu_ids) > 1 and len(peptides_to_dock) > 1:
            raw_docked_dirs = self._dock_parallel(peptides_to_dock, target_structure, 
                                                target_sequence, target_name)
        else:
            raw_docked_dirs = []
            # Use the first GPU for sequential processing
            gpu_id = self.gpu_ids[0] if self.gpu_ids else "0"
            for peptide in peptides_to_dock:
                raw_dir = self._dock_single_peptide(peptide, target_structure, 
                                                  target_sequence, target_name, gpu_id)
                raw_docked_dirs.append(raw_dir)
        
        # Process raw output to standardized format
        processed_dirs = []
        for raw_dir, peptide in zip(raw_docked_dirs, peptides_to_dock):
            if os.path.exists(raw_dir):
                processed_dir = self.process_raw_output(raw_dir, peptide, target_name)
                processed_dirs.append(processed_dir)
        
        # Combine with previously docked results
        all_processed_dirs = processed_dirs + previously_docked_dirs
        
        logging.info(f"Completed docking for {len(peptides_to_dock)} peptides. "
                    f"Total results: {len(all_processed_dirs)}")
        
        return all_processed_dirs
        
    @abstractmethod
    def _dock_single_peptide(self, peptide_sequence: str, target_structure: str,
                           target_sequence: str, target_name: str, gpu_id: str = "0") -> str:

        pass
    
    def _dock_parallel(self, peptide_sequences: List[str], target_structure: str,
                      target_sequence: str, target_name: str) -> List[str]:
        """
        Common parallel docking logic for multiple GPUs.
        
        This method groups peptides by GPU and processes them in parallel.
        Subclasses should implement _dock_peptides_for_gpu as a static method.
        """
        logging.info(f"Starting parallel docking on {len(self.gpu_ids)} GPUs...")
        
        # Group peptides by GPU
        peptides_by_gpu = [[] for _ in range(len(self.gpu_ids))]
        for i, peptide in enumerate(peptide_sequences):
            gpu_index = i % len(self.gpu_ids)
            peptides_by_gpu[gpu_index].append(peptide)
        
        # Create arguments for each worker process
        process_args = []
        for gpu_index, gpu_peptides in enumerate(peptides_by_gpu):
            if not gpu_peptides:
                continue
            
            # Get method-specific parameters
            method_params = self._get_method_parameters()
            
            process_args.append((
                gpu_peptides,
                self.gpu_ids[gpu_index],
                target_structure,
                target_sequence,
                target_name,
                self.raw_output_dir,
                method_params
            ))
        
        # Run in parallel
        # Use fork on Unix-like systems (faster and doesn't require __main__ guard)
        # Use spawn on Windows or as fallback
        try:
            context = get_context("fork")
        except RuntimeError:
            # fork not available on this platform (e.g., Windows)
            context = get_context("spawn")
        
        with context.Pool(processes=len(process_args)) as pool:
            all_docked_dirs = pool.starmap(self._dock_peptides_for_gpu, process_args)
        
        # Flatten the list of lists
        return [dir_path for dirs in all_docked_dirs for dir_path in dirs]
    
    @abstractmethod
    def _get_method_parameters(self) -> dict:
        pass
    
    @staticmethod
    @abstractmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              method_params: dict) -> List[str]:
        pass
    
    def _create_raw_peptide_dir(self, target_name: str, peptide_sequence: str) -> str:
        """
        Create standardized directory name for raw output.
        
        Format: TARGETNAME_PEPTIDESEQ
        """
        peptide_dir_name = f"{target_name}_{peptide_sequence}"
        peptide_dir_path = os.path.join(self.raw_output_dir, peptide_dir_name)
        os.makedirs(peptide_dir_path, exist_ok=True)
        return peptide_dir_path
    
    @abstractmethod
    def process_raw_output(self, raw_peptide_dir: str, peptide_sequence: str, 
                          target_name: str) -> str:
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
    
    def _save_metrics_json(self, metrics: Dict[str, Any], output_dir: str, prefix: str = "metrics") -> None:
        """
        Save metrics dictionary to metrics.json file.
    
        """
        metrics_path = os.path.join(output_dir, f"{prefix}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
    
    def _standardize_model_filename(self, original_path: str, model_index: int) -> str:
        """
        Create standardized model filename with method prefix.
        
        Returns:
        - Standardized filename (e.g., "alphafold_model_1.pdb", "boltz_model_1.cif")
        """
        # Determine file extension
        if original_path.endswith('.pdb'):
            extension = '.pdb'
        elif original_path.endswith('.cif'):
            extension = '.cif'
        else:
            # Default to original extension
            extension = os.path.splitext(original_path)[1]
        
        # Get method name (should be overridden by subclasses)
        method_name = getattr(self, 'method_name', 'model')
        return f"{method_name}_model_{model_index}{extension}"
    
    def check_existing_results(self, peptide_sequence: str, target_name: str) -> tuple:
        peptide_dir_name = f"{target_name}_{peptide_sequence}"
        processed_dir = os.path.join(self.processed_output_dir, peptide_dir_name)
        
        if os.path.exists(processed_dir):
            # Check for method-specific metrics file
            method_name = getattr(self, 'method_name', 'unknown')
            method_metrics_file = os.path.join(processed_dir, f"{method_name}_metrics.json")
            
            if os.path.exists(method_metrics_file):
                return True, processed_dir
        
        return False, None
