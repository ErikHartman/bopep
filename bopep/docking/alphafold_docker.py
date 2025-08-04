import os
import shutil
import subprocess
import json
import glob
import re
from multiprocessing import get_context
from typing import List
from bopep.docking.base_docking_model import BaseDockingModel
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AlphaFoldDocker(BaseDockingModel):
    """
    AlphaFold2-based docking using ColabFold.
    
    This class handles docking peptides to target structures using ColabFold's
    AlphaFold2 implementation and processes the output into a standardized format.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize AlphaFold docker with specific parameters.
        
        Additional parameters beyond BaseDockingModel:
        - num_models: Number of models to generate (default: 5)
        - num_recycles: Number of recycling steps (default: 10)
        - recycle_early_stop_tolerance: Early stopping tolerance (default: 0.01)
        - amber: Whether to use AMBER relaxation (default: True)
        - num_relax: Number of relaxation steps (default: 1)
        """
        super().__init__(**kwargs)
        
        self.num_models = kwargs.get("num_models", 5)
        self.num_recycles = kwargs.get("num_recycles", 10)
        self.recycle_early_stop_tolerance = kwargs.get("recycle_early_stop_tolerance", 0.01)
        self.amber = kwargs.get("amber", True)
        self.num_relax = kwargs.get("num_relax", 1)
        
    def dock(self, peptide_sequences: List[str], target_structure: str, 
             target_sequence: str, target_name: str) -> List[str]:
        """
        Dock peptides using AlphaFold2/ColabFold.
        
        Returns list of processed output directories.
        """
        logging.info(f"Starting AlphaFold2 docking for {len(peptide_sequences)} peptides...")
        
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
            return previously_docked_dirs
        
        logging.info(f"Will dock {len(peptides_to_dock)} peptides...")
        
        # Perform docking to raw directory
        raw_docked_dirs = self._dock_peptides_parallel(
            peptides_to_dock, target_structure, target_sequence, target_name
        )
        
        # Process raw output to standardized format
        processed_dirs = []
        for raw_dir, peptide in zip(raw_docked_dirs, peptides_to_dock):
            if os.path.exists(raw_dir):
                processed_dir = self.process_raw_output(raw_dir, peptide, target_name)
                processed_dirs.append(processed_dir)
        
        # Combine with previously docked results
        all_processed_dirs = processed_dirs + previously_docked_dirs
        
        logging.info(f"Completed docking. {len(all_processed_dirs)} total results available.")
        return all_processed_dirs
    
    def process_raw_output(self, raw_peptide_dir: str, peptide_sequence: str, 
                          target_name: str) -> str:
        """
        Process ColabFold raw output into standardized format.
        
        ColabFold typically produces:
        - *_relaxed_rank_00X_*.pdb files (relaxed models)
        - *_unrelaxed_rank_00X_*.pdb files (unrelaxed models)
        - *_scores_rank_00X_*.json files (metrics)
        - Other files (PAE plots, etc.)
        """
        logging.info(f"Processing raw output for {peptide_sequence}...")
        
        # Create processed directory
        processed_dir = self._create_processed_peptide_dir(target_name, peptide_sequence)
        
        # Find all model files and their corresponding metrics
        relaxed_pdbs = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_relaxed_rank_*.pdb")))
        score_jsons = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_scores_rank_*.json")))
        pae_json = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_predicted_aligned_error_*.json")))
        
        # Process models and collect all metrics
        all_metrics = {
            "peptide_sequence": peptide_sequence,
            "target_name": target_name,
            "model_count": len(relaxed_pdbs),
            "models": {}
        }
        
        # Copy model files with standardized names
        for i, pdb_file in enumerate(relaxed_pdbs, 1):
            # Copy PDB file with standardized name
            standardized_filename = self._standardize_model_filename(pdb_file, i)
            dest_path = os.path.join(processed_dir, standardized_filename)
            shutil.copy2(pdb_file, dest_path)
            
            # Find corresponding metrics file
            rank_match = re.search(r'rank_(\d+)', os.path.basename(pdb_file))
            if rank_match:
                rank_num = rank_match.group(1)
                corresponding_json = None
                for json_file in score_jsons:
                    if f"rank_{rank_num}" in os.path.basename(json_file):
                        corresponding_json = json_file
                        break
                
                # Load metrics for this model
                model_metrics = {"pdb_file": standardized_filename}
                if corresponding_json and os.path.exists(corresponding_json):
                    with open(corresponding_json, 'r') as f:
                        json_data = json.load(f)
                        model_metrics.update(json_data)
                
                all_metrics["models"][f"model_{i}"] = model_metrics
        
        # Add overall best metrics (typically from rank_001)
        if score_jsons:
            best_json = score_jsons[0]  # Assuming first is best ranked
            with open(best_json, 'r') as f:
                best_metrics = json.load(f)
                all_metrics["best_model_metrics"] = best_metrics
        
        if pae_json:
            with open(pae_json[0], 'r') as f:
                pae_data = json.load(f)
                all_metrics["predicted_aligned_error"] = pae_data
        
        # Save metrics
        self._save_metrics_json(all_metrics, processed_dir)
        
        return processed_dir
    
    def _dock_peptides_parallel(self, peptides: List[str], target_structure: str, 
                               target_sequence: str, target_name: str) -> List[str]:
        """
        Dock multiple peptides in parallel using ColabFold.
        """
        num_processes = min(len(self.gpu_ids), len(peptides))
        
        if num_processes <= 1:
            # Sequential processing
            return [self._dock_single_peptide(peptide, self.gpu_ids[0], target_structure, 
                                            target_sequence, target_name) 
                   for peptide in peptides]
        
        # Parallel processing
        logging.info(f"Starting parallel docking on {num_processes} GPUs...")
        
        # Group peptides by GPU
        peptides_by_gpu = [[] for _ in range(len(self.gpu_ids))]
        for i, peptide in enumerate(peptides):
            gpu_index = i % len(self.gpu_ids)
            peptides_by_gpu[gpu_index].append(peptide)
        
        # Create arguments for each worker process
        process_args = []
        for gpu_index, gpu_peptides in enumerate(peptides_by_gpu[:num_processes]):
            if gpu_peptides:
                process_args.append((
                    gpu_peptides, self.gpu_ids[gpu_index], target_structure,
                    target_sequence, target_name, self.raw_output_dir,
                    self.num_models, self.num_recycles, self.recycle_early_stop_tolerance,
                    self.amber, self.num_relax
                ))
        
        # Run in parallel
        context = get_context("spawn")
        with context.Pool(processes=num_processes) as pool:
            all_docked_dirs = pool.starmap(self._dock_peptides_for_gpu, process_args)
        
        # Flatten results
        return [dir_path for dirs in all_docked_dirs for dir_path in dirs]
    
    def _dock_single_peptide(self, peptide_sequence: str, gpu_id: str, 
                           target_structure: str, target_sequence: str, 
                           target_name: str) -> str:
        """
        Dock a single peptide using ColabFold.
        """
        logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")
        
        # Create output directory in raw folder
        peptide_output_dir = os.path.join(
            self.raw_output_dir, f"{target_name}_{peptide_sequence}"
        )
        os.makedirs(peptide_output_dir, exist_ok=True)
        
        # Create FASTA file
        combined_fasta_path = os.path.join(
            peptide_output_dir, f"input_{peptide_sequence}.fasta"
        )
        with open(combined_fasta_path, "w") as f:
            f.write(f">{target_name}_{peptide_sequence}\n{target_sequence}:{peptide_sequence}\n")
        
        # Copy target structure
        target_copy_path = os.path.join(
            peptide_output_dir, os.path.basename(target_structure)
        )
        shutil.copy2(target_structure, target_copy_path)
        
        # Prepare ColabFold command
        command = [
            "colabfold_batch",
            str(combined_fasta_path),
            str(peptide_output_dir),
            "--model-type", "alphafold2_multimer_v3",
            "--msa-mode", "single_sequence",
            "--num-models", str(self.num_models),
            "--num-recycle", str(self.num_recycles),
            "--recycle-early-stop-tolerance", str(self.recycle_early_stop_tolerance),
            "--num-relax", str(self.num_relax),
            "--pair-mode", "unpaired",
            "--pair-strategy", "greedy",
            "--templates",
            "--custom-template-path", str(peptide_output_dir),
            "--rank", "iptm",
        ]
        
        if self.amber:
            command.append("--amber")
        
        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.pop("MPLBACKEND", None)
        
        # Run docking
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=command
                )
            
            logging.info(f"Docking completed successfully for {peptide_sequence} on GPU {gpu_id}.")
            
            # Clean up some unnecessary files but keep all model outputs
            self._clean_up_colabfold_files(peptide_output_dir, target_copy_path)
            
            # Mark as finished
            with open(os.path.join(peptide_output_dir, "finished.txt"), "w") as f:
                f.write("Docking finished successfully.")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred during docking of {peptide_sequence}: {e}")
            return None
        
        return peptide_output_dir
    
    def _clean_up_colabfold_files(self, docking_dir: str, target_structure_copy: str):
        """
        Clean up unnecessary ColabFold files while preserving model outputs.
        """
        try:
            # Remove copied PDB file
            if os.path.exists(target_structure_copy):
                os.remove(target_structure_copy)
            
            # Remove some unnecessary files but keep all model-related outputs
            for file in os.listdir(docking_dir):
                if (file.startswith("pdb70") or 
                    file == "cite.bibtex" or 
                    file.startswith("combined_input")):
                    file_path = os.path.join(docking_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
        except OSError as e:
            logging.warning(f"Error cleaning up files: {e}")
    
    @staticmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              num_models: int, num_recycles: int, 
                              recycle_early_stop_tolerance: float, amber: bool, 
                              num_relax: int) -> List[str]:
        """
        Static method for parallel processing of peptides on a specific GPU.
        """
        # Create a temporary AlphaFoldDocker instance for this process
        docker_kwargs = {
            "output_dir": os.path.dirname(raw_output_dir),  # Parent of raw_output_dir
            "gpu_ids": [gpu_id],
            "num_models": num_models,
            "num_recycles": num_recycles,
            "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
            "amber": amber,
            "num_relax": num_relax,
        }
        
        temp_docker = AlphaFoldDocker(**docker_kwargs)
        
        docked_dirs = []
        for peptide in peptides:
            dir_path = temp_docker._dock_single_peptide(
                peptide, gpu_id, target_structure, target_sequence, target_name
            )
            if dir_path:
                docked_dirs.append(dir_path)
        
        return docked_dirs
