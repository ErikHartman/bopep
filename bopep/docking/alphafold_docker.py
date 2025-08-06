import os
import shutil
import subprocess
import json
import glob
import re
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
        self.method_name = "alphafold"
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
        return self._dock_with_common_logic(peptide_sequences, target_structure, 
                                          target_sequence, target_name)
    
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
        unrelaxed_pdbs = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_unrelaxed_rank_*.pdb")))
        score_jsons = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_scores_rank_*.json")))
        pae_json = sorted(glob.glob(os.path.join(raw_peptide_dir, "*_predicted_aligned_error_*.json")))
        
        logging.info(f"Found {len(relaxed_pdbs)} relaxed and {len(unrelaxed_pdbs)} unrelaxed PDB files")
        
        # Process models and collect metrics for best model only
        all_models_data = []
        
        # Strategy: Use relaxed for rank 1 (if available), unrelaxed for all others
        # First, collect all rank numbers from available files
        available_ranks = set()
        for pdb_file in relaxed_pdbs + unrelaxed_pdbs:
            rank_match = re.search(r'rank_(\d+)', os.path.basename(pdb_file))
            if rank_match:
                available_ranks.add(int(rank_match.group(1)))
        
        processed_count = 0
        for rank_num in sorted(available_ranks):
            rank_str = f"{rank_num:03d}"  # Format as 001, 002, etc.
            
            # For rank 1, prefer relaxed if available, otherwise use unrelaxed
            # For all other ranks, use unrelaxed
            selected_pdb = None
            if rank_num == 1:
                # Look for relaxed version first for rank 1
                relaxed_candidates = [f for f in relaxed_pdbs if f"rank_{rank_str}" in f]
                if relaxed_candidates:
                    selected_pdb = relaxed_candidates[0]
                else:
                    # Fall back to unrelaxed for rank 1 if no relaxed available
                    unrelaxed_candidates = [f for f in unrelaxed_pdbs if f"rank_{rank_str}" in f]
                    if unrelaxed_candidates:
                        selected_pdb = unrelaxed_candidates[0]
            else:
                # For ranks 2+, use unrelaxed
                unrelaxed_candidates = [f for f in unrelaxed_pdbs if f"rank_{rank_str}" in f]
                if unrelaxed_candidates:
                    selected_pdb = unrelaxed_candidates[0]
            
            if selected_pdb:
                processed_count += 1
                model_type = "relaxed" if "relaxed" in os.path.basename(selected_pdb) else "unrelaxed"
                
                # Copy PDB file with standardized name
                standardized_filename = self._standardize_model_filename(selected_pdb, processed_count)
                dest_path = os.path.join(processed_dir, standardized_filename)
                shutil.copy2(selected_pdb, dest_path)
                
                corresponding_json = None
                for json_file in score_jsons:
                    if f"rank_{rank_str}" in os.path.basename(json_file):
                        corresponding_json = json_file
                        break
                
                # Load metrics for this model
                model_metrics = {
                    "pdb_file": standardized_filename,
                    "original_rank": rank_num,
                    "relaxed": model_type == "relaxed"
                }
                if corresponding_json and os.path.exists(corresponding_json):
                    with open(corresponding_json, 'r') as f:
                        json_data = json.load(f)
                        model_metrics.update(json_data)
                
                all_models_data.append(model_metrics)
        
        # Find best model based on ipTM (should be rank 1 since ColabFold ranks by ipTM)
        best_model = max(all_models_data, key=lambda x: x.get("iptm", 0))
        best_model_rank = best_model.get("original_rank", 1)
        
        logging.info(f"Best model based on ipTM: rank_{best_model_rank:03d} (ipTM: {best_model.get('iptm', 'N/A')})")
        
        # Create metrics with only the best model's data
        all_metrics = {
            "peptide_sequence": peptide_sequence,
            "target_name": target_name,
            "docking_method": "alphafold",
            "model_count": processed_count,
            "best_model_rank": best_model_rank,
            # Store all metrics from best model at root level
            **{k: v for k, v in best_model.items() if k not in ["pdb_file"]}
            
        }
        
        if pae_json:
            with open(pae_json[0], 'r') as f:
                pae_data = json.load(f)
                all_metrics["predicted_aligned_error"] = pae_data["predicted_aligned_error"] # so not nested
        
        self._save_metrics_json(all_metrics, processed_dir, prefix="alphafold_metrics")
        
        return processed_dir
    
    def _dock_single_peptide(self, peptide_sequence: str, target_structure: str,
                           target_sequence: str, target_name: str, gpu_id: str = "0") -> str:
        """
        Dock a single peptide using ColabFold.
        """
        logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")
        
        peptide_output_dir = self._create_raw_peptide_dir(target_name, peptide_sequence)
        
        combined_fasta_path = os.path.join(
            peptide_output_dir, f"input_{peptide_sequence}.fasta"
        )
        with open(combined_fasta_path, "w") as f:
            f.write(f">{target_name}_{peptide_sequence}\n{target_sequence}:{peptide_sequence}\n")
        
        target_copy_path = os.path.join(
            peptide_output_dir, os.path.basename(target_structure)
        )
        shutil.copy2(target_structure, target_copy_path)
        
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
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.pop("MPLBACKEND", None)
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            output, _ = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=command, output=output
                )
            
            logging.info(f"Docking completed successfully for {peptide_sequence} on GPU {gpu_id}.")
            
            # Clean up some unnecessary files but keep all model outputs
            self._clean_up_colabfold_files(peptide_output_dir, target_copy_path)
            
            # Mark as finished
            with open(os.path.join(peptide_output_dir, "finished.txt"), "w") as f:
                f.write("Docking finished successfully.")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred during docking of {peptide_sequence}: {e}")
            logging.error(f"ColabFold output:\n{e.output}")
            return None
        
        return peptide_output_dir
    
    def _get_method_parameters(self) -> dict:
        """
        Get AlphaFold-specific parameters for parallel processing.
        """
        return {
            'num_models': self.num_models,
            'num_recycles': self.num_recycles,
            'recycle_early_stop_tolerance': self.recycle_early_stop_tolerance,
            'amber': self.amber,
            'num_relax': self.num_relax
        }
    
    @staticmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              method_params: dict) -> List[str]:
        """
        Process a batch of peptides on a specific GPU using AlphaFold.
        """
        # Create a temporary docker instance for this process
        temp_docker = AlphaFoldDocker(
            output_dir=os.path.dirname(raw_output_dir),  # Parent of raw_output_dir
            gpu_ids=[gpu_id],
            **method_params
        )
        
        docked_dirs = []
        for peptide in peptides:

            dir_path = temp_docker._dock_single_peptide(
                peptide, target_structure, target_sequence, target_name, gpu_id
            )
            if dir_path:
                docked_dirs.append(dir_path)

        
        return docked_dirs
    
    def _clean_up_colabfold_files(self, docking_dir: str, target_structure_copy: str):
        """
        Clean up unnecessary ColabFold files while preserving model outputs.
        """

        # Remove copied PDB file
        if os.path.exists(target_structure_copy):
            os.remove(target_structure_copy)
        
        # Remove some unnecessary files but keep all model-related outputs
        for file in os.listdir(docking_dir):
            if (file.startswith("pdb70") or 
                file == "cite.bibtex" or 
                file.endswith(".cif.bak") or
                file.endswith(".png") or
                file.startswith("combined_input")):
                file_path = os.path.join(docking_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
