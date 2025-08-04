import os
import json
import yaml
import subprocess
import glob
import shutil
from typing import List, Dict, Any
from bopep.docking.base_docking_model import BaseDockingModel
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BoltzDocker(BaseDockingModel):
    """
    Boltz-based docking for biomolecular interactions.
    
    This class handles docking peptides to target structures using Boltz
    and processes the output into a standardized format.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Boltz docker with specific parameters.
        
        Additional parameters beyond BaseDockingModel:
        - recycling_steps: Number of recycling steps (default: 3)
        - diffusion_samples: Number of diffusion samples (default: 1)
        - output_format: Output format - 'pdb' or 'mmcif' (default: 'pdb')
        - sampling_steps: Number of sampling steps (default: 200)
        - step_scale: Step scale for diffusion sampling (default: 1.638)
        """
        super().__init__(**kwargs)
        
        self.recycling_steps = kwargs.get("recycling_steps", 3)
        self.diffusion_samples = kwargs.get("diffusion_samples", 1)
        self.output_format = kwargs.get("output_format", "pdb")
        self.sampling_steps = kwargs.get("sampling_steps", 200)
        self.step_scale = kwargs.get("step_scale", 1.638)
        
    def dock(self, peptide_sequences: List[str], target_structure: str, 
             target_sequence: str, target_name: str) -> List[str]:
        """
        Dock peptides using Boltz.
        
        Returns list of processed output directories.
        """
        logging.info(f"Starting Boltz docking for {len(peptide_sequences)} peptides...")
        
        # Filter peptides that already have results (if not overwriting)
        peptides_to_dock = []
        already_docked = []
        
        for peptide in peptide_sequences:
            processed_dir = os.path.join(self.processed_dir, f"{target_name}_{peptide}")
            if os.path.exists(processed_dir) and not self.overwrite_results:
                already_docked.append(processed_dir)
                logging.info(f"Skipping {peptide} - already docked")
            else:
                peptides_to_dock.append(peptide)
        
        if not peptides_to_dock:
            logging.info("All peptides already docked. Returning existing results.")
            return already_docked
        
        # Perform docking for peptides that need it
        if len(self.gpu_ids) > 1 and len(peptides_to_dock) > 1:
            docked_dirs = self._dock_parallel(peptides_to_dock, target_structure, 
                                            target_sequence, target_name)
        else:
            docked_dirs = []
            for peptide in peptides_to_dock:
                docked_dir = self._dock_single_peptide(peptide, target_structure, 
                                                     target_sequence, target_name)
                docked_dirs.append(docked_dir)
        
        # Process all docked results
        all_processed_dirs = already_docked + docked_dirs
        
        logging.info(f"Completed docking for {len(peptides_to_dock)} peptides. "
                    f"Total results: {len(all_processed_dirs)}")
        
        return all_processed_dirs
    
    def _dock_single_peptide(self, peptide_sequence: str, target_structure: str,
                            target_sequence: str, target_name: str, gpu_id: str = "0") -> str:
        """
        Dock a single peptide using Boltz.
        """
        logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")
        
        # Create output directories
        raw_peptide_dir = os.path.join(self.raw_dir, f"{target_name}_{peptide_sequence}")
        processed_peptide_dir = os.path.join(self.processed_dir, f"{target_name}_{peptide_sequence}")
        
        os.makedirs(raw_peptide_dir, exist_ok=True)
        os.makedirs(processed_peptide_dir, exist_ok=True)
        
        # Create YAML configuration file
        yaml_path = self._create_yaml_config(peptide_sequence, target_sequence, 
                                           target_name, raw_peptide_dir, target_structure)
        
        # Run Boltz prediction
        try:
            self._run_boltz_prediction(yaml_path, raw_peptide_dir, gpu_id)
            
            # Process the raw output into standardized format
            self._process_boltz_output(raw_peptide_dir, processed_peptide_dir, 
                                     peptide_sequence, target_name)
            
            logging.info(f"Successfully completed docking for {peptide_sequence}")
            
        except Exception as e:
            logging.error(f"Failed to dock {peptide_sequence}: {e}")
            raise
        
        return processed_peptide_dir
    
    def _create_yaml_config(self, peptide_sequence: str, target_sequence: str,
                           target_name: str, output_dir: str, target_structure: str = None) -> str:
        """
        Create a YAML configuration file for Boltz.
        """
        config = {
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": target_sequence
                    }
                },
                {
                    "protein": {
                        "id": "B", 
                        "sequence": peptide_sequence
                    }
                }
            ]
        }
        

        config["sequences"][0]["protein"]["msa"] = "empty"
        config["sequences"][1]["protein"]["msa"] = "empty"
        
        # Add template configuration if target structure is provided
        if target_structure and os.path.exists(target_structure):

            template_path = self._prepare_template_file(target_structure, output_dir)
            config["templates"] = [
                {
                    "cif": template_path,
                    "chain_id": "A"  # Specify which chain to template (target protein)
                }
            ]
            logging.info(f"Added template: {template_path}")
        
        yaml_path = os.path.join(output_dir, f"{target_name}_{peptide_sequence}.yaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logging.info(f"Created YAML config: {yaml_path}")
        return yaml_path
    
    def _prepare_template_file(self, target_structure: str, output_dir: str) -> str:
        """
        Prepare template file for Boltz.
        """
        # Check if the file is already in CIF format
        if target_structure.lower().endswith('.cif'):
            # Copy CIF file to output directory
            template_name = os.path.basename(target_structure)
            template_path = os.path.join(output_dir, template_name)
            shutil.copy2(target_structure, template_path)
            return template_path
        
        elif target_structure.lower().endswith('.pdb'):
            raise ValueError(
                "PDB files are not supported directly. Please convert to CIF format e.g using https://mmcif.pdbj.org/converter/index.php?l=en"
            )
    
    def _run_boltz_prediction(self, yaml_path: str, output_dir: str, gpu_id: str = "0"):
        """
        Run Boltz prediction using the command line interface.
        """
        cmd = [
            "boltz", "predict",
            yaml_path,
            "--out_dir", output_dir,
            "--recycling_steps", str(self.recycling_steps),
            "--diffusion_samples", str(self.diffusion_samples),
            "--output_format", self.output_format,
            "--sampling_steps", str(self.sampling_steps),
            "--step_scale", str(self.step_scale)
        ]
        
        
        if self.overwrite_results:
            cmd.append("--override")
        
        # Set environment to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.pop("MPLBACKEND", None)
        
        logging.info(f"Running Boltz command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logging.info(f"Boltz: {line.strip()}")
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=cmd
                )
                
            logging.info("Boltz prediction completed successfully")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Boltz prediction failed: {e}")
            raise
    
    def _process_boltz_output(self, raw_dir: str, processed_dir: str,
                             peptide_sequence: str, target_name: str):
        """
        Process Boltz raw output into standardized format.
        """
        logging.info("Processing Boltz output into standardized format...")
        
        # Find the prediction directory
        predictions_dir = os.path.join(raw_dir, "predictions")
        
        if not os.path.exists(predictions_dir):
            raise ValueError(f"Predictions directory not found: {predictions_dir}")
        
        # Find the input-specific subdirectory
        input_dirs = [d for d in os.listdir(predictions_dir) 
                     if os.path.isdir(os.path.join(predictions_dir, d))]
        
        if not input_dirs:
            raise ValueError(f"No input directories found in {predictions_dir}")
        
        # Use the first (and typically only) input directory
        input_dir = os.path.join(predictions_dir, input_dirs[0])
        
        # Find structure files
        if self.output_format == "pdb":
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.pdb"))
        else:  # mmcif
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.cif"))
        
        structure_files.sort()  # Sort to maintain consistent ordering
        
        # Copy and rename structure files
        for i, struct_file in enumerate(structure_files, 1):
            if self.output_format == "pdb":
                dest_file = os.path.join(processed_dir, f"model_{i}.pdb")
            else:
                dest_file = os.path.join(processed_dir, f"model_{i}.cif")
            
            shutil.copy2(struct_file, dest_file)
            logging.info(f"Copied {os.path.basename(struct_file)} -> {os.path.basename(dest_file)}")
        
        # Process and combine all metrics into a single JSON file
        metrics = self._extract_boltz_metrics(input_dir, peptide_sequence)
        
        metrics_file = os.path.join(processed_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Created standardized metrics file: {metrics_file}")
    
    def _extract_boltz_metrics(self, input_dir: str, peptide_sequence: str) -> Dict[str, Any]:
        """
        Extract all available metrics from Boltz output files.
        """
        metrics = {
            "peptide_sequence": peptide_sequence,
            "docking_method": "boltz",
            "models": []
        }
        
        # Find all confidence files (one per model)
        confidence_files = glob.glob(os.path.join(input_dir, "confidence_*.json"))
        confidence_files.sort()
        
        for i, conf_file in enumerate(confidence_files, 1):
            try:
                with open(conf_file, 'r') as f:
                    confidence_data = json.load(f)
                
                model_metrics = {
                    "model_id": i,
                    "confidence_file": os.path.basename(conf_file),
                    **confidence_data
                }
                
                metrics["models"].append(model_metrics)
                
            except Exception as e:
                logging.warning(f"Failed to read confidence file {conf_file}: {e}")
        
        # Add affinity data if available
        affinity_files = glob.glob(os.path.join(input_dir, "affinity_*.json"))
        if affinity_files:
            try:
                with open(affinity_files[0], 'r') as f:
                    affinity_data = json.load(f)
                
                metrics["affinity"] = affinity_data
                
            except Exception as e:
                logging.warning(f"Failed to read affinity file: {e}")
        
        # Add PAE data if available
        pae_files = glob.glob(os.path.join(input_dir, "pae_*.npz"))
        if pae_files:
            metrics["pae_files"] = [os.path.basename(f) for f in pae_files]
        
        # Add pLDDT data if available  
        plddt_files = glob.glob(os.path.join(input_dir, "plddt_*.npz"))
        if plddt_files:
            metrics["plddt_files"] = [os.path.basename(f) for f in plddt_files]
        
        # Add summary statistics
        if metrics["models"]:
            # Get best model based on confidence score
            best_model = max(metrics["models"], 
                           key=lambda x: x.get("confidence_score", 0))
            
            metrics["best_model_id"] = best_model["model_id"]
            metrics["best_confidence_score"] = best_model.get("confidence_score")
            metrics["best_ptm"] = best_model.get("ptm")
            metrics["best_iptm"] = best_model.get("iptm")
            metrics["best_plddt"] = best_model.get("complex_plddt")
        
        return metrics
    
    def _dock_parallel(self, peptide_sequences: List[str], target_structure: str,
                      target_sequence: str, target_name: str) -> List[str]:
        """
        Dock multiple peptides in parallel using multiple GPUs.
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
            
            process_args.append((
                gpu_peptides,
                self.gpu_ids[gpu_index],
                target_structure,
                target_sequence,
                target_name,
                self  # Pass self to access all parameters
            ))
        
        # Run in parallel
        from multiprocessing import get_context
        context = get_context("spawn")
        
        with context.Pool(processes=len(process_args)) as pool:
            all_docked_dirs = pool.starmap(self._dock_peptides_for_gpu, process_args)
        
        # Flatten the list of lists
        return [dir_path for dirs in all_docked_dirs for dir_path in dirs]
    
    @staticmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, docker_instance) -> List[str]:
        """
        Process a batch of peptides on a specific GPU.
        """
        docked_dirs = []
        for peptide in peptides:
            try:
                dir_path = docker_instance._dock_single_peptide(
                    peptide, target_structure, target_sequence, target_name, gpu_id
                )
                docked_dirs.append(dir_path)
            except Exception as e:
                logging.error(f"Failed to dock {peptide} on GPU {gpu_id}: {e}")
                # Continue with other peptides even if one fails
        
        return docked_dirs
    
    def process_raw_output(self, raw_dir: str, processed_dir: str, 
                          peptide_sequence: str, target_name: str = None) -> str:
        """
        Process raw Boltz output into standardized format.
        
        This method implements the abstract method from BaseDockingModel.
        """
        # Extract target name from directory structure if not provided
        if target_name is None:
            # Try to extract from the directory name
            dir_name = os.path.basename(raw_dir)
            if "_" in dir_name:
                target_name = dir_name.split("_")[0]
            else:
                target_name = "unknown"
        
        self._process_boltz_output(raw_dir, processed_dir, peptide_sequence, target_name)
        return processed_dir
