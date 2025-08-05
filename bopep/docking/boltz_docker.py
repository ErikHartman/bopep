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
        self.method_name = "boltz"
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
        return self._dock_with_common_logic(peptide_sequences, target_structure, 
                                          target_sequence, target_name)
    
    def _dock_single_peptide(self, peptide_sequence: str, target_structure: str,
                            target_sequence: str, target_name: str, gpu_id: str = "0") -> str:
        """
        Dock a single peptide using Boltz.
        """
        logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")
        
        # Create output directories
        raw_peptide_dir = self._create_raw_peptide_dir(target_name, peptide_sequence)
        processed_peptide_dir = os.path.join(self.processed_output_dir, f"{target_name}_{peptide_sequence}")
        
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
        
        return raw_peptide_dir
    
    def _get_method_parameters(self) -> dict:
        """
        Get Boltz-specific parameters for parallel processing.
        """
        return {
            'recycling_steps': self.recycling_steps,
            'diffusion_samples': self.diffusion_samples,
            'output_format': self.output_format,
            'sampling_steps': self.sampling_steps,
            'step_scale': self.step_scale,
            'overwrite_results': self.overwrite_results
        }
    
    @staticmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              method_params: dict) -> List[str]:
        """
        Process a batch of peptides on a specific GPU using Boltz.
        """
        # Create a temporary docker instance for this process
        temp_docker = BoltzDocker(
            output_dir=os.path.dirname(raw_output_dir),  # Parent of raw_output_dir
            gpu_ids=[gpu_id],
            **method_params
        )
        
        docked_dirs = []
        for peptide in peptides:
            try:
                dir_path = temp_docker._dock_single_peptide(
                    peptide, target_structure, target_sequence, target_name, gpu_id
                )
                docked_dirs.append(dir_path)
            except Exception as e:
                logging.error(f"Failed to dock {peptide} on GPU {gpu_id}: {e}")
                # Continue with other peptides even if one fails
        
        return docked_dirs
    
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
        
        # Find the Boltz results directory - it should be named like "boltz_results_{target_name}_{peptide_sequence}"
        boltz_results_pattern = f"boltz_results_{target_name}_{peptide_sequence}"
        boltz_results_dir = os.path.join(raw_dir, boltz_results_pattern)
        
        if not os.path.exists(boltz_results_dir):
            # Fallback: look for any directory starting with "boltz_results_"
            possible_dirs = [d for d in os.listdir(raw_dir) 
                           if d.startswith("boltz_results_") and os.path.isdir(os.path.join(raw_dir, d))]
            if possible_dirs:
                boltz_results_dir = os.path.join(raw_dir, possible_dirs[0])
                logging.warning(f"Expected {boltz_results_pattern} but found {possible_dirs[0]}")
            else:
                raise ValueError(f"Boltz results directory not found. Expected: {boltz_results_dir}")
        
        # Find the prediction directory inside the boltz results
        predictions_dir = os.path.join(boltz_results_dir, "predictions")
        
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
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.pdb")
            else:
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.cif")
            
            shutil.copy2(struct_file, dest_file)
            logging.info(f"Copied {os.path.basename(struct_file)} -> {os.path.basename(dest_file)}")
        
        # Process and combine all metrics into a single JSON file
        metrics = self._extract_boltz_metrics(input_dir, peptide_sequence)
        
        metrics_file = os.path.join(processed_dir, "boltz_metrics.json")
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
    
    def process_raw_output(self, raw_peptide_dir: str, peptide_sequence: str, 
                          target_name: str) -> str:
        """
        Process raw Boltz output into standardized format.
        
        This method implements the abstract method from BaseDockingModel.
        """
        # Create the processed directory path using the same pattern as base class
        peptide_dir_name = f"{target_name}_{peptide_sequence}"
        processed_dir = os.path.join(self.processed_output_dir, peptide_dir_name)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Extract target name from directory structure if not provided
        if target_name is None:
            # Try to extract from the directory name
            dir_name = os.path.basename(raw_peptide_dir)
            if "_" in dir_name:
                target_name = dir_name.split("_")[0]
            else:
                target_name = "unknown"
        
        self._process_boltz_output(raw_peptide_dir, processed_dir, peptide_sequence, target_name)
        return processed_dir
