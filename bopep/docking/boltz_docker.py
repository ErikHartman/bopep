import os
import json
import yaml
import subprocess
import glob
import shutil
from typing import List, Dict, Any
from bopep.docking.base_docking_model import BaseDockingModel
import logging
import numpy as np

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
        
        raw_peptide_dir = self._create_raw_peptide_dir(target_name, peptide_sequence)
        processed_peptide_dir = os.path.join(self.processed_output_dir, f"{target_name}_{peptide_sequence}")
        
        os.makedirs(processed_peptide_dir, exist_ok=True)
        
        yaml_path = self._create_yaml_config(peptide_sequence, target_sequence, 
                                           target_name, raw_peptide_dir, target_structure)

        self._run_boltz_prediction(yaml_path, raw_peptide_dir, gpu_id)
    
        self._process_boltz_output(raw_peptide_dir, processed_peptide_dir, 
                                    peptide_sequence, target_name)
        
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
        temp_docker = BoltzDocker(
            output_dir=os.path.dirname(raw_output_dir),
            gpu_ids=[gpu_id],
            **method_params
        )
        
        docked_dirs = []
        for peptide in peptides:
            dir_path = temp_docker._dock_single_peptide(
                peptide, target_structure, target_sequence, target_name, gpu_id
            )
            docked_dirs.append(dir_path)
 
        
        return docked_dirs
    
    def _create_yaml_config(self, peptide_sequence: str, target_sequence: str,
                           target_name: str, output_dir: str, target_structure: str) -> str:
        """
        Create a YAML configuration file for Boltz.
        """
        config = {
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": target_sequence,
                        "msa": "empty"  # Placeholder for MSA
                    }
                },
                {
                    "protein": {
                        "id": "B", 
                        "sequence": peptide_sequence,
                        "msa": "empty"  # Placeholder for MSA
                    }
                }
            ]
        }
        
        template_path = self._prepare_template_file(target_structure, output_dir)
        config["templates"] = [
            {
                "cif": template_path,
                "chain_id": "A"
            }
        ]
        
        yaml_path = os.path.join(output_dir, f"{target_name}_{peptide_sequence}.yaml")
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logging.info(f"Created YAML config: {yaml_path}")
        return yaml_path
    
    def _prepare_template_file(self, target_structure: str, output_dir: str) -> str:
        """
        Prepare template file for Boltz.
        """
        if target_structure.lower().endswith('.cif'):
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
        

        boltz_results_pattern = f"boltz_results_{target_name}_{peptide_sequence}"
        boltz_results_dir = os.path.join(raw_dir, boltz_results_pattern)
        
        predictions_dir = os.path.join(boltz_results_dir, "predictions")
        
        if not os.path.exists(predictions_dir):
            raise ValueError(f"Predictions directory not found: {predictions_dir}")
        
        # Find the input-specific subdirectory
        input_dir = [d for d in os.listdir(predictions_dir) 
                     if os.path.isdir(os.path.join(predictions_dir, d))][0]
        
        if not input_dir:
            raise ValueError(f"No input directories found in {predictions_dir}")

        # Find structure files
        if self.output_format == "pdb":
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.pdb"))
        else:  # mmcif
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.cif"))
        
        structure_files.sort()  # Sort to maintain consistent ordering
        
        for i, struct_file in enumerate(structure_files, 1):
            if self.output_format == "pdb":
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.pdb")
            else:
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.cif")
            
            shutil.copy2(struct_file, dest_file)
            logging.info(f"Copied {os.path.basename(struct_file)} -> {os.path.basename(dest_file)}")
        
        metrics = self._extract_boltz_metrics(input_dir, peptide_sequence)
        
        metrics_file = os.path.join(processed_dir, "boltz_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Created standardized metrics file: {metrics_file}")
    
    def _extract_boltz_metrics(self, input_dir: str, peptide_sequence: str) -> Dict[str, Any]:
        # Get best model on confidence score and make primary
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
            with open(conf_file, 'r') as f:
                confidence_data = json.load(f)
            
            model_metrics = {
                "model_id": i,
                "confidence_file": os.path.basename(conf_file),
                **confidence_data
            }
            metrics["models"].append(model_metrics)
            
        affinity_files = glob.glob(os.path.join(input_dir, "affinity_*.json"))
        if affinity_files:
            with open(affinity_files[0], 'r') as f:
                affinity_data = json.load(f)
            
            metrics["affinity"] = affinity_data

        
        self._extract_npz_data_to_metrics(input_dir, metrics)
        
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
    
    def _extract_npz_data_to_metrics(self, input_dir: str, metrics: Dict[str, Any]) -> None:
        """
        Extract confidence data from NPZ files and add to metrics as JSON-serializable data.

        Here we assume model 1 is primary model - which might not be the case.
        
        Args:
            input_dir: Directory containing Boltz NPZ files
            metrics: Metrics dict to update with extracted data
        """

        # Extract PAE data
        pae_files = glob.glob(os.path.join(input_dir, "pae_*.npz"))
        if pae_files:
            # Sort files to ensure consistent ordering
            pae_files.sort()

            # Load first PAE file (assuming model 1 is primary)
            pae_data = np.load(pae_files[0])
            if 'pae' in pae_data:
                pae_matrix = pae_data['pae'].tolist()  # Convert to JSON-serializable list
                metrics["pae_matrix"] = pae_matrix
                logging.info(f"Extracted PAE matrix with shape: {np.array(pae_matrix).shape}")

        
        # Extract pLDDT data
        plddt_files = glob.glob(os.path.join(input_dir, "plddt_*.npz"))
        if plddt_files:
            # Sort files to ensure consistent ordering
            plddt_files.sort()

            # Load first pLDDT file (assuming model 1 is primary)
            plddt_data = np.load(plddt_files[0])
            if 'plddt' in plddt_data:
                plddt_vector = plddt_data['plddt'].tolist()  # Convert to JSON-serializable list
                metrics["plddt_vector"] = plddt_vector
                logging.info(f"Extracted pLDDT vector with length: {len(plddt_vector)}")

        
        # Extract PDE data (Protein Distance Error - specific to Boltz)
        pde_files = glob.glob(os.path.join(input_dir, "pde_*.npz"))
        if pde_files:
            # Sort files to ensure consistent ordering
            pde_files.sort()

            # Load first PDE file (assuming model 1 is primary)
            pde_data = np.load(pde_files[0])
            if 'pde' in pde_data:
                pde_matrix = pde_data['pde'].tolist()  # Convert to JSON-serializable list (actually a matrix)
                metrics["pde_vector"] = pde_matrix  # Keep the key name for backward compatibility
                logging.info(f"Extracted PDE matrix with shape: {np.array(pde_matrix).shape}")

                
