import os
import json
import yaml
import subprocess
import glob
import shutil
from typing import List, Dict, Any, Tuple
from bopep.docking.base_docking_model import BaseDockingModel
from bopep.structure.parser import get_chain_sequences
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
        self.cache = kwargs.get("cache") or kwargs.get("cache_dir")
        
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
        processed_peptide_dir = self._create_processed_peptide_dir(target_name, peptide_sequence)
        
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
        params = {
            'recycling_steps': self.recycling_steps,
            'diffusion_samples': self.diffusion_samples,
            'output_format': self.output_format,
            'sampling_steps': self.sampling_steps,
            'step_scale': self.step_scale,
            'overwrite_results': self.overwrite_results
        }
        return params
    
    @staticmethod
    def _dock_peptides_for_gpu(peptides: List[str], gpu_id: str, target_structure: str,
                              target_sequence: str, target_name: str, raw_output_dir: str,
                              method_params: dict) -> List[tuple]:
        """
        Process a batch of peptides on a specific GPU using Boltz.
        
        Returns:
            List of (peptide_sequence, raw_dir_path) tuples
        """
        # raw_output_dir is like: /base/raw/boltz
        # We need to get back to /base (go up 2 levels: remove /boltz and /raw)
        output_dir = os.path.dirname(os.path.dirname(raw_output_dir))
        
        temp_docker = BoltzDocker(
            output_dir=output_dir,
            gpu_ids=[gpu_id],
            **method_params
        )
        
        docked_results = []
        for i, peptide in enumerate(peptides, 1):
            print(f"GPU {gpu_id} progress: {i}/{len(peptides)} - docking {peptide}")
            dir_path = temp_docker._dock_single_peptide(
                peptide, target_structure, target_sequence, target_name, gpu_id
            )
            if dir_path:
                docked_results.append((peptide, dir_path))
        
        
        return docked_results
    
    def _detect_protein_chain_and_sequence(self, target_structure: str) -> Tuple[str, str]:
        """
        Detect the protein chain ID and extract its sequence from the structure.
        
        Returns:
            Tuple of (original_chain_id, sequence)
        """
        chain_sequences = get_chain_sequences(target_structure)
        
        if not chain_sequences:
            raise ValueError(f"No protein chains found in structure: {target_structure}")
        
        # Find the chain with the longest sequence (most likely the main protein)
        protein_chain_id = max(chain_sequences.keys(), key=lambda k: len(chain_sequences[k]))
        protein_sequence = chain_sequences[protein_chain_id]
        
        logging.info(f"Detected protein chain '{protein_chain_id}' with {len(protein_sequence)} residues")
        return protein_chain_id, protein_sequence
    
    def _create_yaml_config(self, peptide_sequence: str, target_sequence: str,
                           target_name: str, output_dir: str, target_structure: str) -> str:
        """
        Create a YAML configuration file for Boltz.
        """
        # Auto-detect the protein chain and sequence from the structure
        protein_chain_id, detected_sequence = self._detect_protein_chain_and_sequence(target_structure)
        
        # Use the detected sequence if target_sequence is empty or if they don't match
        if not target_sequence or target_sequence != detected_sequence:
            if target_sequence and target_sequence != detected_sequence:
                logging.warning(f"Provided target sequence doesn't match detected sequence. Using detected sequence from chain {protein_chain_id}")
            target_sequence = detected_sequence
        
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
                "chain_id": "A"  # Always use "A" - it refers to the first protein sequence above
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
        else:
            raise ValueError(
                f"Unsupported file format: {target_structure}. Only .cif files are supported for Boltz."
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
        
        if getattr(self, 'cache', None):
            cmd.extend(["--cache", self.cache])
        
        if self.overwrite_results:
            cmd.append("--override")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.pop("MPLBACKEND", None)
        if getattr(self, 'cache', None):
            env["BOLTZ_CACHE"] = self.cache
        
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
        Process Boltz raw output into standardized format with best model as model_1.
        """
        logging.info("Processing Boltz output into standardized format...")
        
        boltz_results_pattern = f"boltz_results_{target_name}_{peptide_sequence}"
        boltz_results_dir = os.path.join(raw_dir, boltz_results_pattern)
        
        predictions_dir = os.path.join(boltz_results_dir, "predictions")
        
        if not os.path.exists(predictions_dir):
            raise ValueError(f"Predictions directory not found: {predictions_dir}")
        
        # Find the input-specific subdirectory
        input_dirs = [d for d in os.listdir(predictions_dir) 
                     if os.path.isdir(os.path.join(predictions_dir, d))]
        
        if not input_dirs:
            raise ValueError(f"No input directories found in {predictions_dir}")
        
        input_dir = os.path.join(predictions_dir, input_dirs[0])
        
        # Extract metrics and get best model ID
        metrics, best_model_id = self._extract_boltz_metrics(input_dir, peptide_sequence)
        
        # Find structure files
        if self.output_format == "pdb":
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.pdb"))
        else:  # mmcif
            structure_files = glob.glob(os.path.join(input_dir, "*_model_*.cif"))
        
        structure_files.sort()  # Sort to maintain consistent ordering
        
        if not structure_files:
            raise ValueError(f"No structure files found in {input_dir}")
        
        # Reorder files so best model comes first
        if best_model_id <= len(structure_files):
            best_file = structure_files[best_model_id - 1]  # Convert to 0-based index
            other_files = [f for i, f in enumerate(structure_files) if i != best_model_id - 1]
            ordered_files = [best_file] + other_files
        else:
            logging.warning(f"Best model ID {best_model_id} exceeds number of structure files {len(structure_files)}")
            ordered_files = structure_files
        
        # Copy files with best model as model_1
        for i, struct_file in enumerate(ordered_files, 1):
            if self.output_format == "pdb":
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.pdb")
            else:
                dest_file = os.path.join(processed_dir, f"boltz_model_{i}.cif")
            
            shutil.copy2(struct_file, dest_file)
            if i == 1:
                logging.info(f"Copied BEST model {os.path.basename(struct_file)} -> {os.path.basename(dest_file)} (ipTM: {metrics.get('iptm', 'N/A')})")
            else:
                logging.info(f"Copied {os.path.basename(struct_file)} -> {os.path.basename(dest_file)}")
        
        # Save metrics file
        metrics_file = os.path.join(processed_dir, "boltz_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Created standardized metrics file: {metrics_file}")
    
    def _extract_boltz_metrics(self, input_dir: str, peptide_sequence: str) -> tuple[Dict[str, Any], int]:
        """
        Extract metrics from Boltz output files and return only the best model based on ipTM.
        """
        # Find all confidence files (one per model)
        confidence_files = glob.glob(os.path.join(input_dir, "confidence_*.json"))
        confidence_files.sort()
        
        all_models = []
        for i, conf_file in enumerate(confidence_files, 1):
            with open(conf_file, 'r') as f:
                confidence_data = json.load(f)
            
            model_metrics = {
                "model_id": i,
                "confidence_file": os.path.basename(conf_file),
                **confidence_data
            }
            all_models.append(model_metrics)
        
        if not all_models:
            raise ValueError(f"No confidence files found in {input_dir}")
        
        # Find best model based on ipTM (iptm key)
        best_model = max(all_models, key=lambda x: x.get("iptm", 0))
        best_model_id = best_model["model_id"]
        
        logging.info(f"Best model based on ipTM: model_{best_model_id} (ipTM: {best_model.get('iptm', 'N/A')})")
        
        # Create metrics dict with only the best model's data
        metrics = {
            "peptide_sequence": peptide_sequence,
            "docking_method": "boltz",
            "best_model_id": best_model_id,
            # Store all metrics from best model at root level (like AlphaFold)
            **{k: v for k, v in best_model.items() if k not in ["model_id", "confidence_file"]}
        }
        
        # Add affinity data if available
        affinity_files = glob.glob(os.path.join(input_dir, "affinity_*.json"))
        if affinity_files:
            with open(affinity_files[0], 'r') as f:
                affinity_data = json.load(f)
            metrics["affinity"] = affinity_data
        
        # Extract NPZ data for the best model
        self._extract_npz_data_to_metrics(input_dir, metrics, best_model_id)
        
        return metrics, best_model_id
    
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
    
    def _extract_npz_data_to_metrics(self, input_dir: str, metrics: Dict[str, Any], best_model_id: int) -> None:
        """
        Extract confidence data from NPZ files for the best model only.
        """
        # Extract PAE data for best model
        pae_files = glob.glob(os.path.join(input_dir, "pae_*.npz"))
        if pae_files:
            pae_files.sort()
            if len(pae_files) >= best_model_id:
                pae_file = pae_files[best_model_id - 1]  # Convert to 0-based index
                pae_data = np.load(pae_file)
                if 'pae' in pae_data:
                    pae_matrix = pae_data['pae'].tolist()
                    metrics["pae_matrix"] = pae_matrix
                    logging.info(f"Extracted PAE matrix for model {best_model_id} with shape: {np.array(pae_matrix).shape}")
        
        # Extract pLDDT data for best model
        plddt_files = glob.glob(os.path.join(input_dir, "plddt_*.npz"))
        if plddt_files:
            plddt_files.sort()
            if len(plddt_files) >= best_model_id:
                plddt_file = plddt_files[best_model_id - 1]  # Convert to 0-based index
                plddt_data = np.load(plddt_file)
                if 'plddt' in plddt_data:
                    plddt = plddt_data['plddt'].tolist()
                    metrics["plddt"] = plddt
                    logging.info(f"Extracted pLDDT vector for model {best_model_id} with length: {len(plddt)}")
        
        # Extract PDE data for best model
        pde_files = glob.glob(os.path.join(input_dir, "pde_*.npz"))
        if pde_files:
            pde_files.sort()
            if len(pde_files) >= best_model_id:
                pde_file = pde_files[best_model_id - 1]  # Convert to 0-based index
                pde_data = np.load(pde_file)
                if 'pde' in pde_data:
                    pde_matrix = pde_data['pde'].tolist()
                    metrics["pde_matrix"] = pde_matrix  # Updated key name
                    logging.info(f"Extracted PDE matrix for model {best_model_id} with shape: {np.array(pde_matrix).shape}")


