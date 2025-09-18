import random
import string
from bopep.docking.alphafold_docker import AlphaFoldDocker
from bopep.docking.boltz_docker import BoltzDocker
from bopep.structure.parser import extract_sequence_from_structure, parse_structure
import os
from Bio.PDB import PDBIO, Select
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Docker:
    """
    Docker class for docking peptides to a target structure.
    
    Parameters:
    - **kwargs: Model-specific parameters (passed to individual docking classes)
    
    For AlphaFold parameters, see AlphaFoldDocker documentation.
    For Boltz parameters, see BoltzDocker documentation.
    """
    def __init__(self, kwargs):
        models = kwargs.pop("models", None)
        output_dir = kwargs.pop("output_dir", None)

        if models is not None:
            self.models = models if isinstance(models, list) else [models]
        else:
            raise ValueError("'models' must be specified.")
        
        # Validate supported models
        supported_models = ["alphafold", "boltz"]
        for model in self.models:
            if model not in supported_models:
                raise ValueError(f"Unsupported model: {model}. Supported models are {supported_models}")
        
        # Base parameters
        if output_dir is None:
            raise ValueError("output_dir directory must be specified.")
        
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Store all kwargs for passing to specific docking classes
        self.docking_kwargs = kwargs
        
        # Target structure tracking
        self.target_structure_path = None
        self.original_target_path = None
        self.target_sequence = None
        self.temp_pdb_path = None
        self.target_name = None

    def set_target_structure(self, target_structure_path: str, strip_template: bool = False, 
                        get_first_model: bool = False, keep_chains: str = "A"):
        """
        Set the target structure for docking.
        
        Parameters:
        - target_structure_path: Path to the target PDB file
        - strip_template: If True, keep only the specified chains and amino acids (removes waters/ligands)
        - get_first_model: If True, keep only the first model from the PDB
        - keep_chains: Chains to keep when strip_template is True (default: "A")
        """
        if not os.path.exists(target_structure_path):
            raise FileNotFoundError(
                f"Target structure {target_structure_path} not found."
            )
        
        self.original_target_path = target_structure_path
        # Extract target name by removing both .pdb and .cif extensions
        base_name = os.path.basename(target_structure_path)
        if base_name.lower().endswith('.cif'):
            self.target_name = base_name[:-4]  # Remove .cif
        elif base_name.lower().endswith('.pdb'):
            self.target_name = base_name[:-4]  # Remove .pdb
        else:
            self.target_name = os.path.splitext(base_name)[0]  # Remove any other extension
        
        if self.temp_pdb_path and os.path.exists(self.temp_pdb_path):
            os.remove(self.temp_pdb_path)
            self.temp_pdb_path = None
            
        if strip_template or get_first_model:

            if target_structure_path.lower().endswith('.cif'):
                raise ValueError("CIF files cannot be processed with strip_template or get_first_model options.")
            
            class ChainSelect(Select):
                def __init__(self, chains_to_keep):
                    self.chains_to_keep = chains_to_keep
                
                def accept_chain(self, chain):
                    return chain.id in self.chains_to_keep
                
                def accept_model(self, model):
                    return model.id == 0 if get_first_model else True
                
                def accept_residue(self, residue):
                    # If strip_template is True, only keep standard amino acids
                    if strip_template:
                        if residue.get_resname() in ["HOH", "WAT"]:
                            return 0
                        return residue.get_id()[0] == " "
                    return 1
            
            # Parse the PDB file
            structure = parse_structure(target_structure_path, structure_id="target")
            
            # Create a temporary file for the cleaned structure
            # Generate a 4-character PDB-style ID
            pdb_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            temp_dir = tempfile.mkdtemp()  # Create a temporary directory
            temp_path = os.path.join(temp_dir, f"{pdb_id}.pdb")  # Create path with exact PDB format
            
            # Save the cleaned structure
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_path, ChainSelect(keep_chains))
            
            # Update the path to the temporary file
            self.target_structure_path = temp_path
            self.temp_pdb_path = temp_path
        else:
            # If no processing needed, use the original path
            self.target_structure_path = target_structure_path
        
        self.target_sequence = extract_sequence_from_structure(self.target_structure_path, chain_id=keep_chains[0])
        logging.info(f"Target is set to: {self.original_target_path}")
        if self.temp_pdb_path:
            logging.info(f"Using cleaned version: {self.temp_pdb_path}")
            if strip_template:
                logging.info("Removed waters and ligands, keeping only standard amino acids")
        self._log_config()

    def dock_peptides(self, peptide_sequences: list):
        """
        Dock peptides using all specified models.
        
        Parameters:
        - peptide_sequences: List of peptide sequences to dock
        
        Returns:
        - Dictionary with model names as keys and list of processed directories as values
        """
        if not self.target_structure_path:
            raise ValueError(
                "Target structure not set. Please set the target structure using set_target_structure."
            )


        all_docked_dirs = {}

        for model in self.models:
            logging.info(f"Starting docking with {model.upper()}...")
            
            if model == "alphafold":
                docker_dirs = self._dock_with_alphafold(peptide_sequences)
            elif model == "boltz":
                docker_dirs = self._dock_with_boltz(peptide_sequences)
            else:
                raise ValueError(f"Unsupported model: {model}")
            all_docked_dirs[model] = docker_dirs
            logging.info(f"Completed {model.upper()} docking for {len(docker_dirs)} peptides")
        
        # Clean up temporary files after all docking is complete
        self._clean_up()

        # For item in all results, ensure they are the same
        # Assert that values in all results are the same
        for result_item in all_docked_dirs.values():
            if set(result_item) != set(docker_dirs):
                raise ValueError("Results from different models do not match. Check your docking parameters.")
   
        return docker_dirs
        
    def _dock_with_alphafold(self, peptide_sequences: list):
        """Dock peptides using AlphaFold/ColabFold."""
        # Create alphafold instance and let it handle its own parameters
        alphafold_docker = AlphaFoldDocker(
            output_dir=self.output_dir,
            **self.docking_kwargs  # Pass all kwargs, AlphaFoldDocker will extract what it needs
        )
        
        # Dock the peptides
        return alphafold_docker.dock(
            peptide_sequences, 
            self.target_structure_path, 
            self.target_sequence, 
            self.target_name
        )

    def _dock_with_boltz(self, peptide_sequences: list):
        """Dock peptides using Boltz."""
        # Create boltz instance and let it handle its own parameters
        boltz_docker = BoltzDocker(
            output_dir=self.output_dir,
            **self.docking_kwargs  # Pass all kwargs, BoltzDocker will extract what it needs
        )
        
        # Dock the peptides
        return boltz_docker.dock(
            peptide_sequences, 
            self.target_structure_path, 
            self.target_sequence, 
            self.target_name
        )

    def _clean_up(self):
        """Remove any temporary files created during processing."""
        if self.temp_pdb_path and os.path.exists(self.temp_pdb_path):
            try:
                # Remove the file
                os.remove(self.temp_pdb_path)
                
                # Also remove the parent directory if it was created with tempfile.mkdtemp()
                temp_dir = os.path.dirname(self.temp_pdb_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    
                self.temp_pdb_path = None
            except OSError as e:
                logging.info(f"Error removing temporary PDB file: {e}")

    def _log_config(self):
        """Log the current configuration of the Docker instance."""
        logging.info("Docker configuration:")
        logging.info(f"Target structure: {self.target_structure_path}")
        logging.info(f"Base output directory: {self.output_dir}")
        logging.info(f"Models: {self.models}")
        
        # Log method-specific parameters if present
        if "alphafold" in self.models:
            logging.info("AlphaFold parameters:")
            for param in ["num_models", "num_recycles", "recycle_early_stop_tolerance", "amber", "num_relax"]:
                if param in self.docking_kwargs:
                    logging.info(f"  {param}: {self.docking_kwargs[param]}")
        
        if "boltz" in self.models:
            logging.info("Boltz parameters:")
            for param in ["diffusion_samples", "output_format", "sampling_steps", "step_scale"]:
                if param in self.docking_kwargs:
                    logging.info(f"  {param}: {self.docking_kwargs[param]}")