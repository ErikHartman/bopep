import random
import string
from bopep.docking.alphafold_docker import AlphaFoldDocker
from bopep.docking.utils import extract_sequence_from_pdb
import os
from Bio.PDB import PDBParser, PDBIO, Select
import tempfile
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Docker:
    """
    Docker class for docking peptides to a target structure.
    Necessary parameters are passed in as a dictionary with keys:
    - num_models: Number of models to generate.
    - num_recycles: Number of recycling steps.
    - recycle_early_stop_tolerance: Early stopping tolerance for recycling.
    - amber: Whether to use AMBER relaxation.
    - num_relax: Number of relaxation steps.
    - output_dir: Output directory for PDB files.
    - gpu_ids: List of GPU IDs to use.
    - overwrite_results: Whether to overwrite existing results.
    """
    def __init__(self, docker_kwargs: dict):
        self.model = docker_kwargs.get("model", "alphafold2")
        self.num_models = docker_kwargs.get("num_models", 5)
        self.num_recycles = docker_kwargs.get("num_recycles", 10)
        self.recycle_early_stop_tolerance = docker_kwargs.get("recycle_early_stop_tolerance", 0.01)
        self.amber = docker_kwargs.get("amber", True)
        self.num_relax = docker_kwargs.get("num_relax", 1)
        self.output_dir = docker_kwargs["output_dir"]
        self.gpu_ids = docker_kwargs.get("gpu_ids", [0])
        self.overwrite_results = docker_kwargs.get("overwrite_results", False)
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
        self.target_name = os.path.basename(target_structure_path).replace(".pdb", "")
        
        if self.temp_pdb_path and os.path.exists(self.temp_pdb_path):
            os.remove(self.temp_pdb_path)
            self.temp_pdb_path = None
            
        if strip_template or get_first_model:
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
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("target", target_structure_path)
            
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
        
        self.target_sequence = extract_sequence_from_pdb(self.target_structure_path, chain_id=keep_chains[0])
        logging.info(f"Target is set to: {self.original_target_path}")
        if self.temp_pdb_path:
            logging.info(f"Using cleaned version: {self.temp_pdb_path}")
            if strip_template:
                logging.info("Removed waters and ligands, keeping only standard amino acids")
        self._log_config()

    def dock_peptides(self, peptide_sequences: list):
        if self.model == "alphafold2":
            logging.info("Using AlphaFold2 for docking.")
            return self.dock_alphafold2(peptide_sequences)
        elif self.model == "boltz2":
            logging.info("Using Boltz sampling for docking.")
            return self.dock_boltz2(peptide_sequences)
        else:
            raise ValueError(f"Unsupported model: {self.model}. Supported models are 'alphafold2' and 'boltz2'.")

    def dock_boltz2(self, peptide_sequences: list):
        """
        Dock multiple peptides to a target structure using Boltz.
        
        Returns list of processed output directories.
        """
        if not self.target_structure_path:
            raise ValueError(
                "Target structure not set. Please set the target structure using set_target_structure."
            )
        
        # Import here to avoid circular imports
        from bopep.docking.boltz_docker import BoltzDocker
        
        # Create Boltz docker with current parameters
        boltz_docker = BoltzDocker(
            output_dir=self.output_dir,
            gpu_ids=self.gpu_ids,
            overwrite_results=self.overwrite_results,
            recycling_steps=self.num_recycles,  # Map to Boltz parameter name
            diffusion_samples=getattr(self, 'diffusion_samples', 1),
            use_msa_server=getattr(self, 'use_msa_server', True),
            use_potentials=getattr(self, 'use_potentials', False),
            output_format=getattr(self, 'output_format', 'pdb'),
        )
        
        # Perform docking
        processed_dirs = boltz_docker.dock(
            peptide_sequences=peptide_sequences,
            target_structure=self.target_structure_path,
            target_sequence=self.target_sequence,
            target_name=self.target_name,
        )
        
        # Clean up temporary files
        self._clean_up()
        
        return processed_dirs

    def dock_alphafold2(self, peptide_sequences: list):
        """
        Dock multiple peptides to a target structure using AlphaFold2/ColabFold.
        
        Returns list of processed output directories.
        """
        if not self.target_structure_path:
            raise ValueError(
                "Target structure not set. Please set the target structure using set_target_structure."
            )
        
        # Create AlphaFold docker with current parameters
        alphafold_docker = AlphaFoldDocker(
            output_dir=self.output_dir,
            gpu_ids=self.gpu_ids,
            overwrite_results=self.overwrite_results,
            num_models=self.num_models,
            num_recycles=self.num_recycles,
            recycle_early_stop_tolerance=self.recycle_early_stop_tolerance,
            amber=self.amber,
            num_relax=self.num_relax,
        )
        
        # Perform docking
        processed_dirs = alphafold_docker.dock(
            peptide_sequences=peptide_sequences,
            target_structure=self.target_structure_path,
            target_sequence=self.target_sequence,
            target_name=self.target_name,
        )
        
        # Clean up temporary files
        self._clean_up()
        
        return processed_dirs
    
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
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Number of models: {self.num_models}")
        logging.info(f"Number of recycles: {self.num_recycles}")
        logging.info(f"Recycle early stop tolerance: {self.recycle_early_stop_tolerance}")
        logging.info(f"Using AMBER relaxation: {self.amber}")
        logging.info(f"Number of relaxations: {self.num_relax}")
        logging.info(f"GPU IDs: {self.gpu_ids}")
        logging.info(f"Overwrite results: {self.overwrite_results}")