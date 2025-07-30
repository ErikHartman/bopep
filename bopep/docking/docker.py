import random
import string
from bopep.docking.dock_peptides import dock_peptides_parallel
from bopep.docking.utils import docking_folder_exists, extract_sequence_from_pdb
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
        """
        Dock multiple peptides to a target structure using ColabFold.

        Will not dock peptide if there is a folder called target_peptide
        in the output directory.

        Will remove unnecessary files generated during docking.
        
        Parameters:
        - peptide_records: List of peptide records (with id and seq properties).
        
        Returns:
        - List of directories containing the docked peptide results.
        """
        if not self.target_structure_path:
            raise ValueError(
                "Target structure not set. Please set the target structure using set_target_structure."
            )
        
        previously_docked_dirs = []
        peptides_to_dock = []

        # Use original target path for checking existing docking results
        for peptide in peptide_sequences: 
            exists, peptide_dir = docking_folder_exists(self.output_dir, peptide, self.original_target_path)
            if exists and not self.overwrite_results:
                previously_docked_dirs.append(peptide_dir)
            else:
                peptides_to_dock.append(peptide)

        if len(peptides_to_dock) == 0:
            return previously_docked_dirs
        else:
            logging.info(f"Will dock {len(peptides_to_dock)} peptides...")

        # Dock the peptides and return results
        docked_dirs = dock_peptides_parallel(
            peptides=peptides_to_dock,
            target_structure=self.target_structure_path,  # Use cleaned structure for docking
            target_sequence=self.target_sequence,
            num_models=self.num_models,
            num_recycles=self.num_recycles,
            recycle_early_stop_tolerance=self.recycle_early_stop_tolerance,
            amber=self.amber,
            num_relax=self.num_relax,
            output_dir=self.output_dir,
            gpu_ids=self.gpu_ids,
            target_name=self.target_name,  # Pass the original target name for consistent naming
        )
        docked_dirs += previously_docked_dirs
        # Clean up temporary files
        self._clean_up()
        
        return docked_dirs
    
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