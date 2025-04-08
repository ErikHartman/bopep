from bopep.docking.dock_peptides import dock_peptides_parallel
from bopep.docking.utils import extract_sequence_from_pdb
import os
from Bio.PDB import PDBParser, PDBIO, Select
import tempfile


class Docker:
    """
    Docker class for docking peptides to a target structure.
    Necessary parameters are passed in as a dictionary with keys:
    - num_models: Number of models to generate.
    - num_recycles: Number of recycling steps.
    - recycle_early_stop_tolerance: Early stopping tolerance for recycling.
    - amber: Whether to use AMBER relaxation.
    - num_relax: Number of relaxation steps.
    - pdb_dir: Output directory for PDB files.
    - gpu_ids: List of GPU IDs to use.
    - overwrite_results: Whether to overwrite existing results.
    """
    def __init__(self, docker_kwargs: dict):
        self.num_models = docker_kwargs.get("num_models", 5)
        self.num_recycles = docker_kwargs.get("num_recycles", 10)
        self.recycle_early_stop_tolerance = docker_kwargs.get("recycle_early_stop_tolerance", 0.01)
        self.amber = docker_kwargs.get("amber", True)
        self.num_relax = docker_kwargs.get("num_relax", 1)
        self.output_dir = docker_kwargs["pdb_dir"]
        self.gpu_ids = docker_kwargs.get("gpu_ids", [])
        self.overwrite_results = docker_kwargs.get("overwrite_results", False)
        self.target_structure_path = None
        self.target_sequence = None
        self.temp_pdb_path = None
        self.target_name = None

    def set_target_structure(self, target_structure_path: str, strip_template: bool = False, 
                            get_first_model: bool = False, keep_chains: str = "A"):
        """
        Set the target structure for docking.
        
        Parameters:
        - target_structure_path: Path to the target PDB file
        - strip_template: If True, keep only the specified chains
        - get_first_model: If True, keep only the first model from the PDB
        - keep_chains: Chains to keep when strip_template is True (default: "A")
        """
        if not os.path.exists(target_structure_path):
            raise FileNotFoundError(
                f"Target structure {target_structure_path} not found."
            )
        
        self.target_name = os.path.basename(target_structure_path).replace(".pdb", "")
        
        # Clean up any previous temporary files
        if self.temp_pdb_path and os.path.exists(self.temp_pdb_path):
            os.remove(self.temp_pdb_path)
            self.temp_pdb_path = None
            
        # Store the original path
        original_path = target_structure_path
        
        # Check if we need to process the PDB file
        if strip_template or get_first_model:
            # Define a custom selector for chains
            class ChainSelect(Select):
                def __init__(self, chains_to_keep):
                    self.chains_to_keep = chains_to_keep
                
                def accept_chain(self, chain):
                    return chain.id in self.chains_to_keep
                
                def accept_model(self, model):
                    # For get_first_model, only accept model 0
                    return model.id == 0 if get_first_model else True
            
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("target", target_structure_path)
            
            # Create a temporary file for the cleaned structure
            fd, temp_path = tempfile.mkstemp(suffix=".pdb", prefix="target_")
            os.close(fd)
            
            # Save the cleaned structure
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_path, ChainSelect(keep_chains))
            
            # Update the path to the temporary file
            target_structure_path = temp_path
            self.temp_pdb_path = temp_path
        
        self.target_structure_path = target_structure_path
        self.target_sequence = extract_sequence_from_pdb(self.target_structure_path, chain_id=keep_chains[0])
        print(f"Target is set to: {original_path}")
        if self.temp_pdb_path:
            print(f"Using cleaned version: {self.temp_pdb_path}")

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
        
        # Dock the peptides and return results
        result = dock_peptides_parallel(
            peptides=peptide_sequences,
            target_structure=self.target_structure_path,
            target_sequence=self.target_sequence,
            num_models=self.num_models,
            num_recycles=self.num_recycles,
            recycle_early_stop_tolerance=self.recycle_early_stop_tolerance,
            amber=self.amber,
            num_relax=self.num_relax,
            output_dir=self.output_dir,
            gpu_ids=self.gpu_ids,
            overwrite_results = self.overwrite_results,
            target_name=self.target_name,
        )
        
        # Clean up temporary files
        self._clean_up()
        
        return result
    
    def _clean_up(self):
        """Remove any temporary files created during processing."""
        if self.temp_pdb_path and os.path.exists(self.temp_pdb_path):
            try:
                os.remove(self.temp_pdb_path)
                self.temp_pdb_path = None
            except OSError as e:
                print(f"Error removing temporary PDB file: {e}")
