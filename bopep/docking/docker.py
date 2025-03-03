from bopep.docking.dock_peptides import dock_peptides_parallel
from bopep.docking.utils import extract_sequence_from_pdb
import os


class Docker:
    """
    Docker class for docking peptides to a target structure.
    """
    def __init__(self, docker_kwargs: dict):
        self.num_models = docker_kwargs["num_models"]
        self.num_recycles = docker_kwargs["num_recycles"]
        self.recycle_early_stop_tolerance = docker_kwargs[
            "recycle_early_stop_tolerance"
        ]
        self.amber = docker_kwargs["amber"]
        self.num_relax = docker_kwargs["num_relax"]
        self.output_dir = docker_kwargs["pdb_dir"]
        self.gpu_ids = docker_kwargs["gpu_ids"]
        self.overwrite_results = docker_kwargs.get("overwrite_results", False)
        self.target_structure_path = None
        self.target_sequence = None

    def set_target_structure(self, target_structure_path):
        if not os.path.exists(target_structure_path):
            raise FileNotFoundError(
                f"Target structure {target_structure_path} not found."
            )

        self.target_structure_path = target_structure_path
        self.target_sequence = extract_sequence_from_pdb(self.target_structure_path)
        print("Target is set to: ", self.target_structure_path)

    def dock_peptides(self, peptide_sequences):
        """
        Dock multiple peptides to a target structure using ColabFold.
        
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
        return dock_peptides_parallel(
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
        )
