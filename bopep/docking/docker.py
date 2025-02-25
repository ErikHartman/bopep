from bopep.docking.dock_peptides import dock_peptides_parallel
from bopep.docking.utils import extract_sequence_from_pdb


class Docker:
    def __init__(self, docker_kwargs: dict):
        self.target_structure_path = docker_kwargs["target_structure_path"]
        self.num_models = docker_kwargs["num_models"]
        self.num_recycles = docker_kwargs["num_recycles"]
        self.recycle_early_stop_tolerance = docker_kwargs[
            "recycle_early_stop_tolerance"
        ]
        self.amber = docker_kwargs["amber"]
        self.num_relax = docker_kwargs["num_relax"]
        self.output_dir = docker_kwargs["pdb_dir"]
        self.gpu_ids = docker_kwargs["gpu_ids"]
        self.target_sequence = extract_sequence_from_pdb(self.target_structure)


    def dock_peptides(self, peptides):
        """
        Dock multiple peptides to a target structure using ColabFold.
        """
        return dock_peptides_parallel(
            peptides=peptides,
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
