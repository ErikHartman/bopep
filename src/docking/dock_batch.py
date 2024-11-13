import os

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from .dock_peptides import dock_peptides
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def dock_peptides_batch(peptides, docking_params):
    """
    Dock a batch of peptides using ColabFold and specified docking parameters.

    Parameters:
    - peptides: List of peptides to dock.
    - docking_params: Dictionary of docking parameters.
    """
    fasta_file = create_fasta(peptides, docking_params["output_dir"])

    dock_peptides(
        multi_peptide_fasta=fasta_file,
        target_structure=docking_params["target_structure"],
        output_dir=docking_params["output_dir"],
        num_recycles=docking_params["num_recycles"],
        msa_mode=docking_params["msa_mode"],
        model_type=docking_params["model_type"],
        num_relax=docking_params["num_relax"],
        num_models=docking_params["num_models"],
        recycle_early_stop_tolerance=docking_params["recycle_early_stop_tolerance"],
        amber=docking_params["amber"],
        target_chain=docking_params["target_chain"],
        num_processes=docking_params["num_processes"],
        gpu_ids=docking_params["gpu_ids"],
        overwrite_results=docking_params["overwrite_results"],
    )

    os.remove(fasta_file)

def create_fasta(peptides, output_dir, filename="peptides.fasta"):
    """
    Create a FASTA file from a list of peptide sequences.

    Parameters:
    - peptides: List of peptide sequences.
    - output_dir: Directory where the FASTA file will be saved.
    - filename: Name of the FASTA file (default is 'peptides.fasta').

    Returns:
    - Path to the created FASTA file.
    """
    records = [SeqRecord(Seq(peptide), id=f"{peptide}", description="") for peptide in peptides]
    fasta_path = os.path.join(output_dir, filename)
    SeqIO.write(records, fasta_path, "fasta")
    return fasta_path
