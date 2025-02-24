import os
import shutil
import subprocess
import logging
from functools import partial
from multiprocessing import get_context
from typing import List, Dict, Optional


def dock_peptide(
    peptide_sequence: str,
    gpu_id: str,
    target_sequence: str,
    target_structure: str,
    output_dir: str,
    num_models: int,
    num_recycles: int,
    recycle_early_stop_tolerance: float,
    amber: bool,
    num_relax: int,
) -> Dict:
    """
    Dock a single peptide to the target structure using ColabFold.
    This function returns a dictionary summarizing the result status for the peptide.
    """

    logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")

    target_name = os.path.basename(target_structure).replace(
        ".pdb", ""
    )

    # Create a sub-directory for this peptide’s results
    peptide_output_dir = os.path.join(output_dir, f"{target_name}_{peptide_sequence}")
    os.makedirs(peptide_output_dir, exist_ok=True)

    # Create FASTA file containing the target and peptide sequences
    combined_fasta_path = os.path.join(peptide_output_dir, f"input_{peptide_sequence}.fasta")
    with open(combined_fasta_path, "w") as f:
        f.write(f">{target_name}_{peptide_sequence}\n{target_sequence}:{peptide_sequence}\n")

    # Copy the target structure to the peptide output directory
    target_copy_path = os.path.join(peptide_output_dir, os.path.basename(target_structure))
    shutil.copy2(target_structure, target_copy_path)

    # Prepare the ColabFold command
    command = [
        "colabfold_batch",
        str(combined_fasta_path),
        str(peptide_output_dir),
        "--model-type", "alphafold2_multimer_v3",
        "--msa-mode",
        "single_sequence",
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycles),
        "--recycle-early-stop-tolerance", str(recycle_early_stop_tolerance),
        "--num-relax", str(num_relax),
        "--pair-mode", "unpaired",
        "--pair-strategy", "greedy",
        "--templates",
        "--custom-template-path", str(peptide_output_dir),
        "--rank", "iptm", 
        "--zip"
    ]

    if amber:
        command.append("--amber")

    # Set environment to use a specific GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.pop("MPLBACKEND", None)
    # Run docking
    try:        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
        )
        #for line in iter(process.stdout.readline, ""):
        #    logging.info(line.rstrip())
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=command
            )
        logging.info(
            f"Docking completed successfully for {peptide_sequence} on GPU {gpu_id}."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred during docking of {peptide_sequence}: {e}")




def dock_peptides(
    peptides: list,
    target_structure: str,
    target_sequence: str,
    num_models: int,
    num_recycles: int,
    recycle_early_stop_tolerance: float,
    amber: bool,
    num_relax: int,
    output_dir: str,
    gpu_ids: Optional[List[str]] = None,
    num_processes: Optional[int] = None,
) -> Dict:
    """
    Dock multiple peptides to a target structure using ColabFold.
    
    """
    if gpu_ids is None:
        gpu_ids = ["0"]  # default to GPU 0 if none provided

    os.makedirs(output_dir, exist_ok=True)

    # Decide how many processes to run in parallel
    if num_processes is None:
        num_processes = len(gpu_ids)
    num_processes = max(1, num_processes)

    # Prepare partial function for starmap
    dock_peptide_partial = partial(
        dock_peptide,
        target_sequence=target_sequence,
        target_structure=target_structure,
        output_dir=output_dir,
        num_models=num_models,
        num_recycles=num_recycles,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,

        amber=amber,
        num_relax=num_relax,
    )

    # Assign each peptide to a GPU in a round-robin fashion
    peptide_gpu_pairs = [
        (peptide, gpu_ids[i % len(gpu_ids)]) for i, peptide in enumerate(peptides)
    ]

    logging.info(f"Starting docking on {num_processes} process(es)...")

    # Run the docking in parallel
    context = get_context("spawn")
    with context.Pool(processes=num_processes) as pool:
        pool.starmap(dock_peptide_partial, peptide_gpu_pairs)

