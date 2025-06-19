import os
import shutil
import subprocess
from functools import partial
from multiprocessing import get_context
from typing import List, Optional
from bopep.docking.utils import clean_up_files, docking_folder_exists
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    target_name: str = None
) -> str:
    """
    Dock a single peptide to the target structure using ColabFold.
    Returns the directory path where the peptide's results are stored.
    """
    logging.info(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")

    if not target_name:
        target_name = os.path.basename(target_structure).replace(".pdb", "")
    # Create a sub-directory for this peptide’s results
    peptide_output_dir = os.path.join(
        output_dir, f"{target_name}_{peptide_sequence}"
    )  # peptide output dir is called structure name + peptide sequence
    os.makedirs(peptide_output_dir, exist_ok=True)

    # Create FASTA file containing the target and peptide sequences
    combined_fasta_path = os.path.join(
        peptide_output_dir, f"input_{peptide_sequence}.fasta"
    )
    with open(combined_fasta_path, "w") as f:
        f.write(
            f">{target_name}_{peptide_sequence}\n{target_sequence}:{peptide_sequence}\n"
        )

    # Copy the target structure to the peptide output directory
    target_copy_path = os.path.join(
        peptide_output_dir, os.path.basename(target_structure)
    )
    shutil.copy2(target_structure, target_copy_path)

    # Prepare the ColabFold command
    command = [
        "colabfold_batch",
        str(combined_fasta_path),
        str(peptide_output_dir),
        "--model-type",
        "alphafold2_multimer_v3",
        "--msa-mode",
        "single_sequence",
        "--num-models",
        str(num_models),
        "--num-recycle",
        str(num_recycles),
        "--recycle-early-stop-tolerance",
        str(recycle_early_stop_tolerance),
        "--num-relax",
        str(num_relax),
        "--pair-mode",
        "unpaired",
        "--pair-strategy",
        "greedy",
        "--templates",
        "--custom-template-path",
        str(peptide_output_dir),
        "--rank",
        "iptm",
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
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=command
            )
        logging.info(f"Docking completed successfully for {peptide_sequence} on GPU {gpu_id}.")

        # Clean up temporary files after successful docking
        clean_up_files(peptide_output_dir, target_copy_path, peptide_sequence)
        # Add file called "finished.txt" to indicate that docking is complete
        with open(os.path.join(peptide_output_dir, "finished.txt"), "w") as f:
            f.write("Docking finished successfully.")
    except subprocess.CalledProcessError as e:
        logging.info(f"An error occurred during docking of {peptide_sequence}: {e}")

    # Return the directory containing the docked peptide results
    return peptide_output_dir


def dock_peptides_parallel(
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
    target_name: str = None,
) -> List[str]:
    """
    Dock multiple peptides to a target structure using ColabFold.
    Returns a list of directories for the docked peptides.

    Filters out peptides that already have a docking result in the output directory.
    """
    if gpu_ids is None:
        gpu_ids = ["0"]  # default to GPU 0 if none provided

    os.makedirs(output_dir, exist_ok=True)

    # Decide how many processes to run in parallel
    if num_processes is None:
        num_processes = len(gpu_ids)
    num_processes = max(1, min(num_processes, len(gpu_ids)))  # Can't have more processes than GPUs
    
    logging.info(f"Starting docking on {num_processes} process(es)...")
    
    # Group peptides by GPU - each GPU gets its own batch of peptides
    peptides_by_gpu = [[] for _ in range(len(gpu_ids))]
    for i, peptide in enumerate(peptides):
        gpu_index = i % len(gpu_ids)
        peptides_by_gpu[gpu_index].append(peptide)
    
    # Create arguments for each worker process - one process per GPU
    process_args = []
    for gpu_index, gpu_peptides in enumerate(peptides_by_gpu[:num_processes]):
        if not gpu_peptides:
            continue  # Skip empty peptide lists
        
        process_args.append((
            gpu_peptides,
            gpu_ids[gpu_index],
            target_sequence,
            target_structure,
            output_dir,
            num_models,
            num_recycles,
            recycle_early_stop_tolerance,
            amber,
            num_relax,
            target_name,
        ))
    
    # Run the GPU-specific workers in parallel
    context = get_context("spawn")
    with context.Pool(processes=num_processes) as pool:
        all_docked_dirs = pool.starmap(dock_peptides_for_gpu, process_args)
    
    # Flatten the list of lists
    return [dir_path for dirs in all_docked_dirs for dir_path in dirs]

def dock_peptides_for_gpu(
    peptides: list,
    gpu_id: str,
    target_sequence: str,
    target_structure: str,
    output_dir: str,
    num_models: int,
    num_recycles: int,
    recycle_early_stop_tolerance: float,
    amber: bool,
    num_relax: int,
    target_name: str = None,
) -> List[str]:
    """Process a batch of peptides on a specific GPU"""
    docked_dirs = []
    for peptide in peptides:
        dir_path = dock_peptide(
            peptide,
            gpu_id,
            target_sequence,
            target_structure,
            output_dir,
            num_models,
            num_recycles,
            recycle_early_stop_tolerance,
            amber,
            num_relax,
            target_name,
        )
        docked_dirs.append(dir_path)
    return docked_dirs
