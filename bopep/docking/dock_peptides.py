import os
import shutil
import subprocess
from functools import partial
from multiprocessing import get_context
from typing import List, Optional
from bopep.docking.utils import clean_up_files, docking_folder_exists


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
) -> str:
    """
    Dock a single peptide to the target structure using ColabFold.
    Returns the directory path where the peptide's results are stored.
    """
    print(f"Docking peptide '{peptide_sequence}' on GPU {gpu_id}...")

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
        print(f"Docking completed successfully for {peptide_sequence} on GPU {gpu_id}.")

        # Clean up temporary files after successful docking
        clean_up_files(peptide_output_dir, target_copy_path, peptide_sequence)
        # Add file called "finished.txt" to indicate that docking is complete
        with open(os.path.join(peptide_output_dir, "finished.txt"), "w") as f:
            f.write("Docking finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during docking of {peptide_sequence}: {e}")

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
    overwrite_results: bool = False,
) -> List[str]:
    """
    Dock multiple peptides to a target structure using ColabFold.
    Returns a list of directories for the docked peptides.

    Filters out peptides that already have a docking result in the output directory.
    """

    # We need to filter out peptides that already have a docking result
    # if the peptide has been docked we also need to save the dir name to return it
    previously_docked_dirs = []
    peptides_to_dock = []
    for peptide in peptides:
        exists, peptide_dir = docking_folder_exists(output_dir, peptide, target_structure)
        if exists and not overwrite_results:
            previously_docked_dirs.append(peptide_dir)
        else:
            peptides_to_dock.append(peptide)

    if len(peptides_to_dock) == 0:
        return previously_docked_dirs
    else:
        print(f"Will dock {len(peptides_to_dock)} peptides...")

    if gpu_ids is None:
        gpu_ids = ["0"]  # default to GPU 0 if none provided

    os.makedirs(output_dir, exist_ok=True)

    # Decide how many processes to run in parallel
    if num_processes is None:
        num_processes = len(gpu_ids)
    num_processes = max(1, num_processes)

    print(f"Starting docking on {num_processes} process(es)...")

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
        (peptide, gpu_ids[i % len(gpu_ids)]) for i, peptide in enumerate(peptides_to_dock)
    ]

    # Run the docking in parallel and collect the output directories
    context = get_context("spawn")
    with context.Pool(processes=num_processes) as pool:
        docked_dirs = pool.starmap(dock_peptide_partial, peptide_gpu_pairs)
    
    docked_dirs += previously_docked_dirs
    return docked_dirs
