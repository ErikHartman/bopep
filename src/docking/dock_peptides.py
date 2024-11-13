import os
import subprocess
import shutil
import logging
from typing import List
from multiprocessing import Pool
from functools import partial
from Bio import SeqIO
import zipfile
import glob
from multiprocessing import get_context
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def dock_peptide(
    peptide_record,
    gpu_id,
    target_sequence: str,
    target_structure: str,
    output_dir: str,
    num_recycles: int,
    msa_mode: str,
    model_type: str,
    num_relax: int,
    num_models: int,
    amber: bool,
    recycle_early_stop_tolerance: float,
) -> None:
    peptide_name = peptide_record.id
    peptide_sequence = str(peptide_record.seq)
    logging.info(f"Docking peptide {peptide_name} on GPU {gpu_id}...")

    # Create peptide-specific output directory
    peptide_output_dir = os.path.join(
        output_dir,
        f"{os.path.basename(target_structure).replace('.pdb', '')}_{peptide_name}",
    )
    os.makedirs(peptide_output_dir, exist_ok=True)

    # Create combined FASTA content
    combined_fasta = os.path.join(
        peptide_output_dir, f"combined_input_{peptide_name}.fasta"
    )
    with open(combined_fasta, "w") as combined_file:
        combined_file.write(
            f">{os.path.basename(target_structure).replace('.pdb', '')}:{peptide_name}\n"
        )
        combined_file.write(f"{target_sequence}:{peptide_sequence}")

    # Copy target_structure into the peptide_output_dir
    target_structure_copy = os.path.join(
        peptide_output_dir, os.path.basename(target_structure)
    )
    shutil.copy2(target_structure, target_structure_copy)

    # Prepare the docking command
    command: List[str] = [
        "colabfold_batch",
        combined_fasta,
        peptide_output_dir,
        "--templates",
        "--custom-template-path",
        peptide_output_dir,
        "--model-type",
        model_type,
        "--msa-mode",
        msa_mode,
        "--pair-mode",
        "unpaired",
        "--pair-strategy",
        "greedy",
        "--num-recycle",
        str(num_recycles),
        "--num-models",
        str(num_models),
        "--num-relax",
        str(num_relax),
        "--recycle-early-stop-tolerance",
        str(recycle_early_stop_tolerance),  # ensures that we run all recycles
        "--rank",
        "iptm",
        "--overwrite-existing-results",
        "--zip",
    ]
    if amber:
        command.append("--amber")

    # Set up environment to specify which GPU to use
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set specific GPU for this process

    # Run docking subprocess and handle errors
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
            f"Docking completed successfully for {peptide_name} on GPU {gpu_id}."
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"An error occurred during docking of {peptide_name}: {e}")
    finally:
        # Clean up files that are no longer needed
        _clean_up_files(peptide_output_dir, target_structure_copy, peptide_name)
        _clean_up_zip_files(peptide_output_dir)


def dock_peptides(
    multi_peptide_fasta: str,
    target_structure: str,
    output_dir: str = "dock_afm/output_docking",
    num_recycles: int = 1,
    msa_mode: str = "single_sequence",
    model_type: str = "alphafold2_multimer_v3",
    num_relax: int = 0,
    num_models: int = 1,
    recycle_early_stop_tolerance: float = 0,
    amber: bool = False,
    target_chain: str = "A",
    num_processes: int = None,
    gpu_ids: List[str] = None,
    overwrite_results: bool = True,
) -> None:
    """
    Dock peptides from a multi-sequence FASTA file to a target structure using ColabFold.
    This version distributes jobs across multiple GPUs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate input files
    _validate_input_files(multi_peptide_fasta, target_structure)

    # Extract target sequence
    target_sequence = extract_sequence_from_pdb(target_structure, chain_id=target_chain)

    # Read multi-sequence peptide FASTA
    peptide_records = list(SeqIO.parse(multi_peptide_fasta, "fasta"))

    # Filter out peptides that have already been docked
    peptides_to_process = _filter_peptides(
        peptide_records, target_structure, output_dir, overwrite_results
    )

    if not peptides_to_process:
        logging.info("All peptides have already been processed.")
        return
    
    logging.info(f"Will dock {len(peptides_to_process)} peptide(s).")

    # Set the number of processes for parallel execution
    num_processes = num_processes or len(gpu_ids) or 1
    num_processes = min(len(peptides_to_process), num_processes)
    gpu_ids = gpu_ids or ["0"]  # Default GPU 0 if none provided

    # Assign GPU IDs to peptide records in a round-robin fashion
    logging.info(f"Running docking with {num_processes} GPUs")
    peptide_records_with_gpus = [
        (peptide_record, gpu_ids[idx % len(gpu_ids)])
        for idx, peptide_record in enumerate(peptides_to_process)
    ]

    # Use multiprocessing Pool to run docking in parallel
    dock_peptide_partial = partial(
        dock_peptide,
        target_sequence=target_sequence,
        target_structure=target_structure,
        output_dir=output_dir,
        num_recycles=num_recycles,
        msa_mode=msa_mode,
        model_type=model_type,
        num_relax=num_relax,
        num_models=num_models,
        amber=amber,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
    )
    
    context = get_context("spawn")
    with context.Pool(processes=num_processes) as pool:
        pool.starmap(dock_peptide_partial, peptide_records_with_gpus)

    logging.info(f"All docking jobs completed. Results are stored in: {output_dir}")


def _validate_input_files(multi_peptide_fasta, target_structure):
    if not os.path.isfile(target_structure):
        raise FileNotFoundError(
            f"Target structure PDB file not found: {target_structure}"
        )
    if not os.path.isfile(multi_peptide_fasta):
        raise FileNotFoundError(f"Peptide FASTA file not found: {multi_peptide_fasta}")


def _filter_peptides(peptide_records, target_structure, output_dir, overwrite_results):
    """
    Filters the peptide records by checking whether the results have already been processed. 
    If results are not found in the output directory, the peptide will be reprocessed.
    """
    peptides_to_process = []
    target_base = os.path.basename(target_structure).split(".")[0]

    for peptide_record in peptide_records:
        output_subdir = os.path.join(output_dir, f"{target_base}_{peptide_record.id}")

        if os.path.exists(output_subdir):
            # Check if the directory contains a .result.zip file
            zip_file_pattern = os.path.join(output_subdir, "*.result.zip")
            zip_files = glob.glob(zip_file_pattern)

            if zip_files:
                if overwrite_results:
                    logging.info(f"{peptide_record.id} exists, but it is now being overwritten.")
                else:
                    logging.info(f"Skipping {peptide_record.id}, output directory already exists with results.")
                    continue  # Skip processing if results already exist and overwrite is False
            else:
                # If the directory exists but no .result.zip file is found, we need to reprocess
                logging.info(f"No valid results found for {peptide_record.id}, reprocessing.")
        else:
            # Directory does not exist, add the peptide for processing
            logging.info(f"Processing {peptide_record.id}, no existing results found.")

        peptides_to_process.append(peptide_record)

    return peptides_to_process

def _clean_up_files(peptide_output_dir, target_structure_copy, peptide_name):
    try:
        # Remove copied PDB file
        os.remove(target_structure_copy)

        # Remove unnecessary files generated during docking
        for file in os.listdir(peptide_output_dir):
            if (
                file.startswith("pdb70")
                or file.endswith(".cif")
                or file == "cite.bibtex"
                or file.startswith("combined_input")
            ):
                os.remove(os.path.join(peptide_output_dir, file))
    except OSError as e:
        logging.warning(f"Error deleting temporary files for {peptide_name}: {e}")

def _clean_up_zip_files(peptide_output_dir):
    """
    Remove unnecessary files from the zipped results file to reduce its size.

    Parameters:
    - zip_file_path: Path to the zip file to clean.
    """
    zip_file_path = None
    for file in os.listdir(peptide_output_dir):
        if file.endswith('.result.zip') and os.path.isfile(os.path.join(peptide_output_dir, file)):
            zip_file_path = os.path.join(peptide_output_dir, file)
            break

    if zip_file_path is None:
        print(f"No zip file ending with '.result.zip' found in {peptide_output_dir}")
        return
    
    temp_extract_dir = os.path.join(peptide_output_dir, "temp_zip_extract")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the files to a temporary directory
        zip_ref.extractall(temp_extract_dir)

    # Remove unwanted files from the extracted directory
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            if file.endswith(('.png', '.bibtex', 'config.json')):
                os.remove(os.path.join(root, file))

    # Create a new cleaned zip file
    cleaned_zip_path = zip_file_path.replace('.zip', '_cleaned.zip')
    with zipfile.ZipFile(cleaned_zip_path, 'w') as zip_ref:
        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_extract_dir)
                zip_ref.write(file_path, arcname)

    # Remove the old zip file
    os.remove(zip_file_path)

    # Rename the cleaned zip file to the original name
    os.rename(cleaned_zip_path, zip_file_path)

    # Clean up the temporary extraction directory
    for root, dirs, files in os.walk(temp_extract_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_extract_dir)


def extract_sequence_from_pdb(pdb_file, chain_id="A"):
    """
    Extracts the sequence from a PDB file for a given chain.

    Parameters:
    - pdb_file: Path to the PDB file.
    - chain_id: The chain ID to extract the sequence from (default is 'A').

    Returns:
    - Extracted sequence as a string.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_file)
    aa_dict = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
        "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
        "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
        "TRP": "W", "TYR": "Y", "SEC": "U", "PYL": "O"
    }
    sequence = "".join(
        aa_dict.get(residue.get_resname(), "X")
        for model in structure
        for chain in model
        if chain.id == chain_id
        for residue in chain
        if residue.id[0] == " "
    )
    return sequence