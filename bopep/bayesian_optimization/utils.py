import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Union
import pyrosetta
import torch

from bopep.docking.utils import extract_sequence_from_pdb
from bopep.bayesian_optimization.acquisition_functions import available_acquistion_functions


def check_starting_index_in_pdb(pdb_file: str) -> int:
    """
    Parses PDB file and checks what the starting residue index is. 
    Some PDB files are not 0-indexed.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        The starting residue index (usually 1 for standard PDB files), 
        or None if no valid residue is found
    """
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # Look for ATOM records which define coordinates for standard residues
                if line.startswith("ATOM  "):
                    # Extract residue number from columns 23-26 (0-indexed)
                    residue_number = line[22:26].strip()
                    
                    # Try to parse as integer
                    try:
                        return int(residue_number)
                    except ValueError:
                        continue         
        # If we get here, we didn't find any valid ATOM records with residue numbers
        return None
        
    except FileNotFoundError:
        print(f"Error: PDB file {pdb_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading PDB file: {e}")
        return None

def _save_model(save_path: str, model, surrogate_model_kwargs: dict, best_hyperparams: dict):
        """
        Save the current model to a file with metadata.
        """
        save_dict = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_type": surrogate_model_kwargs["model_type"],
                "network_type": surrogate_model_kwargs["network_type"],
                "input_dim": surrogate_model_kwargs["input_dim"],
                "hyperparameters": best_hyperparams,
            },
            "model_class": model.__class__.__name__,
            "surrogate_model_kwargs": surrogate_model_kwargs,
        }

        torch.save(save_dict, save_path)
        logging.info(f"Model saved to {save_path}")
        logging.info(f"Model type: {surrogate_model_kwargs['model_type']}")
        logging.info(
            f"Network architecture: {surrogate_model_kwargs['network_type']}"
        )
        logging.info(f"Model class: {model.__class__.__name__}")


def _validate_surrogate_model_kwargs(surrogate_model_kwargs: dict):
    """Validate the surrogate model configuration."""
    valid_network_types = ["mlp", "bilstm", "bigru"]
    valid_model_types = ["nn_ensemble", "mc_dropout", "deep_evidential", "mve"]

    network_type = surrogate_model_kwargs.get("network_type")
    if network_type not in valid_network_types:
        raise ValueError(
            f"Invalid network type: {network_type}. "
            f"Must be one of: {', '.join(valid_network_types)}"
        )

    model_type = surrogate_model_kwargs.get("model_type")
    if model_type not in valid_model_types:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Must be one of: {', '.join(valid_model_types)}"
        )

def _validate_dependencies():
    """
    Check if colabfold -help is callable
    Check if output dir exists
    """
    try:
        subprocess.Popen(
            ["colabfold_batch", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except:
        raise ValueError(
            "colabfold is not callable through the command: colabfold_batch --help"
        )

    try:
        pyrosetta.init("-mute all")
    except:
        raise ValueError("Cannot initialize pyrosetta")

def _validate_args(
    schedule: List[Dict[str, Any]], n_validate: Optional[Union[int, float]]
):
    """
    Validates optimization args
    """
    if not schedule:
        raise ValueError("Schedule cannot be empty")

    for idx, phase in enumerate(schedule):
        if not isinstance(phase, dict):
            raise ValueError(f"Phase {idx} must be a dictionary")

        if "acquisition" not in phase:
            raise ValueError(f"Phase {idx} missing required key 'acquisition'")

        if phase["acquisition"] not in available_acquistion_functions:
            raise ValueError(
                f"Invalid acquisition function in phase {idx}: {phase['acquisition']}. "
                f"Must be one of: {', '.join(available_acquistion_functions)}"
            )

        if "iterations" not in phase:
            raise ValueError(f"Phase {idx} missing required key 'iterations'")

        if not isinstance(phase["iterations"], int) or phase["iterations"] <= 0:
            raise ValueError(f"Phase {idx}: iterations must be a positive integer")

    if n_validate is not None:
        if isinstance(n_validate, float) and not (0 < n_validate < 1):
            raise ValueError(
                f"When n_validate is a float, it must be between 0 and 1, got {n_validate}"
            )
        elif isinstance(n_validate, int) and n_validate <= 0:
            raise ValueError(
                f"When n_validate is an integer, it must be positive, got {n_validate}"
            )
        

def _validate_checkpoint(checkpoint_path: Path):
    """Validate checkpoint integrity."""
    required_files = [
        "checkpoint_metadata.json",
        "embeddings.pkl",
        "model.pt",
        "results/scores.csv",
        "results/objectives.csv"
    ]
    optional_files = [
        "results/model_losses.csv",
        "results/predictions.csv.gz",
        "results/acquisition.csv.gz",
        "results/hyperparameters.csv"
    ]
    
    missing = [f for f in required_files if not (checkpoint_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint incomplete. Missing required files: {missing}"
        )
    missing_optional = [f for f in optional_files if not (checkpoint_path / f).exists()]
    if missing_optional:
        logging.warning(f"Checkpoint missing optional files: {missing_optional}")


def _check_binding_site_residue_indices(
    binding_site_residue_indices,
    target_structure_path,
    assume_zero_indexed=None,
):
    """
    Checks if starting index is 0.

    If not, asks the user if the binding site residues are expected to be 0-indexed.
    Corrects for the starting index if wanted.

    Return a visualization of the residues that are selected as binding site residues.
    """
    starting_index = check_starting_index_in_pdb(target_structure_path)
    print("PDB chain starts at residue", starting_index)
    protein_sequence = extract_sequence_from_pdb(target_structure_path)
    if binding_site_residue_indices is None:
        return None

    if starting_index != 0:
        if assume_zero_indexed is None:
            print(
                f"\n\nStarting index is {starting_index}. Are the provided binding site residues 0-indexed?"
            )
            answer = input("y/n: ")
        else:
            if assume_zero_indexed is True:
                print(
                    f"\n\nStarting index is {starting_index}. Assuming binding site residues are 0-indexed."
                )
                answer = "y"
            else:
                print(
                    f"\n\nStarting index is {starting_index}. Assuming binding site residues are 1-indexed."
                )
                answer = "n"
        if answer == "y":
            binding_site_residue_indices = [
                residue - starting_index for residue in binding_site_residue_indices
            ]

    print("\nBinding Site Residues Visualization:")
    print("=" * 60)
    print(f"Full sequence length: {len(protein_sequence)}")
    print(f"Selected binding site residues: {binding_site_residue_indices}")
    print("-" * 60)

    binding_site_residue_indices = sorted(binding_site_residue_indices)
    context_size = 5

    for residue_idx in binding_site_residue_indices:
        if residue_idx < 0 or residue_idx >= len(protein_sequence):
            print(f"Warning: Residue index {residue_idx} out of range")
            continue

        # Calculate start and end positions for context
        start = max(0, residue_idx - context_size)
        end = min(len(protein_sequence), residue_idx + context_size + 1)

        positions = list(range(start + starting_index, end + starting_index))
        vis_seq = list(protein_sequence[start:end])

        # Mark the selected residue
        rel_pos = residue_idx - start
        if 0 <= rel_pos < len(vis_seq):
            vis_seq[rel_pos] = f"[{vis_seq[rel_pos]}]"

        print(
            f"Residue {residue_idx + starting_index} ({protein_sequence[residue_idx]}):"
        )
        print("Position:" + " ".join(f"{pos:3d}" for pos in positions))
        print("Sequence: " + " ".join(f"{aa:3s}" for aa in vis_seq))
        print("-" * 60)

    print("=" * 60)
    

    # increment binding site residues by 1 since alphafold pdbs start at 1
    binding_site_residue_indices = [
        residue + 1 for residue in binding_site_residue_indices
    ]
    print(
        f"The internally stored binding site residues are: {binding_site_residue_indices} (1-indexed)"
    )
    return binding_site_residue_indices

if __name__ == "__main__":
    # Example usage
    pdb_path = "/home/er8813ha/bopep/data/4glf.pdb"
    binding_site_indices = [23, 42]
    _check_binding_site_residue_indices(binding_site_indices, pdb_path)
    # This will print the binding site residues and their context in the sequence