import logging
import subprocess
from typing import Any, Dict, List, Optional, Union
import pyrosetta
import torch

from bopep.bayesian_optimization.acquisition_functions import available_acquistion_functions

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
        

