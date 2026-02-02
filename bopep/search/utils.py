import logging
import subprocess
from typing import Any, Dict, List, Optional, Union
import torch

from bopep.bayes.acquisition import available_acquisition_functions


def print_leaderboard(
    objectives: Dict[str, Any], 
    iteration: int = None,
    print_n: int = 5, 
    objective_directions: Dict[str, str] = None,
    iteration_label: str = "Iteration",
    use_logging: bool = False
):
    """
    Print leaderboard for both single and multi-objective cases.
    
    Args:
        objectives: Dictionary mapping sequences to objective values (float) or 
                   objective dictionaries (for multi-objective)
        iteration: Current iteration/generation number (optional)
        print_n: Number of top sequences to display
        objective_directions: For multi-objective, specify "max" or "min" for each objective.
                            Defaults to "max" if not specified.
        iteration_label: Label for iteration ("Iteration", "Generation", etc.)
        use_logging: If True, use logging.info instead of print
    """
    if not objectives:
        return
    
    output_fn = logging.info if use_logging else print
    
    if iteration is not None:
        output_fn(f"{iteration_label} {iteration} leaderboard:")
    else:
        output_fn("Leaderboard:")
    
    sample_obj = next(iter(objectives.values()))
    if isinstance(sample_obj, dict):
        # Multi-objective case
        obj_names = list(sample_obj.keys())
        output_fn(f"Top {print_n} sequences (multiobjective):")
        
        for obj_name in obj_names:
            output_fn(f"\n--- {obj_name} ---")
            
            # Sort by direction
            if objective_directions and obj_name in objective_directions:
                reverse_sort = objective_directions[obj_name] == "max"
            else:
                reverse_sort = True  # Default to maximization
            
            sorted_sequences = sorted(
                objectives.items(), 
                key=lambda x: x[1][obj_name], 
                reverse=reverse_sort
            )[:print_n]
            output_fn(f"{'Sequence':<20} | {obj_name:<15}")
            output_fn("-" * 40)
            for sequence, obj_dict in sorted_sequences:
                output_fn(f"{sequence:<20} | {obj_dict[obj_name]:<15.4f}")
    else:
        # Single objective case
        sorted_leaderboard = sorted(
            objectives.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:print_n]
        output_fn(f"{'Sequence':<20} | {'Objective':<10}")
        output_fn("-" * 60)
        for rank, (seq, obj) in enumerate(sorted_leaderboard, start=1):
            output_fn(f"  {rank}. {seq} - Objective: {obj:.4f}")


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

        if phase["acquisition"] not in available_acquisition_functions:
            raise ValueError(
                f"Invalid acquisition function in phase {idx}: {phase['acquisition']}. "
                f"Must be one of: {', '.join(available_acquisition_functions)}"
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
        

