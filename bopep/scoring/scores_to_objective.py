from typing import Callable
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def bopep_objective(
    scores: dict, objective_weights: dict = None, invert_keys: set = None
) -> dict:
    """
    Given a dictionary of raw scores for each peptide, scales and scalarizes the objectives.

    Parameters:
        scores: dict
            A mapping of peptide -> {objective_name: value, ..., "is_in_binding_site": bool}
        objective_weights: dict (optional)
            A mapping of objective names (e.g., "iptm_score", "interface_sasa", etc.) to their weights.
            If None, defaults to weight 1 for each objective.

    Returns:
        scalar_objectives: dict
            A mapping of peptide -> scalarized (and weighted) objective value.
    """
    peptides = list(scores.keys())
    if not peptides:
        return {}

    # Exclude the "is_in_binding_site" flag from scaling
    sample_keys = [k for k in scores[peptides[0]].keys() if k != "in_binding_site"]

    # Use default weight 1 for each objective if no weights are provided.
    if objective_weights is None:
        objective_weights = {key: 1 for key in sample_keys}

    objectives_dict_list = []
    in_binding_site = []

    # Define which objective keys need to be inverted. These are scores in which smaller is better.
    if not invert_keys:
        invert_keys = {"rosetta_score", "interface_dG"}

    # Build a list of objective dicts for each peptide
    for peptide in peptides:
        peptide_scores = scores[peptide]
        in_binding_site.append(peptide_scores.get("in_binding_site", True))
        obj_dict = {}
        for key in sample_keys:
            value = peptide_scores.get(key, 0)
            if key in invert_keys:
                value = -value  # invert if necessary
            obj_dict[key] = value
        objectives_dict_list.append(obj_dict)

    # Scale each objective across peptides
    for key in sample_keys:
        # Get all values for this objective as a column vector
        values = np.array([obj_dict[key] for obj_dict in objectives_dict_list]).reshape(
            -1, 1
        )
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        scaled_values = scaler.fit_transform(values).flatten()
        # Update each peptide's objective with its scaled, weighted value
        for i, obj_dict in enumerate(objectives_dict_list):
            obj_dict[key] = scaled_values[i] * objective_weights.get(key, 1)

    # Penalize peptides that are not proximate by zeroing their objectives
    for i, prox in enumerate(in_binding_site):
        if not prox:
            for key in sample_keys:
                objectives_dict_list[i][key] = 0

    # Sum the weighted, scaled objectives for each peptide to get a single scalar value
    scalar_objectives = {}
    for i, peptide in enumerate(peptides):
        total = sum(objectives_dict_list[i][key] for key in sample_keys)
        scalar_objectives[peptide] = total

    return scalar_objectives


class ScoresToObjective:
    """
    Takes a dict of scores and computes a scalarized objective that is used for optimization.
    """

    def __init__(self):
        pass

    def create_objective(
        self, scores: dict, objective_function: Callable = None, **kwargs: dict
    ) -> dict:
        """
        Takes an objective function (callable) and scores (dict) and returns a scalarized objective.

        Parameters:
            scores: dict
                A mapping of peptide -> {objective_name: value, ..., "is_in_binding_site": bool}
            objective_function: callable
                The function to use for objective calculation, defaults to bopep_objective
            **kwargs:
                Additional arguments to pass to the objective function (e.g., objective_weights, invert_keys)

        Returns:
            dict: A mapping of peptide -> scalarized objective value
        """
        if objective_function is None:
            objective_function = bopep_objective
        return objective_function(scores, **kwargs)
