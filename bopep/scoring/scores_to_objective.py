from typing import Callable
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
    
def benchmark_objective(scores: dict) -> dict:
    """
    regression_equation_string = "(3.374078 - 4.3519845*interface_dG)*(distance_score + instability_index) + 5.4904785"
    classification_equation_string = "-iptm*(peptide_pae - 1.2075188)"

    final function = regression_equation_string * classification_equation_string

    scores contains a dict of peptides: {score_name: value, ..., "is_in_binding_site": bool}

    The mins and maxes should be used to do min-max scaling.
    """
    classification_peptide_pae_min, classification_peptide_pae_max = 3.430482456140357,28.19067316394025
    classification_iptm_min, classification_iptm_max = 0.14,0.95
    classification_constant_1 = 1.2075188

    regression_delta_G_min, regression_delta_G_max = -98.70550256759547, 17.115022472542762
    regression_instability_index_min, regression_instability_index_max = -12.405, 237.25
    regression_distance_score_min, regression_distance_score_max = 5.528944453683353, 7.361734707761549
    regression_constant_1 = 3.374078
    regression_contsant_2 = 4.3519845
    regression_constant_3 = 5.4904785

    scalar_objectives = {}
    for peptide, peptide_scores in scores.items():
        interface_dG = peptide_scores.get("interface_dG", 0)
        distance_score = peptide_scores.get("distance_score", 0)
        instability_index = peptide_scores.get("instability_index", 0)
        peptide_pae = peptide_scores.get("peptide_pae", 0)
        iptm = peptide_scores.get("iptm", 0)
        
        scaled_interface_dG = (interface_dG - regression_delta_G_min) / (regression_delta_G_max - regression_delta_G_min)
        scaled_distance_score = (distance_score - regression_distance_score_min) / (regression_distance_score_max - regression_distance_score_min)
        scaled_instability_index = (instability_index - regression_instability_index_min) / (regression_instability_index_max - regression_instability_index_min)
        scaled_peptide_pae = (peptide_pae - classification_peptide_pae_min) / (classification_peptide_pae_max - classification_peptide_pae_min)
        scaled_iptm = (iptm - classification_iptm_min) / (classification_iptm_max - classification_iptm_min)

        classification_value = -scaled_iptm * (scaled_peptide_pae - classification_constant_1)
        regression_value = (regression_constant_1 - regression_contsant_2 * scaled_interface_dG) * (scaled_distance_score + scaled_instability_index) + regression_constant_3
        objective_value = regression_value * classification_value
        if peptide_scores["in_binding_site"]:
            scalar_objectives[peptide] = objective_value
        else:
            scalar_objectives[peptide] = 0

    return scalar_objectives


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