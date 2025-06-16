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
    regression_equation_string = "-rosetta_score - (distance_score + 3.0185924)*(interface_dG - 2.4250882)"
    classification_equation_string = "-iptm*(peptide_pae - 1.2075188)"

    final function = regression_equation_string * classification_equation_string

    scores contains a dict of peptides: {score_name: value, ..., "in_binding_site": bool}

    The mins and maxes should be used to do min-max scaling.
    """
    classification_peptide_pae_min, classification_peptide_pae_max = 3.430482456140357,28.19067316394025
    classification_iptm_min, classification_iptm_max = 0.14,0.95
    classification_constant_1 = 1.2075188

    regression_delta_G_min, regression_delta_G_max = -98.70550256759547,17.115022472542762
    rosetta_score_min, rosetta_score_max = -1021.077541270006,451.3142280816484
    regression_distance_score_min, regression_distance_score_max = 5.528944453683353,7.361734707761549
    regression_constant_1 = 3.0185924
    regression_contsant_2 = 2.4250882

    scalar_objectives = {}
    for peptide, peptide_scores in scores.items():
        interface_dG = peptide_scores["interface_dG"]
        distance_score = peptide_scores["distance_score"]
        rosetta_score = peptide_scores["rosetta_score"]
        peptide_pae = peptide_scores["peptide_pae"]
        iptm = peptide_scores["iptm"]
        
        scaled_interface_dG = (interface_dG - regression_delta_G_min) / (regression_delta_G_max - regression_delta_G_min)
        scaled_distance_score = (distance_score - regression_distance_score_min) / (regression_distance_score_max - regression_distance_score_min)
        scaled_rosetta_score = (rosetta_score - rosetta_score_min) / (rosetta_score_max - rosetta_score_min)
        scaled_peptide_pae = (peptide_pae - classification_peptide_pae_min) / (classification_peptide_pae_max - classification_peptide_pae_min)
        scaled_iptm = (iptm - classification_iptm_min) / (classification_iptm_max - classification_iptm_min)

        classification_value = -scaled_iptm * (scaled_peptide_pae - classification_constant_1)
        regression_value = -scaled_rosetta_score - (scaled_distance_score + regression_constant_1) * (scaled_interface_dG - regression_contsant_2)
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


if __name__ == "__main__":
    # Example usage
    #peptide,objective,rosetta_score,interface_dG,distance_score,iptm,peptide_pae,in_binding_site,fraction_in_binding_site
    #LKNPDDPDMVD,0.21621160762107358,-352.43062003747855,-10.933749597646795,6.070098449822542,0.25,25.55502085070889,True,1.0
    example_scores = {
        "KSLLQQLLTE": {
            "rosetta_score": -636.1152754278565,
            "interface_dG": -44.35348584226472,
            "iptm": 0.92,
            "peptide_pae": 7.250308300395252,
            "distance_score": 7.110018269096763,
            "in_binding_site": True
        },
        "LKNPDDPDMVD": {
            "rosetta_score": -352.43062003747855,
            "interface_dG": -10.933749597646795,
            "iptm": 0.25,
            "peptide_pae": 25.55502085070889,
            "distance_score": 6.070098449822542,
            "in_binding_site": True
        },
    }

    objective = ScoresToObjective()
    result = objective.create_objective(example_scores, benchmark_objective)
    print(result)