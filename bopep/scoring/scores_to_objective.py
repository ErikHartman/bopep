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
                A mapping of sequence -> {objective_name: value, ..., "is_in_binding_site": bool}
            objective_function: callable
                The function to use for objective calculation, defaults to bopep_objective_v1
            **kwargs:
                Additional arguments to pass to the objective function (e.g., objective_weights, invert_keys)

        Returns:
            dict: A mapping of sequence -> scalarized objective value
        """
        if objective_function is None:
            objective_function = bopep_objective_v1
        return objective_function(scores, **kwargs)
    

def mo_objective(scores: dict) -> dict:
    objectives = {}
    for pep, ps in scores.items():
        objectives[pep] = {"interface_dG": ps["interface_dG"], "iptm": ps["iptm"]}
    return objectives

def bopep_objective_v1(scores: dict) -> dict:
    """
    regression_equation_string = "-rosetta_score - (distance_score + 3.0185924)*(interface_dG - 2.4250882)"
    classification_equation_string = "-iptm*(sequence_pae - 1.2075188)"

    final function = regression_equation_string * classification_equation_string

    scores contains a dict of sequences: {score_name: value, ..., "in_binding_site": bool}
    """
    classification_sequence_pae_min, classification_sequence_pae_max = 3.430482456140357, 28.19067316394025
    classification_iptm_min,       classification_iptm_max       = 0.14,                0.95
    classification_constant_1 = 1.2075188

    regression_delta_G_min, regression_delta_G_max       = -98.70550256759547, 17.115022472542762
    rosetta_score_min,      rosetta_score_max            = -1021.077541270006, 451.3142280816484
    regression_distance_score_min, regression_distance_score_max = 5.528944453683353, 7.361734707761549
    regression_constant_1 = 3.0185924
    regression_constant_2 = 2.4250882

    objectives = {}
    for pep, ps in scores.items():
        # scaling each score to a [0, 1] range. Clipping if outside the range.
        sdG   = np.clip((ps["interface_dG"]   - regression_delta_G_min)      / (regression_delta_G_max      - regression_delta_G_min), 0,1)
        sdist = np.clip((ps["distance_score"] - regression_distance_score_min) / (regression_distance_score_max - regression_distance_score_min), 0,1)
        sros  = np.clip((ps["rosetta_score"]  - rosetta_score_min)           / (rosetta_score_max           - rosetta_score_min), 0,1)
        spae  = np.clip((ps["sequence_pae"]    - classification_sequence_pae_min) / (classification_sequence_pae_max - classification_sequence_pae_min), 0,1)
        siptm = np.clip((ps["iptm"]           - classification_iptm_min)     / (classification_iptm_max     - classification_iptm_min), 0,1)

        
        classification_value = - siptm * (spae - classification_constant_1)  # classification value, should be between 0 and 1
        regression_value     = - sros  - (sdist + regression_constant_1) * (sdG - regression_constant_2) # regression value, should be > 0
        value = regression_value * classification_value # final objective value, should be >= 0

        if ps["in_binding_site"]:
            objectives[pep] = value
        else:
            objectives[pep] = -1

    return objectives


def bopep_objective_v0(
    scores: dict, objective_weights: dict = None, invert_keys: set = None
) -> dict:
    """
    Given a dictionary of raw scores for each sequence, scales and scalarizes the objectives.

    Parameters:
        scores: dict
            A mapping of sequence -> {objective_name: value, ..., "is_in_binding_site": bool}
        objective_weights: dict (optional)
            A mapping of objective names (e.g., "iptm_score", "interface_sasa", etc.) to their weights.
            If None, defaults to weight 1 for each objective.

    Returns:
        scalar_objectives: dict
            A mapping of sequence -> scalarized (and weighted) objective value.
    """
    sequences = list(scores.keys())
    if not sequences:
        return {}

    # Exclude the "is_in_binding_site" flag from scaling
    sample_keys = [k for k in scores[sequences[0]].keys() if k != "in_binding_site"]

    # Use default weight 1 for each objective if no weights are provided.
    if objective_weights is None:
        objective_weights = {key: 1 for key in sample_keys}

    objectives_dict_list = []
    in_binding_site = []

    # Define which objective keys need to be inverted. These are scores in which smaller is better.
    if not invert_keys:
        invert_keys = {"rosetta_score", "interface_dG"}

    # Build a list of objective dicts for each sequence
    for sequence in sequences:
        sequence_scores = scores[sequence]
        in_binding_site.append(sequence_scores.get("in_binding_site", True))
        obj_dict = {}
        for key in sample_keys:
            value = sequence_scores.get(key, 0)
            if key in invert_keys:
                value = -value  # invert if necessary
            obj_dict[key] = value
        objectives_dict_list.append(obj_dict)

    # Scale each objective across sequences
    for key in sample_keys:
        # Get all values for this objective as a column vector
        values = np.array([obj_dict[key] for obj_dict in objectives_dict_list]).reshape(
            -1, 1
        )
        scaler = MinMaxScaler(feature_range=(0.1, 1))
        scaled_values = scaler.fit_transform(values).flatten()
        # Update each sequence's objective with its scaled, weighted value
        for i, obj_dict in enumerate(objectives_dict_list):
            obj_dict[key] = scaled_values[i] * objective_weights.get(key, 1)

    # Penalize sequences that are not proximate by zeroing their objectives
    for i, prox in enumerate(in_binding_site):
        if not prox:
            for key in sample_keys:
                objectives_dict_list[i][key] = 0

    # Sum the weighted, scaled objectives for each sequence to get a single scalar value
    scalar_objectives = {}
    for i, sequence in enumerate(sequences):
        total = sum(objectives_dict_list[i][key] for key in sample_keys)
        scalar_objectives[sequence] = total

    return scalar_objectives


if __name__ == "__main__":
    # Example usage
    #sequence,objective,rosetta_score,interface_dG,distance_score,iptm,sequence_pae,in_binding_site,fraction_in_binding_site
    #LKNPDDPDMVD,0.21621160762107358,-352.43062003747855,-10.933749597646795,6.070098449822542,0.25,25.55502085070889,True,1.0
    example_scores = {
        "KSLLQQLLTE": {
            "rosetta_score": -574,
            "interface_dG": -43,
            "iptm": 0.86,
            "sequence_pae": 8.39,
            "distance_score": 6.4,
            "in_binding_site": True
        },
        "LKNPDDPDMVD": {
            "rosetta_score": -352.43062003747855,
            "interface_dG": -10.933749597646795,
            "iptm": 0.25,
            "sequence_pae": 25.55502085070889,
            "distance_score": 6.070098449822542,
            "in_binding_site": True
        },
    }

    objective = ScoresToObjective()
    result = objective.create_objective(example_scores, bopep_objective_v1)
    print(result)