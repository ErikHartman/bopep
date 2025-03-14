import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ScoresToObjective:
    """
    Takes a dict of scores and computes a scalarized objective that is used for optimization.
    """

    def __init__(self):
        pass

    def create_bopep_objective(self, scores: dict, objective_weights: dict = None):
        """
        Given a dictionary of raw scores for each peptide, scales and scalarizes the objectives.

        Parameters:
            scores: dict
                A mapping of peptide -> {objective_name: value, ..., "is_proximate": bool}
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

        # Exclude the "is_proximate" flag from scaling
        sample_keys = [k for k in scores[peptides[0]].keys() if k != "is_proximate"]

        # Use default weight 1 for each objective if no weights are provided.
        if objective_weights is None:
            objective_weights = {key: 1 for key in sample_keys}

        objectives_dict_list = []
        is_proximate_list = []

        # Define which objective keys need to be inverted. These are scores in which smaller is better.
        invert_keys = {"rosetta_score", "interface_dG"}

        # Build a list of objective dicts for each peptide
        for peptide in peptides:
            peptide_scores = scores[peptide]
            is_proximate_list.append(peptide_scores.get("is_proximate", True))
            obj_dict = {}
            for key in sample_keys:
                # Fetch the score with default 0
                value = peptide_scores.get(key, 0)
                # Invert if necessary
                if key in invert_keys:
                    value = -value
                obj_dict[key] = value
            objectives_dict_list.append(obj_dict)

        # Scale each objective across peptides
        for key in sample_keys:
            # Get all values for this objective as a column vector
            values = np.array(
                [obj_dict[key] for obj_dict in objectives_dict_list]
            ).reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0.1, 1))
            scaled_values = scaler.fit_transform(values).flatten()
            # Update each peptide's objective with its scaled, weighted value
            for i, obj_dict in enumerate(objectives_dict_list):
                obj_dict[key] = scaled_values[i] * objective_weights.get(key, 1)

        # Penalize peptides that are not proximate by zeroing their objectives
        for i, prox in enumerate(is_proximate_list):
            if not prox:
                for key in sample_keys:
                    objectives_dict_list[i][key] = 0

        # Sum the weighted, scaled objectives for each peptide to get a single scalar value
        scalar_objectives = {}
        for i, peptide in enumerate(peptides):
            total = sum(objectives_dict_list[i][key] for key in sample_keys)
            scalar_objectives[peptide] = total

        return scalar_objectives
