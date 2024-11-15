import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_data(peptide_embeddings, scores, objective_weights, feature_range=(0.1, 1)):
    """
    Prepare the data for model training.

    Parameters:
    - peptide_embeddings: dict mapping peptides to their embeddings
    - scores: dict mapping peptides to their scores (raw objectives)
    - objective_weights: dict mapping objective names to their weights

    Returns:
    - X_scaled: numpy array of scaled feature vectors
    - y: numpy array of scalarized target scores
    - X_scaler: StandardScaler object used to scale X
    """

    # Initialize lists to collect data
    peptides = list(scores.keys())
    objectives_dict_list = []
    is_proximate_list = []

    # Step 1: Collect raw objective values and is_proximate flags
    for peptide in peptides:
        peptide_scores = scores[peptide]

        objectives = {
            "iptm_score": peptide_scores["iptm_score"],
            "interface_sasa": -peptide_scores["interface_sasa"],  # Inverted
            "rosetta_score": -peptide_scores["rosetta_score"],  # Inverted
            "interface_dG": -peptide_scores["interface_dG"],  # Inverted
            "interface_delta_hbond_unsat": -peptide_scores[
                "interface_delta_hbond_unsat"
            ],  # Inverted
            "packstat": -peptide_scores["packstat"],  # Inverted
        }

        # Collect the objectives and is_proximate flag
        objectives_dict_list.append(objectives)
        is_proximate_list.append(peptide_scores["is_proximate"])

    # Step 2: Scale each objective to [0.1, 1] using MinMaxScaler
    for obj_name in objective_weights.keys():
        obj_values = np.array(
            [obj_dict[obj_name] for obj_dict in objectives_dict_list]
        ).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_values = scaler.fit_transform(obj_values).flatten()
        for i, obj_dict in enumerate(objectives_dict_list):
            obj_dict[obj_name] = scaled_values[i]

    # Step 3: Penalize peptides where is_proximate is False by setting their scaled objectives to 0
    for i in range(len(peptides)):
        if not is_proximate_list[i]:
            for obj_name in objective_weights.keys():
                objectives_dict_list[i][obj_name] = 0

    # Step 4: Scalarize the objectives using the provided weights
    scalarized_scores = []
    for obj_dict in objectives_dict_list:
        scalarized_score = sum(
            obj_dict[obj_name] * weight
            for obj_name, weight in objective_weights.items()
        )
        scalarized_scores.append(scalarized_score)

    # Step 5: Prepare the feature matrix X and target vector y
    X_list = []
    y_list = []

    for i, peptide in enumerate(peptides):
        if peptide in peptide_embeddings:
            embedding = peptide_embeddings[peptide]
            peptide_length = len(peptide)
            feature_vector = np.concatenate((embedding, [peptide_length]))
            X_list.append(feature_vector)
            y_list.append(scalarized_scores[i])

    # Convert lists to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)

    # Step 6: Scale the input features X using StandardScaler
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    return X_scaled, y, X_scaler


def train_single_model(X, y, random_state, hopt_params):

    early_stopping_model = MLPRegressor(
        hidden_layer_sizes=hopt_params.get("hidden_layer_sizes", (256, 128, 64, 32, 16, 8)),
        activation=hopt_params.get("activation", "relu"),
        solver="adam",
        alpha=hopt_params.get("alpha", 0.0001),
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=hopt_params.get("learning_rate_init", 0.001),
        max_iter=5000,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=100,
        tol=1e-4,
    )

    early_stopping_model.fit(X, y)
    n_iter_used = early_stopping_model.n_iter_
    val_score = early_stopping_model.best_validation_score_

    full_training_model = MLPRegressor(
        hidden_layer_sizes=hopt_params.get("hidden_layer_sizes", (256, 128, 64, 32, 16, 8)),
        activation=hopt_params.get("activation", "relu"),
        solver="adam",
        alpha=hopt_params.get("alpha", 0.0001),
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=hopt_params.get("learning_rate_init", 0.001),
        max_iter=max(n_iter_used, 100),
        shuffle=True,
        random_state=random_state,
        early_stopping=False,
    )

    full_training_model.fit(X, y)

    return full_training_model, val_score
