"""
Main run script for the docking to CD14.
"""

import logging
import numpy as np
import pandas as pd
from bopep import BoPep, benchmark_objective
import torch
import pickle
from bopep_cd14_pept_kwargs import (
    OUTPUT_RESULTS_DIR,
    TARGET_STRUCTURE_PATH,
    BO_SCHEDULE,
    SEED,
    BATCH_SIZE,
    SCORES_TO_INCLUDE,
    BINDING_SITE_RESIDUE_INDICES,
    DOCKING_KWARGS,
    INITIAL_PEPTIDES_DIR,
    model_type,
    network_type,
    EMBEDDING_DATA_PATH,
    HPO_KWARGS,
    N_VALIDATE
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    logging.info("Using CUDA")
    torch.cuda.manual_seed(SEED)

def load_benchmark_data():
    """Load the precomputed benchmark data."""
    logging.info(f"Loading benchmark data from {EMBEDDING_DATA_PATH}...")
    with open(EMBEDDING_DATA_PATH, 'rb') as f:
        benchmark_data = pickle.load(f)
    logging.info(f"Loaded benchmark data with keys: {benchmark_data.keys()}")
    return benchmark_data["reduced_embeddings"]["esm_2d_vae"] # dict of peptide -> embedding


if __name__ == "__main__":
    initial_peptides = pd.read_csv(INITIAL_PEPTIDES_DIR)["peptide"].tolist()

    bo = BoPep(
        surrogate_model_kwargs={"model_type": model_type, "network_type": network_type},
        objective_function=benchmark_objective,
        scoring_kwargs = {
            "scores_to_include": SCORES_TO_INCLUDE,
            "binding_site_distance_threshold": 5.0,
            "required_n_contact_residues": 8,
        },
        hpo_kwargs=HPO_KWARGS,
        docker_kwargs=DOCKING_KWARGS,
        log_dir=OUTPUT_RESULTS_DIR,
        overwrite_logs=True
    )
    embeddings = load_benchmark_data()

    print("Residues in binding site:", BINDING_SITE_RESIDUE_INDICES)
    print("These should correspond to: E23, Q42, D44, F49, F69, ... L105 ... etc")
    
    bo.optimize(
        target_structure_path=TARGET_STRUCTURE_PATH,
        schedule=BO_SCHEDULE,
        initial_peptides=initial_peptides,
        batch_size=BATCH_SIZE,
        embeddings = embeddings,
        binding_site_residue_indices=BINDING_SITE_RESIDUE_INDICES,
        assume_zero_indexed=True,
        n_validate=N_VALIDATE,
    )
