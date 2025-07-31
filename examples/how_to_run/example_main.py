"""
BoPep Example Script

This script demonstrates a complete workflow for peptide optimization using BoPep.
It includes configuration for all major components and demonstrates advanced features.
"""

import os
from bopep import BoPep, benchmark_objective, Embedder

# ----- Configuration -----

# Define peptide set
# In a real example, you would load your peptides from a file or database.
# Here we use a placeholder empty list for demonstration purposes.
PEPTIDES = []

# Define output directories with defaults that work out of the box
OUTPUT_RESULTS_DIR = os.path.join(os.getcwd(), "optimization_results")
OUTPUT_DOCKING_DIR = os.path.join(os.getcwd(), "docking_results")
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DOCKING_DIR, exist_ok=True)

# Path to target protein structure
# In a real example, you would provide an actual path:
# TARGET_STRUCTURE_PATH = "path/to/your/target_protein.pdb"
TARGET_STRUCTURE_PATH = "example_target.pdb"  # Replace with your actual PDB file

# Optional: Define a subset of peptides to start with
INITIAL_PEPTIDES = []

# Optional: Path to pre-computed ESM embeddings
# ESM_PATH = "path/to/esm_model"
# EMBEDDINGS = "path/to/precomputed_embeddings.pkl"
EMBEDDINGS = None  # Let BoPep compute embeddings from scratch

# Define binding site residues on target protein
# These are the residue indices that form the binding site
BINDING_SITE_RESIDUE_INDICES = [23, 24, 27, 28, 31, 32, 35, 36, 39]

# Configure the docking process
DOCKING_KWARGS = {
    "num_models": 5,            # Number of models to generate per peptide
    "num_recycles": 10,         # Number of structure refinement cycles
    "recycle_early_stop_tolerance": 0.1,  # Early stopping criteria
    "amber": True,              # Use AMBER force field for relaxation
    "num_relax": 1,             # Number of relaxation steps
    "gpu_ids": ["0"],           # GPU IDs to use (adjust based on your hardware)
    "overwrite_results": False, # Whether to overwrite existing results
    "output_dir": OUTPUT_DOCKING_DIR,
}

# Configure surrogate model
MODEL_TYPE = "deep_evidential"  # Options: "nn_ensemble", "mc_dropout", "deep_evidential", "mve"
NETWORK_TYPE = "bigru"          # Options: "mlp", "bilstm", "bigru"

# Configure hyperparameter optimization
HPO_KWARGS = {
    "n_trials": 20,            # Number of HPO trials
    "hpo_interval": 50,        # Run HPO every X iterations
    "n_splits": 3,             # Number of cross-validation splits
}

# Number of samples to use for validation
NUM_VALIDATE = 10              # Set to None to disable validation

# Random seed for reproducibility
SEED = 42

# Number of peptides to evaluate in each batch
BATCH_SIZE = 4

# Define optimization schedule with different acquisition functions
BO_SCHEDULE = [
    {"acquisition": "standard_deviation", "iterations": 5},   # Exploration phase
    {"acquisition": "expected_improvement", "iterations": 15}, # Balanced phase
    {"acquisition": "mean", "iterations": 3},                  # Exploitation phase
]

# Scores to include in the objective function
SCORES_TO_INCLUDE = [
    "iptm",              # Interface predicted TM-score
    "interface_dG",      # Interface delta G (binding energy)
    "peptide_pae",       # Predicted alignment error for peptide
    "rosetta_score",     # Rosetta energy score
    "distance_score",    # Distance to binding site
    "in_binding_site",   # Whether peptide is in binding site
]

# ----- Main Execution -----
if __name__ == "__main__":
    

    embedder = Embedder()
    embeddings = embedder.embed_esm(PEPTIDES, average=True, model_path="...", batch_size=128) # Example ESM model path
    reduced_embeddings = embedder.reduce_embeddings_autoencoder(embeddings, latent_dim=128) # Reduce to 128 dimensions using a VAE

    # Initialize BoPep with desired configuration
    bo = BoPep(
        surrogate_model_kwargs={
            "model_type": MODEL_TYPE, 
            "network_type": NETWORK_TYPE
        },
        objective_function=benchmark_objective,  # Using built-in benchmark objective
        scoring_kwargs={"scores_to_include": SCORES_TO_INCLUDE},
        hpo_kwargs=HPO_KWARGS,
        docker_kwargs=DOCKING_KWARGS,
        log_dir=OUTPUT_RESULTS_DIR,
        overwrite_logs=True,
    )

    print(f"Optimizing {len(PEPTIDES)} peptides with {BATCH_SIZE} peptides per batch")
    print(f"Using {MODEL_TYPE} model with {NETWORK_TYPE} architecture")
    print(f"Validation set size: {NUM_VALIDATE if NUM_VALIDATE else 'None'}")
    
    # Run the optimization process
    bo.optimize(
        embeddings=reduced_embeddings,
        target_structure_path=TARGET_STRUCTURE_PATH,
        schedule=BO_SCHEDULE,
        initial_peptides=INITIAL_PEPTIDES,
        batch_size=BATCH_SIZE,
        binding_site_residue_indices=BINDING_SITE_RESIDUE_INDICES,
        assume_zero_indexed=True,  # PDB files are 0-indexed
        num_validate=NUM_VALIDATE,
    )
    
    print("Optimization complete!")
    print(f"Results and logs saved to: {OUTPUT_RESULTS_DIR}")