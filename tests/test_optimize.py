import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging

# Add the parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bopep.bayesian_optimization.optimization import BoPep

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def iptm_objective(scores: dict) -> dict:
    """
    Returns the iptm score for each peptide in the scores dictionary.
    """
    scalar_objectives = {}
    for peptide, peptide_scores in scores.items():
        if peptide_scores["in_binding_site"]:
            scalar_objectives[peptide] = peptide_scores["iptm"]
        else:
            scalar_objectives[peptide] = 0
    return scalar_objectives


def load_benchmark_embeddings(embedding_path, embedding_type="esm1d"):
    """Load embeddings from the benchmark data."""
    logging.info(f"Loading benchmark embeddings from {embedding_path}...")
    try:
        with open(embedding_path, 'rb') as f:
            benchmark_data = pickle.load(f)
        logging.info(f"Loaded benchmark data with keys: {benchmark_data.keys()}")
        
        # Check if we have reduced embeddings for the embedding type
        if "reduced_embeddings" in benchmark_data and embedding_type in benchmark_data["reduced_embeddings"]:
            # Get the embeddings for this type
            embeddings_dict = benchmark_data["reduced_embeddings"][embedding_type]
            
            # Convert embeddings_dict to correct format if needed
            if not isinstance(embeddings_dict, dict):
                # If it's a numpy array, convert to dict using peptides list
                if isinstance(embeddings_dict, np.ndarray):
                    all_peptides = benchmark_data["peptides"]
                    embeddings_dict = {p: embeddings_dict[i] for i, p in enumerate(all_peptides)}
            
            # Log a sample embedding dimension
            sample_embedding = next(iter(embeddings_dict.values()))
            logging.info(f"Embedding dimension for {embedding_type}: {sample_embedding.shape}")
            
            return embeddings_dict
        else:
            logging.error(f"Could not find reduced embeddings for {embedding_type}")
            return None
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None


def test_bopep_with_precomputed_data(
    embeddings_path,
    objectives_csv,
    output_dir="./test_output",
    embedding_type="esm_1d_pca", 
    n_trials=5,
    iterations=10,
    batch_size=4
):
    """

    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-computed objectives
    df = pd.read_csv(objectives_csv)
    peptides = df['peptide'].tolist()
    
    all_scores = {}
    for _, row in df.iterrows():
        peptide = row['peptide']
        # Create a dictionary of all scores for this peptide
        score_dict = {col: row[col] for col in df.columns if col != 'peptide'}
        all_scores[peptide] = score_dict
    
    # Load embeddings using the provided function
    embeddings = load_benchmark_embeddings(embeddings_path, embedding_type)
    
    if not embeddings:
        logging.error("Failed to load embeddings. Exiting...")
        return None
    
    # Ensure we have embeddings for all peptides
    filtered_peptides = [p for p in peptides if p in embeddings]
    filtered_embeddings = {p: embeddings[p] for p in filtered_peptides}
    
    missing = len(peptides) - len(filtered_peptides)
    if missing > 0:
        logging.warning(f"Missing embeddings for {missing} peptides")
        logging.warning(f"Proceeding with {len(filtered_peptides)} peptides")
    
    # Determine the appropriate network type based on embedding shape
    any_embedding = next(iter(filtered_embeddings.values()))
    network_type = "mlp"
    logging.info(f"Using {network_type} network based on embedding shape {any_embedding.shape}")
    
    # Initialize BoPep with necessary components only
    bopep = BoPep(
        surrogate_model_kwargs={
            "network_type": network_type,
            "model_type": "deep_evidential",
        },
        docker_kwargs={
        "output_dir": os.path.join(output_dir, "docking"),  # Create a subdirectory for docking
        "num_cores": 1  # Use minimal resources for testing
        },
        hpo_kwargs={
            "n_trials": n_trials,
            "hpo_interval": 20 
        },
        objective_function=iptm_objective,
        scoring_kwargs={"scores_to_include": ["iptm", "in_binding_site"]},
        log_dir=output_dir
    )
    
    # Mock the scoring method to return our pre-computed scores
    def mock_score_batch(docked_dirs):
        # Extract peptide names from the mock "docked_dirs" (which are just peptide names)
        selected_peptides = docked_dirs
        
        # Get the scores we already have for these peptides using our precomputed dictionary
        mock_scores = {}
        for peptide in selected_peptides:
            if peptide in all_scores:
                mock_scores[peptide] = all_scores[peptide]
        
        return mock_scores
    
    # Replace the actual _score_batch method with our mock
    bopep._score_batch = mock_score_batch
    
    # Skip the actual docking step
    bopep.docker.dock_peptides = lambda peptides: peptides  # Just return the peptide names
    
    # Also mock binding site check to avoid interactive prompts
    bopep._check_binding_site_residue_indices = lambda binding_site, target: [1, 2, 3, 4, 5]
    
    # Define a minimal optimization schedule
    schedule = [
        {"acquisition": "expected_improvement", "iterations": iterations},
    ]
    
    # Run the optimization loop
    logging.info("Starting test optimization loop")
    try:
        bopep.optimize(
            peptides=filtered_peptides,
            target_structure_path="/home/er8813ha/bopep/data/4glp.pdb",  # Not used with our mocks
            num_initial=10,
            batch_size=batch_size,
            schedule=schedule,
            embeddings=filtered_embeddings,
        )
        logging.info(f"Test completed, output saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error during optimization: {e}")
    
    # Return the trained model for inspection
    return bopep


if __name__ == "__main__":

    embeddings_path = "/srv/data1/er8813ha/docking-peptide/output_v2/benchmarking/embedding_methods/embedding_benchmark_data.pkl"
    objectives_csv ="/home/er8813ha/docking-peptide/src/benchmark/benchmark_scores.csv"
    
    # Run the test with ESM 1D embeddings
    bopep = test_bopep_with_precomputed_data(
        embeddings_path=embeddings_path,
        objectives_csv=objectives_csv,
        embedding_type="esm_1d_pca",
        output_dir="/home/er8813ha/docking-peptide/src/plot/dummy_logs",
        iterations=20, 
        n_trials=3,
        batch_size=10
    )
    
    if bopep and bopep.model:
        print(f"Trained model: {type(bopep.model).__name__}")