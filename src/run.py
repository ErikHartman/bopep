import numpy as np
from sklearn.decomposition import PCA
import pyrosetta

from bayesian_optimization.model_training import prepare_data
from bayesian_optimization.selection import (
    select_next_peptides, 
    select_initial_peptides
)
from bayesian_optimization.ensemble import (
    train_deep_ensemble_parallel
)
from bayesian_optimization.hparam_opt import optimize_hyperparameters
from docking.docking_utils import dock_peptides_batch

from docking.scoring_utils import extract_scores

from data_handling.data_handling_utils import (
    load_embeddings,
)

from data_handling.logging_utils import (
    save_results,
    save_metrics,
    log_iteration_metrics, 
    save_validation_metrics, 
    write_run_log
)

import logging
import os
import time
    

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_bayesian_optimization(
    peptide_embeddings,
    docking_params,
    binding_site_info,
    n_initial,
    n_exploration_iterations,
    n_exploitation_with_distance_weight,
    n_exploitation_iterations,
    batch_size,
    metrics_output_dir,
    objective_weights,
    agreeing_models,
    proximity_threshold,
    hparam_opt_interval 
):
    # Initialize PyRosetta if not already initialized
    pyrosetta.init('-mute all')

    results_filename = "bo_results.csv"
    metrics_filename = "bo_metrics.gz"
    validation_metrics_filename = "bo_validation_metrics.csv"
    results_path = os.path.join(metrics_output_dir, results_filename)
    metrics_path = os.path.join(metrics_output_dir, metrics_filename)
    validation_metrics_path = os.path.join(metrics_output_dir, validation_metrics_filename)

    # Check and remove existing output files
    for file_path in [results_path, metrics_path, validation_metrics_path]:
        if os.path.exists(file_path):
            logging.warning(f"Output file {file_path} already exists and will be overwritten.")
            os.remove(file_path)
            logging.info(f"Removed existing file: {file_path}")

    os.makedirs(metrics_output_dir, exist_ok=True)

    docked_peptides = set()
    all_scores = {}
    metrics_log = []
    validation_metrics_log = []
    total_iterations = 0

    # Step 1: Initial Peptide Selection and Docking
    logging.info(f"Initializing with {n_initial} peptides (k-means)")
    initial_peptides = select_initial_peptides(peptide_embeddings, n_initial)
    docked_peptides.update(initial_peptides)
    dock_peptides_batch(initial_peptides, docking_params)

    # Normalize the objective weights
    weights_sum = sum(objective_weights.values())
    for objective in objective_weights.keys():
        objective_weights[objective] = objective_weights[objective] / weights_sum
    logging.info(f"Objective weights after normalizing: {objective_weights}")

    # Extract initial scores
    scores = extract_scores(initial_peptides, docking_params, binding_site_info, proximity_threshold, agreeing_models)
    all_scores.update(scores)

    # Save initial results
    save_results(scores, metrics_output_dir, iteration=total_iterations, mode='initialization', filename=results_filename)
    total_iterations += 1

    hparams = None
    study = None

    # Bayesian Optimization Phases
    def run_bo_phase(phase_name, iterations, mode):
        nonlocal hparams, study, total_iterations, all_scores

        for iteration in range(iterations):
            logging.info(f"\n{phase_name} iteration {iteration + 1}/{iterations}, total dockings: {len(docked_peptides)}")

            # Step 1: Prepare the data (X_scaled is the input features, y_train is the scalarized score)
            X_scaled, y_train, X_scaler = prepare_data(peptide_embeddings, all_scores, objective_weights)

            if ((total_iterations+1) % hparam_opt_interval == 0) or (hparams == None):
                logging.info("Optimizing hyperparameters...")
                X_scaled, y_train, _ = prepare_data(peptide_embeddings, all_scores, objective_weights)
                new_params, study = optimize_hyperparameters(X_scaled, y_train, previous_study=study)
                hparams = new_params
                logging.info(f"Updated hyperparameters: {hparams}")


            # Step 2: Train the surrogate model using already scaled data
            models, val_r2s = train_deep_ensemble_parallel(
                X_scaled, y_train, num_models=10, params=hparams
            )

            # Log validation metrics
            validation_metrics = {
                'iteration': total_iterations,
                'mode': mode,
                'val_r2s': val_r2s,
                'hyperparameters': hparams
            }
            validation_metrics_log.append(validation_metrics)

            # Step 3: Select peptides using the provided mode
            next_peptides, metrics = select_next_peptides(
                models, X_scaler, y_train, peptide_embeddings, 
                docked_peptides, batch_size, mode=mode
            )
            docked_peptides.update(next_peptides)

            # Dock next peptides
            dock_peptides_batch(next_peptides, docking_params)

            # Step 4: Score next peptides and update results
            scores = extract_scores(next_peptides, docking_params, binding_site_info, proximity_threshold, agreeing_models)
            all_scores.update(scores)

            # Save results with the current iteration and mode
            save_results(scores, metrics_output_dir, iteration=total_iterations, mode=mode, filename=results_filename)

            log_iteration_metrics(metrics_log=metrics_log, metrics=metrics, iteration=total_iterations, mode=mode, selected_peptides=next_peptides)
            total_iterations += 1

    # Run Exploration and Exploitation Phases
    run_bo_phase("Exploration", n_exploration_iterations, "exploration")
    run_bo_phase("Exploitation (weighted)", n_exploitation_with_distance_weight, "exploitation_weighted")
    run_bo_phase("Exploitation", n_exploitation_iterations, "exploitation")

    # Save all metrics to file
    save_validation_metrics(validation_metrics_log, metrics_output_dir, validation_metrics_filename)
    save_metrics(metrics_log, metrics_output_dir, metrics_filename)


def run():
    start_time = time.time()
    # Load peptide embeddings
    
    embedding_directory = "/srv/data1/er8813ha/docking-peptide/output/embedding_chunks_human"
    peptide_embeddings = load_embeddings(embedding_directory)

    # Reduce the dimensionality of the embeddings using PCA
    embedding_array = np.array(list(peptide_embeddings.values()))
    peptide_sequences = list(peptide_embeddings.keys())

    pca = PCA(n_components=0.95, svd_solver="full")
    embeddings_reduced = pca.fit_transform(embedding_array)

    print(f"Reduced bmbedding size: {np.shape(embeddings_reduced)} (before PCA: {np.shape(embedding_array)}), waiting 5 seconds...")

    
    # Create a dictionary for the reduced and scaled embeddings
    peptide_embeddings_reduced = {
        peptide_sequences[i]: embeddings_reduced[i] for i in range(len(peptide_sequences))
    }

    time.sleep(5)

    # Define settings
    run_settings = {
        "embedding_directory": embedding_directory,
        "PCA_n_components": 0.95,
        "target_structure": "./target_structures/4glf.pdb",
        "output_dir": "/srv/data1/er8813ha/docking-peptide/output/bo_output_production",
        "metrics_output_dir": "/srv/data1/er8813ha/docking-peptide/output/summary_outputs_production",
        "num_recycles": 9,
        "msa_mode": "single_sequence",
        "model_type": "alphafold2_multimer_v3",
        "num_relax": 1,
        "num_models": 5,
        "recycle_early_stop_tolerance": 0.3,
        "amber": True,
        "target_chain": "A",
        "num_processes": 4,
        "gpu_ids": ["0", "1", "2", "3"],
        "overwrite_results": False,
        "binding_site_residue_indices": [44, 49, 74, 82, 89, 105],

        "objective_weights": {
            "iptm_score": 1,
            "interface_sasa": 0.2,
            "interface_dG": 0.2,
            "rosetta_score": 0.2,
            "interface_delta_hbond_unsat": 0.2,
            "packstat": 0.2
        },

        "n_initial": 300, # 300
        "n_exploration_iterations": 300, # 1200
        "n_exploitation_with_distance_weight": 700, # 2800
        "n_exploitation_iterations": 0, # 0
        "batch_size": 4,
        "agreeing_models": 0, # top + 1
        "proximity_threshold": 5, # Å
        "hparam_opt_interval": 50,
    }

    # Run the Bayesian optimization using the settings dictionary
    run_bayesian_optimization(
        peptide_embeddings=peptide_embeddings_reduced,
        docking_params={
            "target_structure": run_settings["target_structure"],
            "output_dir": run_settings["output_dir"],
            "num_recycles": run_settings["num_recycles"],
            "msa_mode": run_settings["msa_mode"],
            "model_type": run_settings["model_type"],
            "num_relax": run_settings["num_relax"],
            "num_models": run_settings["num_models"],
            "recycle_early_stop_tolerance": run_settings["recycle_early_stop_tolerance"],
            "amber": run_settings["amber"],
            "target_chain": run_settings["target_chain"],
            "num_processes": run_settings["num_processes"],
            "gpu_ids": run_settings["gpu_ids"],
            "overwrite_results": run_settings["overwrite_results"]
        },
        binding_site_info=run_settings["binding_site_residue_indices"],
        n_initial=run_settings["n_initial"],
        n_exploration_iterations=run_settings["n_exploration_iterations"],
        n_exploitation_with_distance_weight=run_settings["n_exploitation_with_distance_weight"],
        n_exploitation_iterations=run_settings["n_exploitation_iterations"],
        batch_size=run_settings["batch_size"],
        metrics_output_dir=run_settings["metrics_output_dir"],
        objective_weights=run_settings["objective_weights"],
        agreeing_models=run_settings["agreeing_models"],
        proximity_threshold=run_settings["proximity_threshold"],
        hparam_opt_interval=run_settings["hparam_opt_interval"]
    )

    end_time = time.time()
    
    write_run_log(os.path.join(run_settings["metrics_output_dir"], "log.txt"), run_settings, start_time, end_time)

    logging.info(f"Time for the complete pipeline: {int((end_time - start_time) // 3600)} hours, {int(((end_time - start_time) % 3600) // 60)} minutes, {(end_time - start_time) % 60:.2f} seconds")