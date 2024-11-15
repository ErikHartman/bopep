import time
import logging
import os

from .bayesian_optimization.model_training import prepare_data
from .bayesian_optimization.peptide_selection import (
    select_next_peptides,
    select_initial_peptides,
)
from .bayesian_optimization.ensemble import (
    train_deep_ensemble_parallel,
)
from .bayesian_optimization.hparam_opt import optimize_hyperparameters
from .docking.dock_peptides import dock_peptides
from .docking.scoring import extract_scores

from .logging_utils import save_results, save_validation_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_bayesian_optimization(
    embeddings,
    target_structure,
    target_sequence,
    num_recycles=3,
    num_models=3,
    recycle_early_stop_tolerance=0.3,
    amber=True,
    binding_site_residue_indices=None,
    iptm_score_weight=1.0,
    interface_sasa_weight=0.2,
    interface_dG_weight=0.2,
    rosetta_score_weight=0.2,
    interface_delta_hbond_unsat_weight=0.2,
    packstat_weight=0.2,
    n_initial=100,
    n_exploration_iterations=20,
    n_exploitation_iterations=20,
    batch_size=4,
    agreeing_models=0,
    proximity_threshold=None,
    hparam_opt_interval=10,
    # Hyperparameter optimization parameters
    n_layers_range=(1, 5),
    n_units_l1_range=(64, 1024),
    alpha_range=(1e-5, 1e-3),
    learning_rate_init_range=(1e-4, 1e-2),
    n_splits=5,
    random_state=42,
    n_trials=50,
    n_jobs=1,
    pruner_n_warmup_steps=5,
    direction="maximize",
    pruner_type="MedianPruner",
    sampler_type="TPESampler",
    max_iter=3000,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    hidden_layer_decrease_factor=2,
    min_hidden_layer_size=8,
    bin_edges=None,
):
    """
    Runs the Bayesian optimization pipeline with the specified parameters.
    """
    start_time = time.time()

    output_dir = ("/content/output",)

    # Set up objective weights
    objective_weights = {
        "iptm_score": iptm_score_weight,
        "interface_sasa": interface_sasa_weight,
        "interface_dG": interface_dG_weight,
        "rosetta_score": rosetta_score_weight,
        "interface_delta_hbond_unsat": interface_delta_hbond_unsat_weight,
        "packstat": packstat_weight,
    }

    # Normalize the objective weights
    weights_sum = sum(objective_weights.values())
    for objective in objective_weights.keys():
        objective_weights[objective] = objective_weights[objective] / weights_sum
    logging.info(f"Objective weights after normalizing: {objective_weights}")

    docked_peptides = set()
    all_scores = {}
    validation_metrics_log = []
    total_iterations = 0
    peptide_results = {}

    # Step 1: Initial Peptide Selection and Docking
    logging.info(f"Initializing with {n_initial} peptides (k-means)")
    initial_peptides = select_initial_peptides(embeddings, n_initial)
    docked_peptides.update(initial_peptides)

    # Dock peptides using a function suitable for Colab
    peptide_results = dock_peptides(
        initial_peptides,
        peptide_results,
        target_structure=target_structure,
        target_sequence=target_sequence,
        num_models=num_models,
        num_recycles=num_recycles,
        recycle_early_stop_tolerance=recycle_early_stop_tolerance,
        amber=amber,
        num_relax=1,
    )

    print(peptide_results)

    # Extract initial scores
    scores = extract_scores(
        peptides=initial_peptides,
        target_structure=target_structure,
        binding_site_info=binding_site_residue_indices,
        proximity_threshold=proximity_threshold,
        agreeing_models=agreeing_models,
    )
    all_scores.update(scores)

    # Save initial results
    save_results(
        scores,
        output_dir,
        iteration=total_iterations,
        mode="initialization",
        filename=os.path.join(output_dir, "results.csv"),
    )
    total_iterations += 1

    hparams = None
    study = None

    # Bayesian Optimization Phases
    def run_bo_phase(phase_name, iterations, mode):
        nonlocal hparams, study, total_iterations, all_scores

        for iteration in range(iterations):
            logging.info(
                f"\n{phase_name} iteration {iteration + 1}/{iterations}, total dockings: {len(docked_peptides)}"
            )

            X_scaled, y_train, X_scaler = prepare_data(
                embeddings, all_scores, objective_weights
            )

            # Hyperparameter optimization
            if ((total_iterations + 1) % hparam_opt_interval == 0) or (hparams is None):
                logging.info("Optimizing hyperparameters...")
                X_scaled, y_train, _ = prepare_data(
                    embeddings, all_scores, objective_weights
                )
                new_params, study = optimize_hyperparameters(
                    X_scaled,
                    y_train,
                    previous_study=study,
                    n_layers_range=n_layers_range,
                    n_units_l1_range=n_units_l1_range,
                    alpha_range=alpha_range,
                    learning_rate_init_range=learning_rate_init_range,
                    n_splits=n_splits,
                    random_state=random_state,
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    pruner_n_warmup_steps=pruner_n_warmup_steps,
                    direction=direction,
                    pruner_type=pruner_type,
                    sampler_type=sampler_type,
                    max_iter=max_iter,
                    tol=tol,
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    n_iter_no_change=n_iter_no_change,
                    hidden_layer_decrease_factor=hidden_layer_decrease_factor,
                    min_hidden_layer_size=min_hidden_layer_size,
                    bin_edges=bin_edges,
                )
                hparams = new_params
                logging.info(f"Updated hyperparameters: {hparams}")

            # Step 2: Train the surrogate model
            models, val_r2s = train_deep_ensemble_parallel(
                X_scaled, y_train, num_models=10, params=hparams
            )

            # Log validation metrics
            validation_metrics = {
                "iteration": total_iterations,
                "mode": mode,
                "val_r2s": val_r2s,
                "hyperparameters": hparams,
            }
            validation_metrics_log.append(validation_metrics)

            # Step 3: Select peptides
            next_peptides = select_next_peptides(
                models,
                X_scaler,
                y_train,
                embeddings,
                docked_peptides,
                batch_size,
                mode=mode,
            )
            docked_peptides.update(next_peptides)

            # Dock next peptides
            peptide_results = dock_peptides(
                initial_peptides,
                peptide_results,
                target_structure=target_structure,
                target_sequence=target_sequence,
                num_models=num_models,
                num_recycles=num_recycles,
                recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                amber=amber,
                num_relax=1,
            )

            # Step 4: Score next peptides and update results
            scores = extract_scores(
                next_peptides,
                target_structure=target_structure,
                binding_site_info=binding_site_residue_indices,
                proximity_threshold=proximity_threshold,
                agreeing_models=agreeing_models,
            )
            all_scores.update(scores)

            # Save results
            save_results(
                scores,
                output_dir,
                iteration=total_iterations,
                mode=mode,
                filename=os.path.join(output_dir, "results.csv"),
            )

            total_iterations += 1

    # Run Exploration and Exploitation Phases
    run_bo_phase("Exploration", n_exploration_iterations, "exploration")
    run_bo_phase("Exploitation", n_exploitation_iterations, "exploitation")

    # Save validation metrics
    save_validation_metrics(
        validation_metrics_log,
        output_dir,
        os.path.join(output_dir, "training_metrics.csv"),
    )

    end_time = time.time()
    logging.info(
        f"Time for the complete pipeline: {int((end_time - start_time) // 3600)} hours, {int(((end_time - start_time) % 3600) // 60)} minutes, {(end_time - start_time) % 60:.2f} seconds"
    )
