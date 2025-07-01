import csv
import json
from pathlib import Path
import pickle
import datetime
import shutil
from typing import Callable, List, Optional, Dict, Any, Union
from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import (
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    tune_hyperparams,
    MVE,
)
from bopep.logging.logger import Logger
from bopep.bayesian_optimization.acquisition_functions import AcquisitionFunction
from bopep.bayesian_optimization.utils import (_check_binding_site_residue_indices, _save_model, _validate_checkpoint, _validate_dependencies, _validate_args, _validate_surrogate_model_kwargs)
from bopep.bayesian_optimization.selection import PeptideSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
import torch
import logging
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BoPep:

    def __init__(
        self,
        surrogate_model_kwargs: Dict[str, Any] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        hpo_kwargs: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        overwrite_logs: Optional[bool] = None,
        custom_scorer: Optional[Callable] = None,
        min_validation_samples: int = 3,
        min_training_samples: int = 10,
    ):
        """
        Initialize the BoPep optimizer with various configuration options.

        Args:
            surrogate_model_kwargs: Configuration for the surrogate model including:
                - network_type: 'mlp', 'bilstm', or 'bigru'
                - model_type: 'nn_ensemble', 'mc_dropout', 'deep_evidential', or 'mve'
            objective_function: Custom objective function (defaults to maximize iptm/pae and minimize dG and rosetta_score, see paper)
            objective_function_kwargs: Parameters for the objective function
            docker_kwargs: Configuration for the Docker component
            hpo_kwargs: Configuration for hyperparameter optimization
            log_dir: Directory for logging output
            custom_scorer: Optional function that takes docking directories and
                    returns score dictionaries. If provided, this will be used
                    instead of the default scorer.
        """
        _validate_dependencies()

        self.surrogate_model_kwargs = surrogate_model_kwargs
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.docker_kwargs = docker_kwargs or {}
        self.hpo_kwargs = hpo_kwargs or {}
        self.scoring_kwargs = scoring_kwargs or {}
        self.custom_scorer = custom_scorer

        # Initialize components
        self.scorer = Scorer()
        self.docker = Docker(self.docker_kwargs)
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = PeptideSelector()
        self.scores_to_objective = ScoresToObjective()

        # Store the Optuna study for warm-starting
        self.previous_study = None

        _validate_surrogate_model_kwargs(self.surrogate_model_kwargs)
        self.log_dir = log_dir
        self.overwrite_logs = overwrite_logs

        self.MIN_VALIDATION_SAMPLES = min_validation_samples
        self.MIN_TRAINING_SAMPLES = min_training_samples

    def optimize(
        self,
        schedule: List[Dict[str, Any]],
        batch_size: int,
        target_structure_path: str,
        embeddings: Optional[Dict[str, Any]] = None,
        num_initial: Optional[int] = 10,
        n_validate: Optional[Union[float, int]] = None,
        binding_site_residue_indices: Optional[List[int]] = None,
        initial_peptides: Optional[List[str]] = None,
        assume_zero_indexed: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Runs Bayesian optimization on peptide sequences.

        Args:
            schedule: List of dictionaries defining the optimization phases.
                Each dict should have 'acquisition' and 'iterations' keys.
            batch_size: Number of peptides to dock in each iteration.
            target_structure_path: Path to the target structure PDB file.
            embeddings: Dictionary of peptide embeddings {peptide: embedding}.
            num_initial: Number of initial peptides to dock and score.
            n_validate: Number of peptides to use for validation.
                If None, no validation is performed.
            binding_site_residue_indices: List of residue indices defining the binding site.
            initial_peptides: Optional list of initial peptides to dock.
            assume_zero_indexed: If True, assumes residue indices are zero-indexed.
            checkpoint_path: Path to a checkpoint directory to continue from. If none is provided,
                a fresh optimization will be started.

        """
        continue_from_checkpoint = False if checkpoint_path is None else True

        self.checkpoint_path = checkpoint_path
        self.logger = Logger(log_dir=self.log_dir, overwrite_logs=self.overwrite_logs, continue_from_checkpoint=continue_from_checkpoint)

        self.embeddings = embeddings
        self.target_structure_path = target_structure_path
        self.schedule = schedule
        self.batch_size = batch_size
        self.n_validate = n_validate
        self.binding_site_residue_indices = binding_site_residue_indices

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not continue_from_checkpoint:
            if not embeddings:
                raise ValueError("embeddings must be provided for fresh initialization")
            logging.info("Fresh initialization")
            self._fresh_init(initial_peptides, num_initial, assume_zero_indexed)
            starting_iteration = 0
        else:
            logging.info("Continuing from checkpoint")
            starting_iteration = self._continue_init()
        
        logging.info("Starting optimization loop")
        self._run_phase_loop(
            schedule=schedule,
            all_logged_objectives=self.all_logged_objectives,
            docked_peptides=self.docked_peptides,
            not_docked_peptides=self.not_docked_peptides,
            scores=self.scores,
            batch_size=batch_size,
            n_validate=n_validate,
            starting_global_iteration=starting_iteration,
        )

    def _fresh_init(
        self,
        initial_peptides: Optional[List[str]] = None,
        num_initial: int = 10,
        assume_zero_indexed: Optional[bool] = None,
    ):
        """
        Initializes the BoPep optimizer for a fresh search
        """
        peptides = list(self.embeddings.keys())
        self.docked_peptides = set()
        self.not_docked_peptides = set(peptides)
        self.scores = dict()

        # Validate args
        _validate_args(schedule=self.schedule, n_validate=self.n_validate)
        self.docker.set_target_structure(self.target_structure_path)
        self.binding_site_residue_indices = _check_binding_site_residue_indices(
            self.binding_site_residue_indices,
            self.target_structure_path,
            assume_zero_indexed=assume_zero_indexed,
        )

        self._check_embedding_shapes(self.embeddings)

        if self.surrogate_model_kwargs["network_type"] == "mlp":
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = len(self.embeddings[any_key])
        else:
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = self.embeddings[any_key].shape[
                -1
            ]

        logging.info(
            f"Embeddings dimension: {self.surrogate_model_kwargs['input_dim']}"
        )


        self.best_hyperparams = None
        if initial_peptides is None:
            initial_peptides = self.selector.select_initial_peptides(
                embeddings=self.embeddings, num_initial=num_initial, random_state=42
            )

        logging.info("Docking initial peptides")
        docked_dirs = self.docker.dock_peptides(initial_peptides)
        self.docked_peptides.update(initial_peptides)
        self.not_docked_peptides.difference_update(initial_peptides)

        logging.info(f"Scoring {len(initial_peptides)} initial peptides")
        initial_scores = self._score_batch(docked_dirs=docked_dirs)
        self.scores.update(initial_scores)
        self.logger.log_scores(initial_scores, iteration=0, acquisition_name="initial")

        objectives = self.scores_to_objective.create_objective(
            self.scores, self.objective_function, **self.objective_function_kwargs
        )
        self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")
        self.all_logged_objectives = set(objectives.keys())

        self._optimize_hyperparameters(
            0, {p: self.embeddings[p] for p in self.docked_peptides}, objectives
        )
        self._initialize_model(self.best_hyperparams)

    def _next_checkpoint_dir(self) -> Path:
        """Find the next available checkpoint_{i} directory under self.log_dir."""
        base = Path(self.log_dir)
        existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
        # Extract suffix numbers
        idxs = []
        for d in existing:
            try:
                idxs.append(int(d.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        next_idx = max(idxs) + 1 if idxs else 0
        return base / f"checkpoint_{next_idx}"

    def _save_checkpoint(self, global_iteration: int):
        checkpoint_dir = self._next_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        _save_model(str(checkpoint_dir / "model.pt"), model=self.model, surrogate_model_kwargs=self.surrogate_model_kwargs, best_hyperparams=self.best_hyperparams)
        meta_json_path = checkpoint_dir / "checkpoint_metadata.json"
        
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "global_iteration": global_iteration,
            "surrogate_model_kwargs": self.surrogate_model_kwargs,
            "target_structure_path": self.target_structure_path,
            "binding_site_residue_indices": self.binding_site_residue_indices,
            "docker_kwargs": self.docker_kwargs,
            "hpo_kwargs": self.hpo_kwargs,
            "objective_function_kwargs": self.objective_function_kwargs,
            "scoring_kwargs": self.scoring_kwargs,
            "num_docked_peptides": len(self.docked_peptides),
            "num_remaining_peptides": len(self.not_docked_peptides),
        }

        if self.checkpoint_path:
            meta["checkpoint_path"] = self.checkpoint_path
        
        with open(meta_json_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        with open(checkpoint_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

        self._copy_logs_to_checkpoint(checkpoint_dir)

        logging.info(f"Saved checkpoint at iteration {global_iteration}")
        logging.info(f"Checkpoint metadata saved to {meta_json_path}")

    def _copy_logs_to_checkpoint(self, checkpoint_dir: Path):
        """Copy all log files to the checkpoint directory."""
        
        results_dir = checkpoint_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        log_files = [
            "scores.csv",
            "objectives.csv", 
            "model_losses.csv",
            "predictions.csv.gz",
            "acquisition.csv.gz",
            "hyperparameters.csv"
        ]
        
        logs_copied = 0
        for log_file in log_files:
            source_path = Path(self.log_dir) / log_file
            if source_path.exists():
                dest_path = results_dir / log_file
                shutil.copy2(source_path, dest_path)
                logs_copied += 1
                logging.debug(f"Copied {log_file} to checkpoint")
        
        logging.info(f"Copied {logs_copied} log files to checkpoint")

    def _continue_init(self) -> int:
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint directory found at {self.checkpoint_path}")
        _validate_checkpoint(checkpoint_path=checkpoint_path)
        
        embeddings_path = checkpoint_path / "embeddings.pkl"
        meta_path = checkpoint_path / "checkpoint_metadata.json"
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        logging.info(f"Loading metadata from {meta_path}")

        logging.info("Loading embeddings from checkpoint...")
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        
        self._check_embedding_shapes(self.embeddings)

        if self.surrogate_model_kwargs != meta.get("surrogate_model_kwargs"):
            logging.warning(
                "Note that current surrogate model kwargs differ from checkpoint. " \
                "Using provided kwargs"
            )
            logging.warning("Previous kwargs: " + str(self.surrogate_model_kwargs))
            logging.warning("Checkpoint kwargs: " + str(meta.get("surrogate_model_kwargs")))
        
        if self.surrogate_model_kwargs["network_type"] == "mlp":
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = len(self.embeddings[any_key])
        else:
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = self.embeddings[any_key].shape[-1]
        
        logging.info(f"Embeddings dimension: {self.surrogate_model_kwargs['input_dim']}")

        logging.info("Rebuilding optimization state from logs...")
        self._rebuild_logs_from_csvs()

        _validate_args(schedule=self.schedule, n_validate=self.n_validate)
        self.docker.set_target_structure(self.target_structure_path)
        
        objectives = self.scores_to_objective.create_objective(
            self.scores, self.objective_function, **self.objective_function_kwargs
        )
        docked_embs = {p: self.embeddings[p] for p in self.docked_peptides}
        
        last_iteration = meta["global_iteration"]
        
        logging.info("Running hyperparameter optimization...")
        self._optimize_hyperparameters(last_iteration, docked_embs, objectives)
        self._initialize_model(self.best_hyperparams)

        logging.info(f"Resumed from iteration {last_iteration} with "
                    f"{len(self.docked_peptides)} peptides docked")
        
        return last_iteration

    
    def _rebuild_logs_from_csvs(self):
        scores_path = Path(self.log_dir) / "scores.csv"
        self.scores = {}
        with open(scores_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pep = row["peptide"]
                score_cols = [c for c in reader.fieldnames
                              if c not in ("timestamp", "iteration", "peptide", "phase")]
                sc = {
                    col: float(row[col]) if row[col] not in (None, "") else None
                    for col in score_cols
                }
                self.scores[pep] = sc

        obj_path = Path(self.log_dir) / "objectives.csv"
        self.all_logged_objectives = set()
        with open(obj_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.all_logged_objectives.add(row["peptide"])

        self.docked_peptides = set(self.scores.keys())
        self.not_docked_peptides = set(self.embeddings.keys()) - self.docked_peptides


    def _run_phase_loop(
        self,
        schedule: List[Dict[str, Any]],
        all_logged_objectives: set,
        docked_peptides: set,
        not_docked_peptides: set,
        scores: dict,
        batch_size: int,
        n_validate: Optional[Union[int, float]] = None,
        starting_global_iteration: int = 0,
    ):
        """
        Runs the main optimization loop over the defined phases.
        """
        global_iteration = starting_global_iteration
        for phase_index, phase in enumerate(schedule, start=1):
            acquisition = phase["acquisition"]
            iterations = phase["iterations"]

            for iteration in range(1, iterations + 1):
                global_iteration += 1
                logging.info("=" * 60)
                logging.info(
                    f"Starting iteration {iteration} (global: {global_iteration}) out of {iterations} of phase {phase_index} with acquisition '{acquisition}'"
                )

                # Initialize fresh model for each iteration
                self._initialize_model(self.best_hyperparams)

                # Turn scores into a scalarized score dict of peptide: score
                objectives = self.scores_to_objective.create_objective(
                    scores, self.objective_function, **self.objective_function_kwargs
                )

                # Log the new objective values
                new_objective_peptides = set(objectives.keys()) - all_logged_objectives
                new_objectives = {
                    peptide: objectives[peptide] for peptide in new_objective_peptides
                }

                if new_objectives:
                    self.logger.log_objectives(
                        new_objectives,
                        iteration=global_iteration,
                        acquisition_name=acquisition,
                    )
                    all_logged_objectives.update(new_objectives.keys())

                # Train the model on *only the peptides we have scores for*
                docked_embeddings = {p: self.embeddings[p] for p in docked_peptides}

                # Run hyperparameter optimization every N steps
                if global_iteration % self.hpo_kwargs.get("hpo_interval", 10) == 0:
                    self._optimize_hyperparameters(
                        global_iteration, docked_embeddings, objectives
                    )
                    self._initialize_model(self.best_hyperparams)

                logging.info(
                    f"Model will be trained (potentially validated) on {len(docked_peptides)} peptides"
                )

                # Check if we can split into train/validation
                split = None
                if n_validate is not None:
                    split = self._compute_split_indices(
                        len(docked_peptides), n_validate
                    )

                # If we can split, do it; otherwise, train on all
                if split:
                    train_emb, train_obj, val_emb, val_obj = (
                        self._split_train_validation(
                            docked_embeddings, objectives, split
                        )
                    )
                    loss, metrics = self._train_and_validate(
                        train_emb, train_obj, val_emb, val_obj
                    )
                else:
                    loss, metrics = self._train_on_all(docked_embeddings, objectives)

                # Log the loss and metrics
                self.logger.log_model_metrics(loss, global_iteration, metrics)

                # Predict for *the not-yet-docked* peptides
                candidate_embeddings = {
                    p: self.embeddings[p] for p in not_docked_peptides
                }
                predictions = self.model.predict_dict(
                    candidate_embeddings, device=self.device
                )  # predictions is {peptide: (mean, std)}

                # Log predictions
                self.logger.log_predictions(predictions, global_iteration)

                # Compute acquisition
                acquisition_values = self.acquisition_function_obj.compute_acquisition(
                    predictions=predictions,
                    acquisition_function=acquisition,
                )  # acquisition_values is {peptide: acquisition_value}

                # Log acquisition
                self.logger.log_acquisition(
                    acquisition_values=acquisition_values,
                    acquisition_name=acquisition,
                    iteration=global_iteration,
                )

                # Select the next set of peptides to dock
                next_peptides = self.selector.select_next_peptides(
                    peptides=not_docked_peptides,
                    embeddings=candidate_embeddings,
                    acquisition_values=acquisition_values,
                    n_select=batch_size,
                )

                # Dock them
                docked_dirs = self.docker.dock_peptides(next_peptides)
                docked_peptides.update(next_peptides)
                not_docked_peptides.difference_update(next_peptides)

                # Score them
                new_scores = self._score_batch(docked_dirs=docked_dirs)
                scores.update(new_scores)

                # Log new scores
                self.logger.log_scores(
                    new_scores, iteration=global_iteration, acquisition_name=acquisition
                )

                # Print top performers
                self._print_top_performers(
                    objectives=objectives,
                )

        # Loop is over
        final_objectives = self.scores_to_objective.create_objective(
            scores, self.objective_function, **self.objective_function_kwargs
        )
        final_new_objective_peptides = (
            set(final_objectives.keys()) - all_logged_objectives
        )
        final_new_objectives = {
            peptide: final_objectives[peptide]
            for peptide in final_new_objective_peptides
        }
        if final_new_objectives:
            self.logger.log_objectives(
                final_new_objectives,
                iteration=global_iteration + 1,
                acquisition_name=acquisition,
            )

        # Here we'll want to save the checkpoint and model
        self._save_checkpoint(global_iteration)

    def _score_batch(self, docked_dirs: list):
        """
        Scores a batch (list of dirs) and outputs the score dict
        """
        if self.custom_scorer:
            return self.custom_scorer(docked_dirs)

        new_scores = {}
        scores_to_include = self.scoring_kwargs.get("scores_to_include", [])

        if not scores_to_include:
            raise ValueError(
                "No scores to include for scoring. Please specify valid scores in 'scores_to_include'."
            )

        new_scores = self.scorer.score_batch(
            scores_to_include=scores_to_include,
            inputs=docked_dirs,
            input_type="colab_dir",
            binding_site_residue_indices=self.binding_site_residue_indices,
            n_jobs=self.scoring_kwargs.get("n_jobs", 12),
        )

        return new_scores


    def _check_embedding_shapes(self, embeddings: Dict[str, Any]):
        """
        Check that the embeddings' dimensionality matches the expected shape
        for the chosen network type:
          - mlp => expect 1D array (e.g. shape == (embedding_dim,))
          - bilstm/bigru => expect 2D array (e.g. (sequence_length, embedding_dim))
        """
        network_type = self.surrogate_model_kwargs["network_type"]
        for pep, emb in embeddings.items():
            if network_type in ["bilstm", "bigru"]:
                if emb.ndim != 2:
                    raise ValueError(
                        f"For {network_type}, each peptide embedding must be 2D. "
                        f"Peptide '{pep}' has shape {emb.shape}."
                    )
            elif network_type == "mlp":
                if emb.ndim != 1:
                    raise ValueError(
                        f"For mlp, each peptide embedding must be 1D. "
                        f"Peptide '{pep}' has shape {emb.shape}."
                    )

    def _optimize_hyperparameters(self, iteration: int, embeddings: dict, scores: dict):
        """
        Use the Hyperparameter Tuner to optimize all params.
        Sets self.best_hyperparams to the best found hyperparameters.
        Stores the Optuna study for future warm-starting.
        """
        hpo_config = self.hpo_kwargs.copy() if self.hpo_kwargs else {}

        n_trials = hpo_config.get("n_trials", 20)
        n_splits = hpo_config.get("n_splits", 3)
        random_state = hpo_config.get("random_state", 42)

        logging.info(
            f"Starting hyperparameter optimization for {self.surrogate_model_kwargs['network_type']} {self.surrogate_model_kwargs['model_type']} model..."
        )

        if self.previous_study is not None:
            logging.info(
                f"Using previous study with {len(self.previous_study.trials)} trials"
            )

        self.best_hyperparams, self.previous_study = tune_hyperparams(
            model_type=self.surrogate_model_kwargs["model_type"],
            embedding_dict=embeddings,
            scores_dict=scores,
            network_type=self.surrogate_model_kwargs["network_type"],
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            previous_study=self.previous_study,
        )

        self.logger.log_hyperparameters(
            iteration,
            self.best_hyperparams,
            model_type=self.surrogate_model_kwargs["model_type"],
            network_type=self.surrogate_model_kwargs["network_type"],
        )

        logging.info(
            f"Hyperparameter optimization complete. Best parameters: {self.best_hyperparams}"
        )

    def _initialize_model(self, hyperparams: dict):
        """
        Create model based on hyperparameters and self.surrogate_model_kwargs.
        Sets self.model to the created model.
        """
        model_type = self.surrogate_model_kwargs["model_type"]
        network_type = self.surrogate_model_kwargs["network_type"]
        input_dim = self.surrogate_model_kwargs["input_dim"]

        # Extract hyperparameters
        hidden_dims = hyperparams.get("hidden_dims")
        hidden_dim = hyperparams.get("hidden_dim")
        num_layers = hyperparams.get("num_layers", 2)
        uncertainty_param = hyperparams.get("uncertainty_param")

        if model_type == "mve":
            self.model = MVE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                mve_regularization=uncertainty_param,
            )
        elif model_type == "deep_evidential":
            self.model = DeepEvidentialRegression(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                evidential_regularization=uncertainty_param,
            )
        elif model_type == "mc_dropout":
            self.model = MonteCarloDropout(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                dropout_rate=uncertainty_param,
            )
        elif model_type == "nn_ensemble":
            self.model = NeuralNetworkEnsemble(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                n_networks=int(uncertainty_param),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logging.info(
            f"Initialized {model_type} model with {network_type} network architecture"
        )
        self.model.to(self.device)

    def _compute_model_metrics(self, predictions_dict: dict, objectives: dict):
        peptides = list(predictions_dict.keys())
        actual = np.array([objectives[p] for p in peptides])
        predicted = np.array([predictions_dict[p][0] for p in peptides])
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        return {"r2": r2, "mae": mae}

    def _compute_split_indices(
        self, total_samples: int, n_validate: Union[int, float]
    ) -> Optional[int]:
        """Return num_validate or None if split is infeasible."""
        if isinstance(n_validate, float):
            num = int(total_samples * n_validate)
        else:
            num = n_validate

        if (
            num < self.MIN_VALIDATION_SAMPLES
            or (total_samples - num) < self.MIN_TRAINING_SAMPLES
        ):
            logging.warning(
                f"Cannot split {total_samples} samples into "
                f"{self.MIN_TRAINING_SAMPLES} train + "
                f"{self.MIN_VALIDATION_SAMPLES} val; training on all."
            )
            return None

        return num

    def _train_and_validate(self, train_emb, train_obj, val_emb, val_obj):
        """Train on train set, evaluate on both splits."""
        loss = self.model.fit_dict(
            embedding_dict=train_emb,
            scores_dict=train_obj,
            epochs=self.best_hyperparams.get("epochs", 100),
            learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
            batch_size=self.best_hyperparams.get("batch_size", 16),
            device=self.device,
        )
        train_pred = self.model.predict_dict(train_emb, device=self.device)
        val_pred = self.model.predict_dict(val_emb, device=self.device)
        train_m = self._compute_model_metrics(train_pred, train_obj)
        val_m = self._compute_model_metrics(val_pred, val_obj)

        metrics = {
            "train_r2": train_m["r2"],
            "train_mae": train_m["mae"],
            "val_r2": val_m["r2"],
            "val_mae": val_m["mae"],
        }
        logging.info(
            f"Loss {loss:.4f}, train R2 {train_m['r2']:.4f}, "
            f"val R2 {val_m['r2']:.4f} "
            f"(N_train={len(train_emb)}, N_val={len(val_emb)})"
        )
        return loss, metrics

    def _train_on_all(self, embeddings, objectives):
        """Train on the entire dataset (no validation)."""
        loss = self.model.fit_dict(
            embedding_dict=embeddings,
            scores_dict=objectives,
            epochs=self.best_hyperparams.get("epochs", 100),
            learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
            batch_size=self.best_hyperparams.get("batch_size", 16),
            device=self.device,
        )
        preds = self.model.predict_dict(embeddings, device=self.device)
        m = self._compute_model_metrics(preds, objectives)
        metrics = {"r2": m["r2"], "mae": m["mae"]}
        logging.info(f"Loss {loss:.4f}, R2 {m['r2']:.4f}, N={len(embeddings)}")
        return loss, metrics

    def _split_train_validation(
        self, docked_embeddings: dict, objectives: dict, num_validate: int
    ):
        """
        Split the available data into training and validation sets.
        """
        peptides = list(objectives.keys())
        val_indices = np.random.choice(len(peptides), num_validate, replace=False)
        val_peptides = [peptides[i] for i in val_indices]
        train_peptides = [p for p in peptides if p not in val_peptides]
        train_embeddings = {p: docked_embeddings[p] for p in train_peptides}
        train_objectives = {p: objectives[p] for p in train_peptides}
        val_embeddings = {p: docked_embeddings[p] for p in val_peptides}
        val_objectives = {p: objectives[p] for p in val_peptides}

        logging.info(
            f"Split data into {len(train_peptides)} training and {len(val_peptides)} validation samples"
        )

        return train_embeddings, train_objectives, val_embeddings, val_objectives

    def _print_top_performers(self, objectives: dict, top_n: int = 10):
        sorted_peptides = sorted(objectives.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        logging.info(f"Top {top_n} peptides:")
        logging.info(f"{'Peptide':<20} | {'Objective':<10} ")
        logging.info("-" * 60)
        for peptide, obj_value in sorted_peptides:
            logging.info(f"{peptide:<20} | {obj_value:<10.4f} ")

