import json
from pathlib import Path
import pickle

from typing import Callable, List, Optional, Dict, Any, Union
from bopep import _AMINO_ACIDS
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
from bopep.bayesian_optimization.utils import (_validate_dependencies, _validate_args, _validate_surrogate_model_kwargs)
from bopep.bayesian_optimization.pdb_utils import _check_binding_site_residue_indices
from bopep.bayesian_optimization.checkpointing import _next_checkpoint_dir, _save_checkpoint, _copy_logs_to_checkpoint, _setup_checkpoint_dir, _rebuild_logs_from_csvs, _validate_checkpoint
from bopep.bayesian_optimization.train_validate_utils import _compute_model_metrics, _compute_split_indices, _train_and_validate, _split_train_validation, _train_on_all
from bopep.bayesian_optimization.selection import PeptideSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
import torch
import logging


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
        checkpoint_interval: int = 5,
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
            checkpoint_interval: Number of iterations between automatic checkpoints (default: 5)
        """
        _validate_dependencies()

        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
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
        self.checkpoint_interval = checkpoint_interval
        
        # checkpointing functions
        self._next_checkpoint_dir = _next_checkpoint_dir.__get__(self, BoPep)
        self._save_checkpoint = _save_checkpoint.__get__(self, BoPep)
        self._copy_logs_to_checkpoint = _copy_logs_to_checkpoint.__get__(self, BoPep)
        self._setup_checkpoint_dir = _setup_checkpoint_dir.__get__(self, BoPep)
        self._rebuild_logs_from_csvs = _rebuild_logs_from_csvs.__get__(self, BoPep)

        # training and validation functions
        self._compute_model_metrics = _compute_model_metrics.__get__(self, BoPep)
        self._compute_split_indices = _compute_split_indices.__get__(self, BoPep)
        self._train_and_validate = _train_and_validate.__get__(self, BoPep)
        self._split_train_validation = _split_train_validation.__get__(self, BoPep)
        self._train_on_all = _train_on_all.__get__(self, BoPep)


    def optimize(
        self,
        schedule: List[Dict[str, Any]],
        batch_size: int,
        target_structure_path: str,
        embeddings: Optional[Dict[str, Any]] = None,
        num_initial: Optional[int] = 10,
        n_validate: Optional[Union[float, int]] = None,
        binding_site_residue_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        initial_peptides: Optional[List[str]] = None,
        initial_method: str = "kmeans",
        assume_zero_indexed: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
        template_pdbs: Optional[Dict[str, str]] = None,
    ):
        """
        Runs Bayesian optimization on peptide sequences.

        Args:
            schedule: List of dictionaries defining the optimization phases.
                Each dict should have 'acquisition' and 'iterations' keys.
            batch_size: Number of peptides to dock in each iteration.
            target_structure_path: Path to the target structure PDB file.
            embeddings: Dictionary of peptide embeddings {peptide: embedding}. Required for fresh initialization.
            num_initial: Number of initial peptides to dock and score.
            n_validate: Number of peptides to use for validation.
                If None, no validation is performed.
            binding_site_residue_indices: List of residue indices defining the binding site,
                or dict mapping peptides to their specific binding site residue indices.
            initial_peptides: Optional list of initial peptides to dock.
            assume_zero_indexed: If True, assumes residue indices are zero-indexed.
            checkpoint_path: Path to a checkpoint directory to continue from. If none is provided,
                a fresh optimization will be started.
            template_pdbs: Optional dictionary mapping peptide sequences to template PDB paths.

        """
        continue_from_checkpoint = False if checkpoint_path is None else True
        self.template_pdbs = template_pdbs
        self.checkpoint_path = checkpoint_path
        self._setup_checkpoint_dir(continue_from_checkpoint)
        self.logger = Logger(log_dir=self.log_dir, overwrite_logs=self.overwrite_logs, continue_from_checkpoint=continue_from_checkpoint)
        self.initial_method = initial_method
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
        for peptide in peptides:
            if not all(aa in _AMINO_ACIDS for aa in peptide):
                raise ValueError(f"Invalid amino acids in peptide sequence: {peptide}. Allowed amino acids are: {_AMINO_ACIDS}")
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
                embeddings=self.embeddings, num_initial=num_initial, random_state=42, method=self.initial_method
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
        
        # Save initial checkpoint
        logging.info("Creating initial checkpoint")
        self._save_checkpoint(0, force_embeddings=True)


    def _continue_init(self) -> int:
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint directory found at {self.checkpoint_path}")
        _validate_checkpoint(checkpoint_path=checkpoint_path)
        
        embeddings_path = checkpoint_path / "embeddings.pkl"
        meta_path = checkpoint_path / "metadata.json"
        
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
        self._rebuild_logs_from_csvs(checkpoint_path)

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

        # Save initial checkpoint for continued run
        logging.info("Creating initial checkpoint for continued run")
        self._save_checkpoint(last_iteration)

        logging.info(f"Resumed from iteration {last_iteration} with "
                    f"{len(self.docked_peptides)} peptides docked")
        
        return last_iteration

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
                
                # Save checkpoint every N iterations
                if global_iteration % self.checkpoint_interval == 0:
                    logging.info(f"Creating regular checkpoint (every {self.checkpoint_interval} iterations)")
                    self._save_checkpoint(global_iteration)
                
                if not not_docked_peptides:
                    logging.info("All peptides docked, stopping early")
                    break
                
        logging.info(f"Optimization completed at global iteration {global_iteration}")
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
                iteration=global_iteration,
                acquisition_name=acquisition,
            )
        # Final checkpoint save - ensure embeddings are included
        self._save_checkpoint(global_iteration, force_embeddings=True)

    def _score_batch(self, docked_dirs: list):
        """
        Scores a batch (list of dirs) and outputs the score dict
        """
        if self.custom_scorer:
            return self.custom_scorer(docked_dirs)

        scores_to_include = self.scoring_kwargs.get("scores_to_include", [])
        binding_site_distance_threshold = self.scoring_kwargs.get("binding_site_distance_threshold", 5)
        required_n_contact_residues = self.scoring_kwargs.get(
            "required_n_contact_residues", 5
        )

        if not scores_to_include:
            raise ValueError(
                "No scores to include for scoring. Please specify valid scores in 'scores_to_include'."
            )

        if isinstance(self.binding_site_residue_indices, dict):
            peptides_in_batch = []
            for dir_path in docked_dirs:
                dir_name = Path(dir_path).name
                if '_' in dir_name:
                    peptide = dir_name.split('_')[-1]
                else:
                    peptide = dir_name
                peptides_in_batch.append(peptide)

            new_scores = {}
            for dir_path, peptide in zip(docked_dirs, peptides_in_batch):
                if peptide in self.binding_site_residue_indices:
                    peptide_binding_sites = self.binding_site_residue_indices[peptide]
                else:
                    logging.warning(f"No binding site defined for peptide {peptide}, skipping scoring")
                    continue
                
                peptide_scores = self.scorer.score_batch(
                    scores_to_include=scores_to_include,
                    inputs=[dir_path],
                    input_type="colab_dir",
                    binding_site_residue_indices=peptide_binding_sites,
                    n_jobs=1,
                    binding_site_distance_threshold=binding_site_distance_threshold,
                    required_n_contact_residues=required_n_contact_residues,
                    template_pdbs=self.template_pdbs
                )
                new_scores.update(peptide_scores)
        else:
            new_scores = self.scorer.score_batch(
                scores_to_include=scores_to_include,
                inputs=docked_dirs,
                input_type="colab_dir",
                binding_site_residue_indices=self.binding_site_residue_indices,
                n_jobs=self.scoring_kwargs.get("n_jobs", 12),
                binding_site_distance_threshold=binding_site_distance_threshold,
                required_n_contact_residues=required_n_contact_residues,
                template_pdbs=self.template_pdbs,
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

    def _optimize_hyperparameters(self, iteration: int, embeddings: dict, objectives: dict):
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
            objective_dict=objectives,
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

        if hasattr(self, "model"):
            try:
                self.model.cpu()
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Couldn't clean up previous model: {e}")

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

    def _print_top_performers(self, objectives: dict, top_n: int = 10):
        sorted_peptides = sorted(objectives.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        logging.info(f"Top {top_n} peptides:")
        logging.info(f"{'Peptide':<20} | {'Objective':<10} ")
        logging.info("-" * 60)
        for peptide, obj_value in sorted_peptides:
            logging.info(f"{peptide:<20} | {obj_value:<10.4f} ")

