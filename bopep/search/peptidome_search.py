import json
from pathlib import Path
import pickle

from typing import Callable, List, Optional, Dict, Any, Union
from bopep.docking.docker import Docker
from bopep.scoring.complex_scorer import ComplexScorer
from bopep.surrogate_model.manager import SurrogateModelManager
from bopep.logging.logger import Logger
from bopep.bayes.acquisition import AcquisitionFunction
from bopep.search.utils import (_validate_args, _validate_surrogate_model_kwargs)
from bopep.search.structure_utils import _check_binding_site_residue_indices
from bopep.search.checkpointing import _next_checkpoint_dir, _save_checkpoint, _copy_logs_to_checkpoint, _setup_checkpoint_dir, _rebuild_logs_from_csvs, _validate_checkpoint
from bopep.search.selection import SequenceSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.config import Config
import torch
import logging

_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PeptidomeSearch:

    def __init__(
        self,
        surrogate_model_kwargs: Dict[str, Any] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        hpo_kwargs: Optional[Dict[str, Any]] = None,
        log_dir: str = None,
        overwrite_logs: Optional[bool] = None,
        custom_scorer: Optional[Callable] = None,
        min_validation_samples: int = None,
        min_training_samples: int = None,
        checkpoint_interval: int = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the PeptidomeSearch optimizer with various configuration options.

        Args:
            surrogate_model_kwargs: Configuration for the surrogate model including:
                - network_type: 'mlp', 'bilstm', or 'bigru'
                - model_type: 'nn_ensemble', 'mc_dropout', 'deep_evidential', or 'mve'
                - multi_model: If True, use separate models per objective (default: False)
            objective_function: Custom objective function (defaults to maximize iptm/pae and minimize dG and rosetta_score, see paper)
            objective_function_kwargs: Parameters for the objective function
            docker_kwargs: Configuration for the Docker component
            hpo_kwargs: Configuration for hyperparameter optimization
            log_dir: Directory for logging output
            custom_scorer: Optional function that takes docking directories and
                    returns score dictionaries. If provided, this will be used
                    instead of the default scorer.
            checkpoint_interval: Number of iterations between automatic checkpoints (default: 5)
            config: Optional Config object for PeptidomeSearch. If not provided, defaults will be loaded.
        """
        # Initialize or load config
        if config is None:
            config = Config(script="PeptidomeSearch")  # Load defaults
        self.config = config
        
        # Get flattened config for easy parameter access
        cfg = self.config.flatten()
        
        # Helper function to get parameter value (user override > config)
        def get_param(user_val, config_key):
            return user_val if user_val is not None else cfg.get(config_key)

        # Get surrogate model kwargs
        if surrogate_model_kwargs is not None:
            self.surrogate_model_kwargs = surrogate_model_kwargs
        else:
            # Extract from flattened config
            self.surrogate_model_kwargs = {
                'model_type': cfg.get('surrogate_model.model_type'),
                'network_type': cfg.get('surrogate_model.network_type'),
            }
        
        # Get HPO kwargs
        if hpo_kwargs is not None:
            self.hpo_kwargs = hpo_kwargs
        else:
            # Extract from flattened config
            self.hpo_kwargs = {
                'n_trials': cfg.get('hpo.n_trials'),
                'n_splits': cfg.get('hpo.n_splits'),
                'random_state': cfg.get('hpo.random_state'),
                'hpo_interval': cfg.get('hpo.hpo_interval'),
            }
        
        # Get scoring kwargs
        if scoring_kwargs is not None:
            self.scoring_kwargs = scoring_kwargs
        else:
            # Extract all scoring.* keys from flattened config
            self.scoring_kwargs = {
                k.replace('scoring.', ''): v 
                for k, v in cfg.items() 
                if k.startswith('scoring.')
            }
        
        # Get docker kwargs
        if docker_kwargs is not None:
            self.docker_kwargs = docker_kwargs
        else:
            # Extract all docker.* keys from flattened config
            self.docker_kwargs = {
                k.replace('docker.', ''): v 
                for k, v in cfg.items() 
                if k.startswith('docker.')
            }

        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.custom_scorer = custom_scorer

        # Initialize components
        self.scorer = ComplexScorer()
        self.docker = Docker(self.docker_kwargs)
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = SequenceSelector()
        self.scores_to_objective = ScoresToObjective()

        # Initialize surrogate model manager
        self.surrogate_manager = SurrogateModelManager(
            surrogate_model_kwargs=self.surrogate_model_kwargs,
            device=None  # Will be set automatically in run()
        )

        _validate_surrogate_model_kwargs(self.surrogate_model_kwargs)
        
        # Get logging parameters from config
        self.log_dir = get_param(log_dir, 'logging.log_dir')
        self.overwrite_logs = get_param(overwrite_logs, 'logging.overwrite_logs')

        # Get validation parameters from config
        self.min_validation_samples = get_param(min_validation_samples, 'validation.min_validation_samples')
        self.min_training_samples = get_param(min_training_samples, 'validation.min_training_samples')
        
        # Get checkpointing parameters from config
        self.checkpoint_interval = get_param(checkpoint_interval, 'checkpointing.checkpoint_interval')
        
        # checkpointing functions
        self._next_checkpoint_dir = _next_checkpoint_dir.__get__(self, PeptidomeSearch)
        self._save_checkpoint = _save_checkpoint.__get__(self, PeptidomeSearch)
        self._copy_logs_to_checkpoint = _copy_logs_to_checkpoint.__get__(self, PeptidomeSearch)
        self._setup_checkpoint_dir = _setup_checkpoint_dir.__get__(self, PeptidomeSearch)
        self._rebuild_logs_from_csvs = _rebuild_logs_from_csvs.__get__(self, PeptidomeSearch)


    def run(
        self,
        schedule: List[Dict[str, Any]],
        batch_size: int = None,
        target_structure_path: str = None,
        embeddings: Optional[Dict[str, Any]] = None,
        num_initial: Optional[int] = None,
        n_validate: Optional[Union[float, int]] = None,
        binding_site_residue_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        initial_sequences: Optional[List[str]] = None,
        initial_method: str = None,
        assume_zero_indexed: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
        template_structures: Optional[Dict[str, str]] = None,
    ):
        """
        Runs Bayesian optimization on sequence sequences.

        Args:
            schedule: List of dictionaries defining the optimization phases.
                Each dict should have 'acquisition' and 'iterations' keys.
                Optionally can include 'acquisition_kwargs' for multi-objective
                acquisition functions (e.g., parego_chebyshev_ei).
            batch_size: Number of sequences to dock in each iteration.
            target_structure_path: Path to the target structure PDB file.
            embeddings: Dictionary of sequence embeddings {sequence: embedding}. Required for fresh initialization.
            num_initial: Number of initial sequences to dock and score.
            n_validate: Number of sequences to use for validation.
                If None, no validation is performed.
            binding_site_residue_indices: List of residue indices defining the binding site,
                or dict mapping sequences to their specific binding site residue indices.
            initial_sequences: Optional list of initial sequences to dock.
            initial_method: Method for selecting initial sequences ('kmeans' or 'random').
            assume_zero_indexed: If True, assumes residue indices are zero-indexed.
            checkpoint_path: Path to a checkpoint directory to continue from. If none is provided,
                a fresh optimization will be started.
            template_structures: Optional dictionary mapping sequence sequences to template PDB paths.

        """
        # Get flattened config for parameter access
        cfg = self.config.flatten()
        
        # Helper function to get parameter value (user override > config)
        def get_param(user_val, config_key):
            return user_val if user_val is not None else cfg.get(config_key)
        
        # Get parameters from config or user overrides
        batch_size = get_param(batch_size, 'batch_size')
        num_initial = get_param(num_initial, 'num_initial')
        initial_method = get_param(initial_method, 'initial_method')
        n_validate = get_param(n_validate, 'validation.n_validate')
        
        if batch_size is None:
            raise ValueError("batch_size must be provided either as argument or in config")
        if target_structure_path is None:
            raise ValueError("target_structure_path must be provided")
        
        # Update config with actually-used values for reproducibility
        self.config.update_from_used_values(
            batch_size=batch_size,
            num_initial=num_initial,
            initial_method=initial_method,
            **{'validation.n_validate': n_validate},
            **{'validation.min_validation_samples': self.min_validation_samples},
            **{'validation.min_training_samples': self.min_training_samples},
            **{'logging.log_dir': self.log_dir},
            **{'logging.overwrite_logs': self.overwrite_logs},
            **{'checkpointing.checkpoint_interval': self.checkpoint_interval},
        )
        
        # Save config to log directory
        config_path = self.config.save(self.log_dir, filename='bopep_config_used.yaml')
        logging.info(f"Saved configuration to {config_path}")
        
        continue_from_checkpoint = False if checkpoint_path is None else True
        self.template_structures = template_structures
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
        
        # Set device for surrogate model manager
        self.surrogate_manager.device = self.device

        if not continue_from_checkpoint:
            if not embeddings:
                raise ValueError("embeddings must be provided for fresh initialization")
            logging.info("Fresh initialization")
            self._fresh_init(initial_sequences, num_initial, assume_zero_indexed)
            starting_iteration = 0
        else:
            logging.info("Continuing from checkpoint")
            starting_iteration = self._continue_init()
        
        logging.info("Starting optimization loop")
        self._run_phase_loop(
            schedule=schedule,
            all_logged_objectives=self.all_logged_objectives,
            docked_sequences=self.docked_sequences,
            not_docked_sequences=self.not_docked_sequences,
            scores=self.scores,
            batch_size=batch_size,
            n_validate=n_validate,
            starting_global_iteration=starting_iteration,
        )

    def _fresh_init(
        self,
        initial_sequences: Optional[List[str]] = None,
        num_initial: int = 10,
        assume_zero_indexed: Optional[bool] = None,
    ):
        """
        Initializes the PeptidomeSearch optimizer for a fresh search
        """
        sequences = list(self.embeddings.keys())
        for sequence in sequences:
            if not all(aa in _AMINO_ACIDS for aa in sequence):
                raise ValueError(f"Invalid amino acids in sequence sequence: {sequence}. Allowed amino acids are: {_AMINO_ACIDS}")
        self.docked_sequences = set()
        self.not_docked_sequences = set(sequences)
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
        
        # Set max_seq_len based on maximum sequence length in the dataset if not already specified
        if 'max_seq_len' not in self.surrogate_model_kwargs:
            max_seq_len = max(len(seq) for seq in self.embeddings.keys())
            self.surrogate_model_kwargs['max_seq_len'] = max_seq_len
            logging.info(f"Set max_seq_len to {max_seq_len} based on dataset")


        self.best_hyperparams = None

        if initial_sequences is None:
            initial_sequences = self.selector.select_initial_sequences(
                embeddings=self.embeddings, num_initial=num_initial, random_state=42, method=self.initial_method
            )

        logging.info("Docking initial sequences")
        docked_dirs = self.docker.dock_sequences(initial_sequences)
        self.docked_sequences.update(initial_sequences)
        self.not_docked_sequences.difference_update(initial_sequences)

        logging.info(f"Scoring {len(initial_sequences)} initial sequences")
        initial_scores = self._score_batch(docked_dirs=docked_dirs)
        self.scores.update(initial_scores)
        self.logger.log_scores(initial_scores, iteration=0, acquisition_name="initial")

        objectives = self.scores_to_objective.create_objective(
            self.scores, self.objective_function, **self.objective_function_kwargs
        )
        self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")
        self.all_logged_objectives = set(objectives.keys())

        # Optimize hyperparameters and initialize model using surrogate manager
        docked_embs = {p: self.embeddings[p] for p in self.docked_sequences}
        # Filter hpo_kwargs to only include parameters accepted by the manager
        manager_hpo_kwargs = {k: v for k, v in self.hpo_kwargs.items() 
                             if k in ['n_trials', 'n_splits', 'random_state']}
        self.best_hyperparams = self.surrogate_manager.optimize_hyperparameters(
            docked_embs, objectives, **manager_hpo_kwargs
        )
        
        # Initialize the model with optimized hyperparameters
        self.surrogate_manager.initialize_model(self.best_hyperparams, docked_embs, objectives)
        
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
        docked_embs = {p: self.embeddings[p] for p in self.docked_sequences}
        
        last_iteration = meta["global_iteration"]
        
        logging.info("Running hyperparameter optimization...")
        # Optimize hyperparameters and initialize model using surrogate manager
        # Filter hpo_kwargs to only include parameters accepted by the manager
        manager_hpo_kwargs = {k: v for k, v in self.hpo_kwargs.items() 
                             if k in ['n_trials', 'n_splits', 'random_state']}
        self.best_hyperparams = self.surrogate_manager.optimize_hyperparameters(
            docked_embs, objectives, **manager_hpo_kwargs
        )
        
        # Initialize the model with optimized hyperparameters
        self.surrogate_manager.initialize_model(self.best_hyperparams, docked_embs, objectives)

        # Save initial checkpoint for continued run
        logging.info("Creating initial checkpoint for continued run")
        self._save_checkpoint(last_iteration)

        logging.info(f"Resumed from iteration {last_iteration} with "
                    f"{len(self.docked_sequences)} sequences docked")
        
        return last_iteration

    def _run_phase_loop(
        self,
        schedule: List[Dict[str, Any]],
        all_logged_objectives: set,
        docked_sequences: set,
        not_docked_sequences: set,
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
                # Use embeddings from current docked sequences for initialization
                current_docked_embs = {p: self.embeddings[p] for p in self.docked_sequences}
                
                # Turn scores into a scalarized score dict of sequence: score
                objectives = self.scores_to_objective.create_objective(
                    scores, self.objective_function, **self.objective_function_kwargs
                )
                
                self.surrogate_manager.initialize_model(self.best_hyperparams, current_docked_embs, objectives)

                # Log the new objective values
                new_objective_sequences = set(objectives.keys()) - all_logged_objectives
                new_objectives = {
                    sequence: objectives[sequence] for sequence in new_objective_sequences
                }

                if new_objectives:
                    self.logger.log_objectives(
                        new_objectives,
                        iteration=global_iteration,
                        acquisition_name=acquisition,
                    )
                    all_logged_objectives.update(new_objectives.keys())

                # Train the model on *only the sequences we have scores for*
                docked_embeddings = {p: self.embeddings[p] for p in docked_sequences}

                # Run hyperparameter optimization every N steps
                if global_iteration % self.hpo_kwargs.get("hpo_interval", 10) == 0:
                    self.surrogate_manager.optimize_hyperparameters(
                        embeddings=docked_embeddings, 
                        objectives=objectives,
                        n_trials=self.hpo_kwargs.get("n_trials", 20),
                        n_splits=self.hpo_kwargs.get("n_splits", 3),
                        random_state=self.hpo_kwargs.get("random_state", 42),
                        iteration=global_iteration
                    )
                    self.surrogate_manager.initialize_model(embeddings=docked_embeddings, objectives=objectives)

                    # Log hyperparameters
                    if self.surrogate_manager.best_hyperparams:
                        self.logger.log_hyperparameters(
                            global_iteration,
                            self.surrogate_manager.best_hyperparams,
                            model_type=self.surrogate_model_kwargs["model_type"],
                            network_type=self.surrogate_model_kwargs["network_type"],
                        )

                logging.info(
                    f"Model will be trained (potentially validated) on {len(docked_sequences)} sequences"
                )

                # Train the model with automatic (optional) validation split
                metrics = self.surrogate_manager.train_with_validation_split(
                    embeddings=docked_embeddings,
                    objectives=objectives,
                    validation_size=n_validate,
                    min_training_samples=self.min_training_samples,
                    min_validation_samples=self.min_validation_samples
                )

                # Extract loss - use validation loss if available, otherwise training loss
                loss = metrics["val_mse"] if metrics["val_mse"] is not None else metrics["train_mse"]

                # Log the loss and metrics
                self.logger.log_model_metrics(loss, global_iteration, metrics)

                # Predict for *the not-yet-docked* sequences
                candidate_embeddings = {
                    p: self.embeddings[p] for p in not_docked_sequences
                }
                predictions = self.surrogate_manager.predict(candidate_embeddings)

                # Log predictions
                self.logger.log_predictions(predictions, global_iteration)

                # Compute acquisition
                acquisition_kwargs = phase.get("acquisition_kwargs", {})
                acquisition_values = self.acquisition_function_obj.compute_acquisition(
                    predictions=predictions,
                    acquisition_function=acquisition,
                    **acquisition_kwargs
                )  # acquisition_values is {sequence: acquisition_value}

                # Log acquisition
                self.logger.log_acquisition(
                    acquisition_values=acquisition_values,
                    acquisition_name=acquisition,
                    iteration=global_iteration,
                )

                # Select the next set of sequences to dock
                next_sequences = self.selector.select_next_sequences(
                    sequences=not_docked_sequences,
                    embeddings=candidate_embeddings,
                    acquisition_values=acquisition_values,
                    n_select=batch_size,
                )

                # Dock them
                docked_dirs = self.docker.dock_sequences(next_sequences)
                docked_sequences.update(next_sequences)
                not_docked_sequences.difference_update(next_sequences)

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
                
                if not not_docked_sequences:
                    logging.info("All sequences docked, stopping early")
                    break
                
        logging.info(f"Optimization completed at global iteration {global_iteration}")
        # Loop is over
        final_objectives = self.scores_to_objective.create_objective(
            scores, self.objective_function, **self.objective_function_kwargs
        )
        final_new_objective_sequences = (
            set(final_objectives.keys()) - all_logged_objectives
        )
        final_new_objectives = {
            sequence: final_objectives[sequence]
            for sequence in final_new_objective_sequences
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
        receptor_chain = self.scoring_kwargs.get("receptor_chain", "A")
        sequence_chain = self.scoring_kwargs.get("sequence_chain", "B")

        if not scores_to_include:
            raise ValueError(
                "No scores to include for scoring. Please specify valid scores in 'scores_to_include'."
            )

        if isinstance(self.binding_site_residue_indices, dict):
            sequences_in_batch = []
            for dir_path in docked_dirs:
                dir_name = Path(dir_path).name
                if '_' in dir_name:
                    sequence = dir_name.split('_')[-1]
                else:
                    sequence = dir_name
                sequences_in_batch.append(sequence)

            new_scores = {}
            for dir_path, sequence in zip(docked_dirs, sequences_in_batch):
                if sequence in self.binding_site_residue_indices:
                    sequence_binding_sites = self.binding_site_residue_indices[sequence]
                else:
                    logging.warning(f"No binding site defined for sequence {sequence}, skipping scoring")
                    continue
                
                sequence_scores = self.scorer.score_batch(
                    scores_to_include=scores_to_include,
                    inputs=[dir_path],
                    input_type="colab_dir",
                    binding_site_residue_indices=sequence_binding_sites,
                    n_jobs=1,
                    binding_site_distance_threshold=binding_site_distance_threshold,
                    required_n_contact_residues=required_n_contact_residues,
                    template_structures=self.template_structures,
                    receptor_chain=receptor_chain,
                    sequence_chain=sequence_chain,
                )
                new_scores.update(sequence_scores)
        else:
            new_scores = self.scorer.score_batch(
                scores_to_include=scores_to_include,
                inputs=docked_dirs,
                input_type="colab_dir",
                binding_site_residue_indices=self.binding_site_residue_indices,
                n_jobs=self.scoring_kwargs.get("n_jobs", 1), # Default to 1 job unless specified (not safe otherwise)
                binding_site_distance_threshold=binding_site_distance_threshold,
                required_n_contact_residues=required_n_contact_residues,
                template_structures=self.template_structures,
                receptor_chain=receptor_chain,
                sequence_chain=sequence_chain,
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
                        f"For {network_type}, each sequence embedding must be 2D. "
                        f"Sequence '{pep}' has shape {emb.shape}."
                    )
            elif network_type == "mlp":
                if emb.ndim != 1:
                    raise ValueError(
                        f"For mlp, each sequence embedding must be 1D. "
                        f"Sequence '{pep}' has shape {emb.shape}."
                    )

    def _print_top_performers(self, objectives: dict, top_n: int = 10):
        if not objectives:
            return
        
        # Check if multiobjective case
        sample_obj = next(iter(objectives.values()))
        if isinstance(sample_obj, dict):
            # Multi-objective case: show top performers for each objective
            obj_names = list(sample_obj.keys())
            logging.info(f"Top {top_n} sequences (multiobjective):")
            
            for obj_name in obj_names:
                logging.info(f"\n--- {obj_name} ---")
                sorted_sequences = sorted(objectives.items(), 
                                       key=lambda x: x[1][obj_name], reverse=True)[:top_n]
                logging.info(f"{'Sequence':<20} | {obj_name:<15}")
                logging.info("-" * 40)
                for sequence, obj_dict in sorted_sequences:
                    logging.info(f"{sequence:<20} | {obj_dict[obj_name]:<15.4f}")
        else:
            # Single objective case (original)
            sorted_sequences = sorted(objectives.items(), key=lambda x: x[1], reverse=True)[:top_n]
            logging.info(f"Top {top_n} sequences:")
            logging.info(f"{'Sequence':<20} | {'Objective':<10} ")
            logging.info("-" * 60)
            for sequence, obj_value in sorted_sequences:
                logging.info(f"{sequence:<20} | {obj_value:<10.4f} ")

