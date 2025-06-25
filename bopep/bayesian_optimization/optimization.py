import subprocess
from typing import Callable, List, Optional, Dict, Any
from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import (
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    tune_hyperparams,
    MVE,
)
from bopep.embedding.embedder import Embedder
from bopep.logging.logger import Logger
from bopep.bayesian_optimization.acquisition_functions import AcquisitionFunction
from bopep.bayesian_optimization.selection import PeptideSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.bayesian_optimization.utils import check_starting_index_in_pdb
from bopep.docking.utils import extract_sequence_from_pdb
import pyrosetta
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
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        hpo_kwargs: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
        overwrite_logs: Optional[bool] = None,
        custom_scorer: Optional[Callable] = None, 
          ):
        """
        Initialize the BoPep optimizer with various configuration options.

        Args:
            surrogate_model_kwargs: Configuration for the surrogate model including:
                - network_type: 'mlp', 'bilstm', or 'bigru'
                - model_type: 'nn_ensemble', 'mc_dropout', 'deep_evidential', or 'mve'
            objective_function: Custom objective function (defaults to maximize iptm/pae and minimize dG and rosetta_score, see paper)
            objective_function_kwargs: Parameters for the objective function
            embedding_kwargs: Configuration for peptide embedding generation
            docker_kwargs: Configuration for the Docker component
            hpo_kwargs: Configuration for hyperparameter optimization
            log_dir: Directory for logging output
            custom_scorer: Optional function that takes docking directories and
                    returns score dictionaries. If provided, this will be used
                    instead of the default scorer.
        """
        self._setup()

        # Set default kwargs with proper handling of mutable defaults
        self.surrogate_model_kwargs = self._get_default_surrogate_model_kwargs()
        if surrogate_model_kwargs:
            self.surrogate_model_kwargs.update(surrogate_model_kwargs)

        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.embedding_kwargs = embedding_kwargs or {}
        self.docker_kwargs = docker_kwargs or {}
        self.hpo_kwargs = hpo_kwargs or {}
        self.scoring_kwargs = scoring_kwargs or {}
        self.custom_scorer = custom_scorer

        # Initialize components
        self.embedder = Embedder()
        self.scorer = Scorer()
        self.docker = Docker(self.docker_kwargs)
        self.logger = Logger(log_dir=log_dir, overwrite_logs=overwrite_logs)
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = PeptideSelector()
        self.scores_to_objective = ScoresToObjective()

        self.embeddings_save_path = log_dir + "/embeddings.npy"
        
        # Store the Optuna study for warm-starting
        self.previous_study = None

        self.available_acquistion_functions = [
            "expected_improvement",
            "standard_deviation",
            "upper_confidence_bound",
            "probability_of_improvement",
            "mean",
        ]

        self._validate_surrogate_model_kwargs()
        self.log_dir = log_dir

    def optimize(
        self,
        peptides: List[str],
        target_structure_path: str,
        num_initial: int = 10,
        num_validate: Optional[int] = None, # new arg
        batch_size: int = 4,
        binding_site_residue_indices: Optional[List[int]] = None,
        schedule: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[Dict[str, Any]] = None,
        initial_peptides: Optional[List[str]] = None,
        assume_zero_indexed: Optional[bool] = None,
    ):
        """
        Runs Bayesian optimization on peptide sequences.

        Args:
            peptides: List of peptide sequences to optimize
            target_structure_path: Path to the target protein structure file
            num_initial: Number of initial peptides to sample before optimization
            num_validate: Number of peptides to use for validation (optional). When specified,
                          the model will be trained on the remaining data and metrics will be
                          reported for both training and validation sets.
            batch_size: Number of peptides to evaluate in each iteration
            binding_site_residue_indices: List of residue indices defining the binding site
            schedule: List of optimization phases, each with:
                - acquisition: The acquisition function to use
                - iterations: Number of iterations for this phase
            embeddings: Pre-computed embeddings for peptides (optional). These should be COMPLETELY PROCESSED.
            initial_peptides: Specific peptides to start with instead of k-means selection

        Returns:
            None: Results are logged through the logger component
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set default schedule if none provided
        if schedule is None:
            schedule = [
                {"acquisition": "standard_deviation", "iterations": 10},
                {"acquisition": "expected_improvement", "iterations": 10},
            ]

        # Validate schedule
        self._validate_schedule(schedule)

        self.docker.set_target_structure(target_structure_path)

        self.binding_site_residue_indices = self._check_binding_site_residue_indices(
            binding_site_residue_indices, target_structure_path, assume_zero_indexed=assume_zero_indexed
        )  # Checks if pdb starting index is 0, adjusts if needed.

        docked_peptides = set()
        not_docked_peptides = set(peptides)

        # Keep a dict of {peptide: score}, to update after each docking/score
        scores = dict()

        # Generate embeddings for all peptides if not provided
        if embeddings is None:
            self.embeddings = self._generate_embeddings(peptides) # loads if str
        else:
            # If you passed precomputed embeddings, we can still do shape checks:
            self._check_embedding_shapes(embeddings)
            self.embeddings = embeddings

        # Once embeddings exist, set input_dim based on shape
        if self.surrogate_model_kwargs["network_type"] == "mlp":
            # For MLP we expect 1D embeddings => input_dim is the length
            # (If user gave shape mismatch, `_check_embedding_shapes` will raise)
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = len(self.embeddings[any_key])
        else:
            # For LSTM/GRU we expect 2D embeddings => input_dim is the last dimension
            any_key = next(iter(self.embeddings))
            self.surrogate_model_kwargs["input_dim"] = self.embeddings[any_key].shape[-1]

        logging.info(f"Embeddings dimension: {self.surrogate_model_kwargs['input_dim']}")

        # Store best hyperparameters
        self.best_hyperparams = None

        # 1) Select initial peptides for docking
        if initial_peptides is None:
            initial_peptides = self.selector.select_initial_peptides(
                embeddings=self.embeddings, num_initial=num_initial, random_state=42
            )
        # Dock them
        logging.info("Docking initial peptides")
        docked_dirs = self.docker.dock_peptides(initial_peptides)
        docked_peptides.update(initial_peptides)
        not_docked_peptides.difference_update(initial_peptides)

        # Score them
        logging.info(f"Scoring {len(initial_peptides)} initial peptides")
        initial_scores = self._score_batch(docked_dirs=docked_dirs)
        scores.update(initial_scores)

        # Log scores
        self.logger.log_scores(initial_scores, iteration=0, acquisition_name="initial")

        # Create and log initial objectives
        objectives = self.scores_to_objective.create_objective(
            scores, self.objective_function, **self.objective_function_kwargs
        )
        self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")
        all_logged_objectives = set(objectives.keys())

        # Optimize and create initial model
        self._optimize_hyperparameters(0, {p: self.embeddings[p] for p in docked_peptides}, objectives)
        self._initialize_model(self.best_hyperparams)

        # 2) Main BO Loop over phases
        logging.info("Starting optimization loop")
        for phase_index, phase in enumerate(schedule, start=1):
            acquisition = phase["acquisition"]
            iterations = phase["iterations"]

            # (Optional) double-check acquisition name:
            if acquisition not in self.available_acquistion_functions:
                raise ValueError(
                    f"Invalid acquisition function: {acquisition}. "
                    f"Must be one of {self.available_acquistion_functions}"
                )
            
            for iteration in range(1, iterations+1):
                logging.info("=" * 60)
                logging.info(f"Starting iteration {iteration} out of {iterations} of phase {phase_index} with acquisition '{acquisition}'")

                # Initialize fresh model for each iteration
                self._initialize_model(self.best_hyperparams)

                # 2.0) Turn scores into a scalarized score dict of peptide: score
                objectives = self.scores_to_objective.create_objective(
                    scores, self.objective_function, **self.objective_function_kwargs
                )
                
                # Log the new objective values
                new_objective_peptides = set(objectives.keys()) - all_logged_objectives
                new_objectives = {peptide: objectives[peptide] for peptide in new_objective_peptides}

                if new_objectives:
                    self.logger.log_objectives(new_objectives, iteration=iteration, acquisition_name=acquisition)
                    all_logged_objectives.update(new_objectives.keys())

                # 2.1) Train the model on *only the peptides we have scores for*
                docked_embeddings = {p: self.embeddings[p] for p in docked_peptides}

                # Run hyperparameter optimization every N steps
                if  (
                    iteration % self.hpo_kwargs.get("hpo_interval", 10) == 0
                ):
                    self._optimize_hyperparameters(iteration, docked_embeddings, objectives)
                    self._initialize_model(self.best_hyperparams)

                logging.info(f"Training model on {len(docked_peptides)} peptides")
                
                # Split data into training and validation sets if num_validate is specified
                if num_validate is not None and num_validate > 0:
                    train_embeddings, train_objectives, val_embeddings, val_objectives = self._split_train_validation(
                        docked_embeddings, objectives, num_validate
                    )
                    
                    # Train on the training set only
                    loss = self.model.fit_dict(
                        embedding_dict=train_embeddings,
                        scores_dict=train_objectives,
                        epochs=self.best_hyperparams.get("epochs", 100),
                        learning_rate=self.best_hyperparams.get("learning_rate", 0.001),
                        batch_size=self.best_hyperparams.get("batch_size", 16),
                        device=self.device,
                    )

                    # Get predictions for both training and validation sets
                    train_predictions = self.model.predict_dict(train_embeddings, device=self.device)
                    val_predictions = self.model.predict_dict(val_embeddings, device=self.device)
                    
                    # Compute metrics for both sets
                    train_metrics = self._compute_model_metrics(
                        predictions_dict=train_predictions,
                        objectives=train_objectives,
                    )
                    
                    val_metrics = self._compute_model_metrics(
                        predictions_dict=val_predictions,
                        objectives=val_objectives,
                    )
                    
                    # Combine metrics
                    metrics = {
                        "train_r2": train_metrics["r2"], 
                        "train_mae": train_metrics["mae"],
                        "val_r2": val_metrics["r2"],
                        "val_mae": val_metrics["mae"]
                    }
                    
                    r2 = train_metrics["r2"]
                    val_r2 = val_metrics["r2"]
                    
                    logging.info(f"Iteration {iteration} - Loss: {loss:.4f}, Train R²: {r2:.4f}, Val R²: {val_r2:.4f}, Train N: {len(train_embeddings)}, Val N: {len(val_embeddings)}")
                else:
                    # Train on all data when no validation is requested
                    loss = self.model.fit_dict(
                        embedding_dict=docked_embeddings,
                        scores_dict=objectives,
                        epochs=self.best_hyperparams.get("epochs", 100),
                        learning_rate=self.best_hyperparams.get("learning_rate", 0.001),
                        batch_size=self.best_hyperparams.get("batch_size", 16),
                        device=self.device,
                    )

                    predictions_dict = self.model.predict_dict(docked_embeddings, device=self.device)
                    
                    metrics = self._compute_model_metrics(
                        predictions_dict=predictions_dict,
                        objectives=objectives,
                    )

                    r2 = metrics["r2"]

                    logging.info(f"Iteration {iteration} - Loss: {loss:.4f}, R²: {r2:.4f}, Total N: {len(docked_embeddings)})")

                # Log the loss and metrics
                self.logger.log_model_metrics(loss, iteration, metrics)

                # 2.2) Predict for *the not-yet-docked* peptides
                candidate_embeddings = {p: self.embeddings[p] for p in not_docked_peptides}
                predictions = self.model.predict_dict(candidate_embeddings, device=self.device)
                # predictions is {peptide: (mean, std)}

                # Log predictions
                self.logger.log_predictions(predictions, iteration)

                # 2.3) Compute acquisition
                acquisition_values = self.acquisition_function_obj.compute_acquisition(
                    predictions=predictions,
                    acquisition_function=acquisition,
                )
                # acquisition_values is {peptide: acquisition_value}

                # Log acquisition
                self.logger.log_acquisition(
                    acquisition_values=acquisition_values,
                    acquisition_name=acquisition,
                    iteration=iteration,
                )

                # 2.4) Select the next set of peptides to dock
                next_peptides = self.selector.select_next_peptides(
                    peptides=not_docked_peptides,
                    embeddings=candidate_embeddings,
                    acquisition_values=acquisition_values,
                    n_select=batch_size,
                )
                logging.info(
                    f"Selected {next_peptides} peptides for docking in iteration {iteration}"
                )

                # Dock them
                docked_dirs = self.docker.dock_peptides(next_peptides)
                docked_peptides.update(next_peptides)
                not_docked_peptides.difference_update(next_peptides)

                # 2.5) Score them
                new_scores = self._score_batch(docked_dirs=docked_dirs)
                scores.update(new_scores)

                # Log new scores
                self.logger.log_scores(new_scores, iteration=iteration, acquisition_name=acquisition)
                
                # Print top performers
                self._print_top_performers(
                    objectives=objectives,
                    scores=scores,
                )
                
        # Loop is over
        final_objectives = self.scores_to_objective.create_objective(
            scores, self.objective_function, **self.objective_function_kwargs
        )

        # Log any final new objectives
        final_new_objective_peptides = set(final_objectives.keys()) - all_logged_objectives
        final_new_objectives = {peptide: final_objectives[peptide] for peptide in final_new_objective_peptides}

        if final_new_objectives:
            # Use the last iteration number + 1, or create a "final" entry
            last_iteration = sum(phase["iterations"] for phase in schedule)
            self.logger.log_objectives(final_new_objectives, iteration=last_iteration, acquisition_name=acquisition)

        self._save_model(self.log_dir + "/model.pth")

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

    def _generate_embeddings(self, peptides):
        # Create embeddings for all peptides
        if self.surrogate_model_kwargs["network_type"] in ["bilstm", "bigru"]:
            average = False
        elif self.surrogate_model_kwargs["network_type"] == "mlp":
            average = True

        embeddings_path = self.embedding_kwargs.get("embeddings_path")
        embedding_function = self.embedding_kwargs.get("embedding_function")

        if embeddings_path:
            embeddings = self.embedder.load_embeddings(embeddings_path)
        else:
            if embedding_function == "esm":
                embeddings = self.embedder.embed_esm(
                    peptides,
                    average=average,
                    model_path=self.embedding_kwargs.get("model_path", None),
                )
            elif embedding_function == "aaindex":
                embeddings = self.embedder.embed_aaindex(peptides, average=average)
            else:
                raise ValueError(
                    f"Invalid or missing embedding function: {embedding_function}"
                )
            

        if self.embedding_kwargs.get("scale", False):
            embeddings = self.embedder.scale_embeddings(embeddings)

        if self.embedding_kwargs.get("reduce_embeddings", False):
            if self.embedding_kwargs["reduce_method"] == "vae":
                embeddings = self.embedder.reduce_embeddings_autoencoder(
                    embeddings,
                    hidden_dim=self.embedding_kwargs.get("hidden_dim", 256),
                    latent_dim=self.embedding_kwargs.get("feature_dim", 128),
                )
            elif self.embedding_kwargs["reduce_method"] == "pca":
                embeddings = self.embedder.reduce_embeddings_pca(
                    embeddings,
                    n_components=self.embedding_kwargs.get("feature_dim", 128),
                )
        if not embeddings_path:
            # Save embeddings
            self.embedder.save_embeddings(
                embeddings, self.embeddings_save_path
            )

        # **Check final shapes**:
        self._check_embedding_shapes(embeddings)

        logging.info("Embeddings generated.")
        return embeddings

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

    def _optimize_hyperparameters(self, iteration:int, embeddings: dict, scores: dict):
        """
        Use the Hyperparameter Tuner to optimize all params.
        Sets self.best_hyperparams to the best found hyperparameters.
        Stores the Optuna study for future warm-starting.
        """
        # Get hyperparameter tuning configuration
        hpo_config = self.hpo_kwargs.copy() if self.hpo_kwargs else {}

        # Default hyperparameter tuning configurations
        n_trials = hpo_config.get("n_trials", 20)
        n_splits = hpo_config.get("n_splits", 3)
        random_state = hpo_config.get("random_state", 42)

        # Log that we're starting hyperparameter optimization
        logging.info(
            f"Starting hyperparameter optimization for {self.surrogate_model_kwargs['network_type']} {self.surrogate_model_kwargs['model_type']} model..."
        )
        
        # If we have a previous study, let's log that we're using it
        if self.previous_study is not None:
            logging.info(f"Using previous study with {len(self.previous_study.trials)} trials")

        # Run hyperparameter tuning with warm-starting from previous study
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
            network_type=self.surrogate_model_kwargs["network_type"]
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

    def _check_binding_site_residue_indices(
        self, binding_site_residue_indices, target_structure_path, assume_zero_indexed=None
    ):
        """
        Checks if starting index is 0.

        If not, asks the user if the binding site residues are expected to be 0-indexed.
        Corrects for the starting index if wanted.

        Return a visualization of the residues that are selected as binding site residues.
        """
        starting_index = check_starting_index_in_pdb(target_structure_path)
        protein_sequence = extract_sequence_from_pdb(target_structure_path)
        if binding_site_residue_indices is None:
            return None

        if starting_index != 0:
            if assume_zero_indexed is None:
                print(
                    f"\n\nStarting index is {starting_index}. Are the provided binding site residues 0-indexed?"
                )
                answer = input("y/n: ")
            else:
                if assume_zero_indexed is True:
                    print(
                        f"\n\nStarting index is {starting_index}. Assuming binding site residues are 0-indexed."
                    )
                    answer = "y"
                else:
                    print(
                        f"\n\nStarting index is {starting_index}. Assuming binding site residues are 1-indexed."
                    )
                    answer = "n"
            if answer == "y":
                binding_site_residue_indices = [
                    residue - starting_index for residue in binding_site_residue_indices
                ]

        print("\nBinding Site Residues Visualization:")
        print("=" * 60)
        print(f"Full sequence length: {len(protein_sequence)}")
        print(f"Selected binding site residues: {binding_site_residue_indices}")
        print("-" * 60)

        binding_site_residue_indices = sorted(binding_site_residue_indices)
        context_size = 5

        for residue_idx in binding_site_residue_indices:
            if residue_idx < 0 or residue_idx >= len(protein_sequence):
                print(f"Warning: Residue index {residue_idx} out of range")
                continue

            # Calculate start and end positions for context
            start = max(0, residue_idx - context_size)
            end = min(len(protein_sequence), residue_idx + context_size + 1)

            positions = list(range(start + starting_index, end + starting_index))
            vis_seq = list(protein_sequence[start:end])

            # Mark the selected residue
            rel_pos = residue_idx - start
            if 0 <= rel_pos < len(vis_seq):
                vis_seq[rel_pos] = f"[{vis_seq[rel_pos]}]"

            print(
                f"Residue {residue_idx + starting_index} ({protein_sequence[residue_idx]}):"
            )
            print("Position:" + " ".join(f"{pos:3d}" for pos in positions))
            print("Sequence: " + " ".join(f"{aa:3s}" for aa in vis_seq))
            print("-" * 60)

        print("=" * 60)
        print(f"The internally stored binding site residues are: {binding_site_residue_indices}")
        return binding_site_residue_indices

    def _get_default_surrogate_model_kwargs(self) -> Dict[str, Any]:
        """Return the default surrogate model configuration."""
        return {
            "network_type": "mlp",  # mlp, bilstm, bigru
            "model_type": "mc_dropout",  # nn_ensemble, mc_dropout, deep_evidential, mve
            "uncertainty_param": 0.1,  # For mc_dropout, this is dropout rate; for others, it's regularization strength
        }

    def _validate_surrogate_model_kwargs(self):
        """Validate the surrogate model configuration."""
        valid_network_types = ["mlp", "bilstm", "bigru"]
        valid_model_types = ["nn_ensemble", "mc_dropout", "deep_evidential", "mve"]

        network_type = self.surrogate_model_kwargs.get("network_type")
        if network_type not in valid_network_types:
            raise ValueError(
                f"Invalid network type: {network_type}. "
                f"Must be one of: {', '.join(valid_network_types)}"
            )

        model_type = self.surrogate_model_kwargs.get("model_type")
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model type: {model_type}. "
                f"Must be one of: {', '.join(valid_model_types)}"
            )

    def _setup(self):
        """
        Check if colabfold -help is callable
        Check if output dir exists
        """
        try:
            subprocess.Popen(
                ["colabfold_batch", "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        except:
            raise ValueError(
                "colabfold is not callable through the command: colabfold_batch --help"
            )

        try:
            pyrosetta.init("-mute all")
        except:
            raise ValueError("Cannot initialize pyrosetta")

    def _validate_schedule(self, schedule: List[Dict[str, Any]]):
        """
        Validate the optimization schedule.

        Args:
            schedule: List of phase configurations

        Raises:
            ValueError: If the schedule is invalid
        """
        if not schedule:
            raise ValueError("Schedule cannot be empty")

        for idx, phase in enumerate(schedule):
            if not isinstance(phase, dict):
                raise ValueError(f"Phase {idx} must be a dictionary")

            if "acquisition" not in phase:
                raise ValueError(f"Phase {idx} missing required key 'acquisition'")

            if phase["acquisition"] not in self.available_acquistion_functions:
                raise ValueError(
                    f"Invalid acquisition function in phase {idx}: {phase['acquisition']}. "
                    f"Must be one of: {', '.join(self.available_acquistion_functions)}"
                )

            if "iterations" not in phase:
                raise ValueError(f"Phase {idx} missing required key 'iterations'")

            if not isinstance(phase["iterations"], int) or phase["iterations"] <= 0:
                raise ValueError(f"Phase {idx}: iterations must be a positive integer")

    def _save_model(self, save_path: str):
        """
        Save the current model to a file with metadata.
        
        Args:
            save_path: Path to save the model
        """
        # Create a comprehensive save dictionary
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_type': self.surrogate_model_kwargs['model_type'],
                'network_type': self.surrogate_model_kwargs['network_type'],
                'input_dim': self.surrogate_model_kwargs['input_dim'],
                'hyperparameters': self.best_hyperparams,
            },
            'model_class': self.model.__class__.__name__,
            'embedding_kwargs': self.embedding_kwargs,
            'surrogate_model_kwargs': self.surrogate_model_kwargs,
        }
        
        torch.save(save_dict, save_path)
        logging.info(f"Model saved to {save_path}")
        logging.info(f"Model type: {self.surrogate_model_kwargs['model_type']}")
        logging.info(f"Network architecture: {self.surrogate_model_kwargs['network_type']}")
        logging.info(f"Model class: {self.model.__class__.__name__}")

    
    def _print_top_performers(self, objectives, scores, top_n=5):

        sorted_peptides = sorted(
            objectives.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        logging.info(f"Top {top_n} peptides:")
        logging.info(f"{'Peptide':<20} | {'Objective':<10} ")
        logging.info("-" * 60)
        
        for peptide, obj_value in sorted_peptides:
        
            logging.info(f"{peptide:<20} | {obj_value:<10.4f} ")

    def _compute_model_metrics(self, predictions_dict : dict, objectives : dict):
        peptides = list(predictions_dict.keys())
        actual = np.array([objectives[p]   for p in peptides])
        predicted = np.array([predictions_dict[p][0] for p in peptides])
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        return {"r2": r2, "mae": mae}
    
    def _split_train_validation(self, docked_embeddings: dict, objectives: dict, num_validate: int):
        """
        Split the available data into training and validation sets.

        """
        # Get all peptides that have been docked and scored
        peptides = list(objectives.keys())
        
        # Check if we have enough data for validation
        if len(peptides) <= num_validate:
            logging.warning(f"Not enough data for validation. Using {len(peptides) // 2} samples for validation instead of requested {num_validate}.")
            num_validate = max(1, len(peptides) // 2)
        
        # Randomly select validation peptides
        np.random.seed(42)  # For reproducibility
        val_indices = np.random.choice(len(peptides), num_validate, replace=False)
        val_peptides = [peptides[i] for i in val_indices]
        train_peptides = [p for p in peptides if p not in val_peptides]
        
        # Create training and validation dictionaries
        train_embeddings = {p: docked_embeddings[p] for p in train_peptides}
        train_objectives = {p: objectives[p] for p in train_peptides}
        val_embeddings = {p: docked_embeddings[p] for p in val_peptides}
        val_objectives = {p: objectives[p] for p in val_peptides}
        
        logging.info(f"Split data into {len(train_peptides)} training and {len(val_peptides)} validation samples")
        
        return train_embeddings, train_objectives, val_embeddings, val_objectives