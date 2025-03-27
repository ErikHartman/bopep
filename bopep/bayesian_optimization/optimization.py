import subprocess
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BoPep:
    def __init__(
        self,
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        objective_function: Optional[Callable] = None,  # defaults to bopep_objective
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        hpo_kwargs: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
    ):
        """
        Initialize the BoPep optimizer with various configuration options.

        Args:
            surrogate_model_kwargs: Configuration for the surrogate model including:
                - network_type: 'mlp', 'bilstm', or 'bigru'
                - model_type: 'nn_ensemble', 'mc_dropout', 'deep_evidential', or 'mve'
            objective_function: Custom objective function (defaults to bopep_objective if None)
            objective_function_kwargs: Parameters for the objective function
            embedding_kwargs: Configuration for peptide embedding generation
            docker_kwargs: Configuration for the Docker component
            hpo_kwargs: Configuration for hyperparameter optimization
            log_dir: Directory for logging output
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

        # Initialize components
        self.embedder = Embedder()
        self.scorer = Scorer()
        self.docker = Docker(self.docker_kwargs)
        self.logger = Logger(log_dir=log_dir)
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = PeptideSelector()
        self.scores_to_objective = ScoresToObjective()
        
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

    def optimize(
        self,
        peptides: List[str],
        target_structure_path: str,
        num_initial: int = 10,
        batch_size: int = 4,
        binding_site_residue_indices: Optional[List[int]] = None,
        schedule: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[Dict[str, Any]] = None,
        initial_peptides: Optional[List[str]] = None,
    ):
        """
        Runs Bayesian optimization on peptide sequences.

        Args:
            peptides: List of peptide sequences to optimize
            target_structure_path: Path to the target protein structure file
            num_initial: Number of initial peptides to sample before optimization
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
            binding_site_residue_indices, target_structure_path
        )  # Checks if pdb starting index is 0, adjusts if needed.

        docked_peptides = set()
        not_docked_peptides = set(peptides)

        # Keep a dict of {peptide: score}, to update after each docking/score
        scores = dict()

        # Generate embeddings for all peptides if not provided
        if embeddings is None:
            self.embeddings = self._generate_embeddings(peptides)
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
        self.logger.log_scores(initial_scores, iteration=0)

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

            for iteration in range(iterations):

                # 2.0) Turn scores into a scalarized score dict of peptide: score
                objectives = self.scores_to_objective.create_objective(
                    scores, self.objective_function, **self.objective_function_kwargs
                )

                # 2.1) Train the model on *only the peptides we have scores for*
                train_embeddings = {p: self.embeddings[p] for p in docked_peptides}

                # Possibly re-run hyperparameter optimization every N steps
                if iteration == 0 or (
                    iteration % self.hpo_kwargs.get("hpo_interval", 10) == 0
                ):
                    self._optimize_hyperparameters(train_embeddings, objectives)
                    self._initialize_model(self.best_hyperparams)
                    self.model.to(device)

                logging.info(f"Training model on {len(docked_peptides)} peptides")
                loss = self.model.fit_dict(
                    embedding_dict=train_embeddings,
                    scores_dict=objectives,
                    epochs=self.best_hyperparams.get("epochs", 100),
                    learning_rate=self.best_hyperparams.get("learning_rate", 0.001),
                    batch_size=self.best_hyperparams.get("batch_size", 16),
                    device=device,
                )
                
                # Calculate R² on training data
                predictions_dict = self.model.predict_dict(train_embeddings, device=device)
                actual_values = np.array(list(objectives.values()))
                predicted_values = np.array([pred[0] for pred in predictions_dict.values()])  # Using mean predictions
                
                # Calculate R²
                ss_total = np.sum((actual_values - np.mean(actual_values))**2)
                ss_residual = np.sum((actual_values - predicted_values)**2)
                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0

                # Log the loss and R²
                self.logger.log_model_loss(loss, iteration, r2)

                # 2.2) Predict for *the not-yet-docked* peptides
                candidate_embeddings = {p: self.embeddings[p] for p in not_docked_peptides}
                predictions = self.model.predict_dict(candidate_embeddings, device=device)
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
                self.logger.log_scores(new_scores, iteration=iteration)

    def _score_batch(self, docked_dirs: list):
        """
        Scores a batch (list of dirs) and outputs the score dict
        """
        new_scores = {}
        scores_to_include = self.scoring_kwargs.get("scores_to_include", [])
        for colab_dir in docked_dirs:
            colab_dir_scores = self.scorer.score(
                scores_to_include=scores_to_include,
                colab_dir=colab_dir,
                binding_site_residue_indices=self.binding_site_residue_indices,
            )
            new_scores.update(colab_dir_scores)

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

    def _optimize_hyperparameters(self, embeddings: dict, scores: dict):
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
        rmse_weight = hpo_config.get("rmse_weight", 1.0)
        msce_weight = hpo_config.get("msce_weight", 1.0)
        coverage_weight = hpo_config.get("coverage_weight", 1.0)

        # Log that we're starting hyperparameter optimization
        logging.info(
            f"Starting hyperparameter optimization for {self.surrogate_model_kwargs['model_type']} model..."
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
            rmse_weight=rmse_weight,
            msce_weight=msce_weight,
            coverage_weight=coverage_weight,
            previous_study=self.previous_study,
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

    def _check_binding_site_residue_indices(
        self, binding_site_residue_indices, target_structure_path
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
            print(
                f"\n\nStarting index is {starting_index}. Are the provided binding site residues 0-indexed?"
            )
            answer = input("y/n: ")
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
            "n_networks": 5,
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
