import subprocess
from typing import List, Optional
from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import (
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    OptunaOptimizer,
)
from bopep.embedding.embedder import Embedder
from bopep.logging.logger import Logger
from bopep.bayesian_optimization.acquisition_functions import AcquisitionFunction
from bopep.bayesian_optimization.selection import PeptideSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.bayesian_optimization.utils import check_starting_index_in_pdb
from bopep.docking.utils import extract_sequence_from_pdb
import pyrosetta

from bopep.surrogate_model.base_models import BiLSTMNetwork, MLPNetwork


class BoPep:
    def __init__(
        self,
        surrogate_model_kwargs: dict = {
            "network_type": "mlp",
            "model_type": "mc_dropout",
            "n_networks": 5,
        },
        objective_weights: dict = {"rosetta_score": 1},
        embedding_kwargs: dict = dict(),
        docker_kwargs=dict(),
        hpo_kwargs=dict(),
        log_dir: str = "logs",
    ):
        self._setup()
        self.embedder = Embedder()
        self.scorer = Scorer()
        self.docker = Docker(docker_kwargs)
        self.logger = Logger(log_dir=log_dir)
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = PeptideSelector()
        self.scores_to_objective = ScoresToObjective()

        self.objective_weights = objective_weights
        self.embedding_kwargs = embedding_kwargs
        self.surrogate_model_kwargs = surrogate_model_kwargs
        self.hpo_kwargs = hpo_kwargs

        # These are used by your code to verify or choose an acquisition
        self.available_acquistion_functions = [
            "expected_improvement",
            "standard_deviation",
            "upper_confidence_bound",
            "probability_of_improvement",
            "mean",
        ]
        if surrogate_model_kwargs["network_type"] not in ["mlp", "bilstm"]:
            raise ValueError(
                f"Invalid network type: {surrogate_model_kwargs['network_type']}."
            )
        if surrogate_model_kwargs["model_type"] not in [
            "nn_ensemble",
            "mc_dropout",
            "deep_evidential_regression",
        ]:
            raise ValueError(
                f"Invalid model type: {surrogate_model_kwargs['model_type']}."
            )

    def optimize(
        self,
        peptides: List[str],
        target_structure_path: str,
        num_initial: int = 10,
        batch_size: int = 4,
        binding_site_residue_indices: Optional[List[int]] = None,
        schedule: list[dict] = [
            dict(acquisition="standard_deviation", iterations=10),
            dict(acquisition="expected_improvement", iterations=10),
        ],
        embeddings: Optional[dict] = None,
    ):
        """
        Runs Bayesian optimization, separated into phases given by 'schedule'.
        Each phase specifies an acquisition function and how many iterations to run.
        """

        for phase in schedule:
            if phase["acquisition"] not in self.available_acquistion_functions:
                raise ValueError(
                    f"Invalid acquisition function: {acquisition['acquisition']}."
                )

        self.docker.set_target_structure(target_structure_path)

        self.binding_site_residue_indices = self._check_binding_site_residue_indices(
            binding_site_residue_indices, target_structure_path
        )  # Checks if pdb starting index is 0, adjusts if needed.

        docked_peptides = set()
        not_docked_peptides = set(peptides)

        # Keep a dict of {peptide: score}, to update after each docking/score
        scores = dict()

        # Generate embeddings for all peptides
        if embeddings is None:
            embeddings = self._generate_embeddings(peptides)

        # Create surrogate model
        self.surrogate_model_kwargs["input_dim"] = len(
            embeddings[list(embeddings.keys())][0]
        )  # this might not work for bilstm

        # 1) Select initial peptides for docking
        initial_peptides = self.selector.select_initial_peptides(
            embeddings=embeddings, num_initial=num_initial, random_state=42
        )
        # Dock them
        docked_dirs = self.docker.dock_peptides(
            initial_peptides
        )  # -> list of dirs for docked peptides
        docked_peptides.update(initial_peptides)
        not_docked_peptides.difference_update(initial_peptides)

        # Score them
        initial_scores = self._score_batch(
            docked_dirs=docked_dirs,
        )  # {peptide: scores} where scores itself is a dict of {score: value}
        scores.update(initial_scores)

        # Log scores
        self.logger.log_scores(initial_scores, iteration=0)

        # 2) Main BO Loop over phases
        phase_index = 0
        for phase in schedule:
            phase_index += 1
            acquisition = phase["acquisition"]
            iterations = phase["iterations"]

            for iteration in range(iterations):

                # 2.0) Turn scores into a scalarized score dict of peptide: score
                objectives = self.scores_to_objective.create_bopep_objective(
                    scores, self.objective_weights
                )
                
                if iteration == 0 or (
                    iteration % self.hpo_kwargs.get("hpo_interval", 10) == 0
                ):
                    self._optimize_hyperparameters(
                        embeddings, objectives
                    )  # This sets self.model

                # 2.1) Train the model on *only the peptides we have scores for*
                train_embeddings = {p: embeddings[p] for p in docked_peptides}
                loss = self.model.fit_dict(
                    embedding_dict=train_embeddings,
                    scores_dict=objectives,
                    epochs=50,
                    batch_size=16,
                    learning_rate=1e-3,
                )
                # Log the loss
                self.logger.log_model_loss(loss, iteration)

                # 2.2) Predict for *the not-yet-docked* peptides
                # Because we only need acquisitions for undiscovered peptides
                candidate_embeddings = {p: embeddings[p] for p in not_docked_peptides}
                predictions = self.model.predict_dict(candidate_embeddings)
                # predictions is {peptide: (mean, std)}

                # Log predictions
                self.logger.log_predictions(predictions, iteration)

                # 2.3) Compute acquisition
                # We pass `candidate_embeddings` plus the 'predictions' dict
                # so the method can parse means & stds from predictions.
                acquisition_values = self.acquisition_function_obj.compute_acquisition(
                    embeddings=candidate_embeddings,
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

        for colab_dir in docked_dirs:
            colab_dir_scores = self.scorer.score(
                scores_to_include=list(self.objective_weights.keys()),
                colab_dir=colab_dir,
                binding_site_residue_indices=self.binding_site_residue_indices,
            )
            new_scores.update(colab_dir_scores)

        return new_scores

    def _generate_embeddings(self, peptides):
        # Create embeddings for all peptides
        if self.embedding_kwargs["embedding_function"] == "esm":
            if self.surrogate_model_kwargs["network_type"] == "bilstm":
                embeddings = self.embedder.embed_esm(
                    peptides,
                    average=False,
                    model_path=self.embedding_kwargs.get("model_path", None),
                )  # {peptide: np.ndarray}
            elif self.surrogate_model_kwargs["network_type"] == "mlp":
                embeddings = self.embedder.embed_esm(
                    peptides,
                    average=True,
                    model_path=self.embedding_kwargs.get("model_path", None),
                )
        elif self.embedding_kwargs["embedding_function"] == "aaindex":
            if self.surrogate_model_kwargs["network_type"] == "bilstm":
                embeddings = self.embedder.embed_aaindex(peptides, average=False)
            elif self.surrogate_model_kwargs["network_type"] == "mlp":
                embeddings = self.embedder.embed_aaindex(peptides, average=True)
        else:
            raise ValueError(
                f"Invalid embedding function: {self.embedding_kwargs('embedding_function')}"
            )

        embeddings = self.embedder.scale_embeddings(embeddings)

        if self.embedding_kwargs.get("reduce_embeddings", False):
            if self.surrogate_model_kwargs["network_type"] == "bilstm":
                embeddings = self.embedder.reduce_embeddings_autoencoder(
                    embeddings,
                    hidden_dim=self.embedding_kwargs("hidden_dim", 256),
                    latent_dim=self.embedding_kwargs("latent_dim", 128),
                )
            elif self.surrogate_model_kwargs["network_type"] == "mlp":
                embeddings = self.embedder.reduce_embeddings_pca(
                    embeddings,
                    explained_variance_ratio=self.embedding_kwargs.get(
                        "explained_variance_ratio", 0.95
                    ),
                )
        return embeddings

    def _optimize_hyperparameters(self, embeddings: dict, scores: dict):
        """
        Optimize hyperparameters for the surrogate model using Optuna.

        Sets self.model which is to be used for the next iterations.
        """
        # Decide which model_class to pass to OptunaOptimizer
        if self.surrogate_model_kwargs["network_type"] == "mlp":
            model_class = MLPNetwork
        elif self.surrogate_model_kwargs["network_type"] == "bilstm":
            model_class = BiLSTMNetwork
        else:
            print("HPO not implemented for this network type.")
            return

        optuna_optimizer = OptunaOptimizer(
            model_class=model_class,
            embedding_dict=embeddings,
            scores_dict=scores,
            n_trials=self.hpo_kwargs.get("n_trials", 10),
            test_size=self.hpo_kwargs.get("test_size", 0.2),
            random_state=42,
            early_stopping_rounds=5,
        )

        best_params = optuna_optimizer.optimize()

        if self.surrogate_model_kwargs["model_type"] == "nn_ensemble":
            self.model = NeuralNetworkEnsemble(
                input_dim=self.surrogate_model_kwargs["input_dim"],
                hidden_dims=best_params.get("hidden_dims"),
                n_networks=self.surrogate_model_kwargs.get("n_networks"),
                network_type=self.surrogate_model_kwargs.get("network_type"),
                lstm_layers=best_params.get("lstm_layers"),
                lstm_hidden_dim=best_params.get("lstm_hidden_dim"),
            )
        elif self.surrogate_model_kwargs["model_type"] == "mc_dropout":
            self.model = MonteCarloDropout(
                input_dim=self.surrogate_model_kwargs["input_dim"],
                hidden_dims=best_params.get("hidden_dims"),
                dropout_rate=self.surrogate_model_kwargs.get("dropout_rate"),
                mc_samples=self.surrogate_model_kwargs.get("mc_samples"),
                network_type=self.surrogate_model_kwargs.get("network_type"),
                lstm_layers=best_params.get("lstm_layers"),
                lstm_hidden_dim=best_params.get("lstm_hidden_dim"),
            )
        elif self.surrogate_model_kwargs["model_type"] == "deep_evidential_regression":
            self.model = DeepEvidentialRegression(
                input_dim=self.surrogate_model_kwargs["input_dim"],
                hidden_dims=best_params.get("hidden_dims"),
                dropout_rate=self.surrogate_model_kwargs.get("dropout_rate"),
                network_type=self.surrogate_model_kwargs.get("network_type"),
                lstm_layers=best_params.get("lstm_layers"),
                lstm_hidden_dim=best_params.get("lstm_hidden_dim"),
            )
        else:
            raise ValueError(
                f"Invalid model type: {self.surrogate_model_kwargs}."
            )

    def _check_binding_site_residue_indices(
        self, binding_site_residue_indices, target_structure_path
    ):
        """
        Checks if starting index is 0.

        If not, asks the user if the binding site residues are expected to be 0 indexed.
        Corrects for the starting index if wanted.

        Return a visualization of the residues that are selected as binding site residues.
        """
        starting_index = check_starting_index_in_pdb(target_structure_path)
        protein_sequence = extract_sequence_from_pdb(target_structure_path)
        if starting_index != 0:
            print(
                f"Starting index is {starting_index}. Are the provided binding site residues 0-indexed?"
            )
            answer = input("y/n")
            if answer == "y":
                binding_site_residue_indices = [
                    residue - starting_index for residue in binding_site_residue_indices
                ]

        # Visualize the binding site residues
        print("\nBinding Site Residues Visualization:")
        print("=" * 60)
        print(f"Full sequence length: {len(protein_sequence)}")
        print(f"Selected binding site residues: {binding_site_residue_indices}")
        print("-" * 60)

        # Sort binding site residues for sequential display
        binding_site_residue_indices = sorted(binding_site_residue_indices)

        # Show each binding site residue with context
        context_size = 5  # Show 5 residues before and after
        for residue_idx in binding_site_residue_indices:
            # Ensure residue_idx is valid
            if residue_idx < 0 or residue_idx >= len(protein_sequence):
                print(f"Warning: Residue index {residue_idx} out of range")
                continue

            # Calculate start and end positions for context
            start = max(0, residue_idx - context_size)
            end = min(len(protein_sequence), residue_idx + context_size + 1)

            # Generate residue numbers for display
            positions = list(range(start + starting_index, end + starting_index))

            # Create the visualization row
            vis_seq = list(protein_sequence[start:end])

            # Mark the selected residue
            rel_pos = residue_idx - start
            if 0 <= rel_pos < len(vis_seq):
                vis_seq[rel_pos] = f"[{vis_seq[rel_pos]}]"

            # Print the visualization
            print(
                f"Residue {residue_idx + starting_index} ({protein_sequence[residue_idx]}):"
            )
            print("Position: " + " ".join(f"{pos:3d}" for pos in positions))
            print("Sequence: " + " ".join(f"{aa:3s}" for aa in vis_seq))
            print("-" * 60)

        print("=" * 60)
        return binding_site_residue_indices

    def _setup(self):
        """
        Check if colabfold -help is callable
        Check if output dir exists
        """
        try:
            subprocess.Popen(
                "colabfold --help",
            )
        except:
            raise ValueError(
                "colabfold is not callable through the command: colabfold --help"
            )

        try:
            pyrosetta.init()
        except:
            raise ValueError("Cannot initialize pyrosetta")
