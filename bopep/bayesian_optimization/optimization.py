import subprocess
from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import NeuralNetworkEnsemble, BayesianNeuralNetwork
from bopep.embedding.embedder import Embedder
from bopep.logging.logger import Logger
from bopep.bayesian_optimization.acquisition_functions import AcquisitionFunction
from bopep.bayesian_optimization.selection import PeptideSelector
from bopep.scoring.scores_to_objective import ScoresToObjective
import pyrosetta


class BoPep:
    def __init__(
        self,
        surrogate_model_kwargs: dict = {
            "model_type": "nn_ensemble",
            "hidden_dims": [64, 64],
            "n_networks": 5,
        },
        objective_weights: dict = {"rosetta_score": 1},
        embedding_function="esm",
        docker_kwargs=dict(),
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
        self.embedding_function = embedding_function
        self.surrogate_model_kwargs = surrogate_model_kwargs

        # These are used by your code to verify or choose an acquisition
        self.available_acquistion_functions = [
            "expected_improvement",
            "standard_deviation",
            "upper_confidence_bound",
            "probability_of_improvement",
            "mean",
        ]

    def optimize(
        self,
        peptides: list,
        target_structure_path: str,
        num_initial: int = 10,
        batch_size: int = 4,
        schedule: list[dict] = [
            dict(acquisition="standard_deviation", iterations=10),
            dict(acquisition="expected_improvement", iterations=10),
        ],
    ):
        """
        Runs Bayesian optimization, separated into phases given by 'schedule'.
        Each phase specifies an acquisition function and how many iterations to run.
        """
        self.docker.set_target_structure(target_structure_path)

        docked_peptides = set()
        not_docked_peptides = set(peptides)

        # Keep a dict of {peptide: score}, to update after each docking/score
        scores = dict()

        # Create embeddings for all peptides
        embeddings = self.embedder.embed_esm(peptides)  # {peptide: np.ndarray}
        embeddings = self.embedder.reduce_embeddings_pca(
            embeddings, explained_variance_ratio=0.95
        )

        # Create surrogate model
        self.surrogate_model_kwargs["input_dim"] = len(embeddings[list(embeddings.keys())][0])
        self._create_model()

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
        # There should be a function that adds a score to scores called `MODEL_SCORE` or similar. This is the score that the model will use to optimize.

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
            )
            new_scores.update(colab_dir_scores)

        return new_scores

    def _create_model(self):

        if self.surrogate_model_kwargs["model_type"] == "nn_ensemble":
            self.model = NeuralNetworkEnsemble(
                input_dim=self.surrogate_model_kwargs.get("input_dim"),
                hidden_dims=self.surrogate_model_kwargs.get("hidden_dims"),
                output_dim=1,
                n_networks=self.surrogate_model_kwargs.get("n_networks"),
            )
        elif self.surrogate_model_kwargs["model_type"] == "bnn":
            self.model = BayesianNeuralNetwork(
                input_dim=self.surrogate_model_kwargs.get("input_dim"),
                hidden_dims=self.surrogate_model_kwargs.get("hidden_dims"),
                output_dim=1,
                dropout_rate=self.surrogate_model_kwargs.get("dropout_rate"),
                mc_samples=self.surrogate_model_kwargs.get("mc_samples"),
            )

        else:
            raise ValueError(
                f"Invalid model type: {self.surrogate_model_kwargs}. Only nn_ensemble is supported."
            )

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
