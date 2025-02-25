from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model.nn_ensemble import NNEnsemble
from bopep.embedding.embedder import Embedder
from bopep.logging.logger import Logger
from bopep.bayesian_optimization.acquisition_functions import AcquisitionFunction
from bopep.bayesian_optimization.selection import PeptideSelector

import time

class BoPep:

    def __init__(
        self,
        model_type="nn_ensemble",
        scoring_function="bopep",
        embedding_function="esm",
        docker_kwargs=dict(),
    ):
        self.embedder = Embedder()
        self.scorer = Scorer()
        self.docker = Docker(docker_kwargs)
        self.logger = Logger()
        self.acquisition_function_obj = AcquisitionFunction()
        self.selector = PeptideSelector()

        if model_type == "nn_ensemble":
            self.model = NNEnsemble()
        else:
            raise ValueError(
                f"Invalid model type: {model_type}. Only nn_ensemble is supported."
            )

        self.scoring_function = scoring_function
        self.embedding_function = embedding_function

        self.available_acquistion_functions = [
            "expected_improvement",
            "standard_deviation",
        ]

    def optimize(
        self,
        peptides: list,
        target_structure_path: str,
        num_initial: int = 10,
        batch_size: int = 4,
        schedule: list[dict] = list(
            dict(acquisition="standard_deviation", iterations=10),
            dict(acquisition="expected_improvement", iterations=10)
        ),
    ):
        """
        Runs Bayesian optimization.
        """
        docked_peptides = set()
        not_docked_peptides = set(peptides)
        scores = dict() # dict of peptide : score
        target_structure_path = target_structure_path

        # Create embeddings
        embeddings = self.embedder.embed_esm(peptides) # dict of peptide : embedding

        # Initialize
        initial_peptides = self.selector.select_initial_peptides(
            embeddings=embeddings, num_initial=num_initial, random_state=42
        )
        self.docker.dock_peptides(initial_peptides)
        docked_peptides += set(initial_peptides)
        not_docked_peptides -= set(initial_peptides)

        # Score
        new_scores = self.scorer.score(self.scoring_function)
        scores.update(new_scores)

        # Log
        self.logger.log_scores(scores)

        # Run Bayesian Optimization loop
        for phase in schedule:
            acquisition = phase["acquisition"]
            iterations = phase["iterations"]

            for iteration in range(iterations):
                # Train surrogate model
                self.model.fit()

                # Predict
                means, stds = self.model.predict(embeddings)

                # Log predictions
                self.logger.log_predictions(embeddings, means, stds)

                # Compute acquistion function
                acquisition_values = self.acquisition_function_obj.compute_acquisition(
                    acquisition=acquisition, means=means, stds=stds
                )

                # Log acquisition
                self.logger.log_acquisition(embeddings, acquisition_values, acquisition)

                # Select next peptides
                next_peptides = self.selector.select_next_peptides(
                    peptides=not_docked_peptides,
                    acquisition_values=acquisition_values,
                    n_select=batch_size,
                )

                # Dock
                self.docker.dock_peptides(next_peptides)
                docked_peptides += set(next_peptides)
                not_docked_peptides -= set(next_peptides)

                # Score
                new_scores = self.scorer.score(self.scoring_function)
                scores.update(new_scores)
