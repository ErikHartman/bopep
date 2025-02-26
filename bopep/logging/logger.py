import os
import csv
from datetime import datetime


class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self._scores_header_written = False
        self._model_losses_header_written = False
        self._predictions_header_written = False
        self._acquisition_header_written = False

        self._scores_file = os.path.join(self.log_dir, "scores.csv")
        self._model_losses_file = os.path.join(self.log_dir, "model_losses.csv")
        self._predictions_file = os.path.join(self.log_dir, "predictions.csv")
        self._acquisition_file = os.path.join(self.log_dir, "acquisition.csv")

    def log_scores(self, scores: dict, iteration: int):
        """
        scores: {peptide: float}
        """
        timestamp = datetime.now().isoformat()
        with open(self._scores_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._scores_header_written:
                writer.writerow(["timestamp", "iteration", "peptide", "score"])
                self._scores_header_written = True
            for peptide, score in scores.items():
                writer.writerow([timestamp, iteration, peptide, score])

    def log_model_loss(self, loss: float, iteration: int):
        """
        losses: a float
        """
        timestamp = datetime.now().isoformat()
        with open(self._model_losses_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._model_losses_header_written:
                writer.writerow(["timestamp", "iteration", "epoch_in_fit", "loss"])
                self._model_losses_header_written = True

            writer.writerow([timestamp, iteration, loss])

    def log_predictions(self, predictions: dict, iteration: int):
        """
        predictions: {peptide: (mean, std)}
        """
        timestamp = datetime.now().isoformat()
        with open(self._predictions_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._predictions_header_written:
                writer.writerow(["timestamp", "iteration", "peptide", "mean", "std"])
                self._predictions_header_written = True
            for peptide, (mean_val, std_val) in predictions.items():
                writer.writerow([timestamp, iteration, peptide, mean_val, std_val])

    def log_acquisition(
        self, acquisition_values: dict, acquisition_name: str, iteration: int
    ):
        """
        acquisition_values: {peptide: float}
        """
        timestamp = datetime.now().isoformat()
        peptides = sorted(acquisition_values.keys())
        with open(self._acquisition_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._acquisition_header_written:
                writer.writerow(
                    [
                        "timestamp",
                        "iteration",
                        "peptide",
                        "acquisition_name",
                        "acquisition_value",
                    ]
                )
                self._acquisition_header_written = True
            for peptide in peptides:
                writer.writerow(
                    [
                        timestamp,
                        iteration,
                        peptide,
                        acquisition_name,
                        acquisition_values[peptide],
                    ]
                )
