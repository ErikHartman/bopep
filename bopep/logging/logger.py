import os
import csv
from datetime import datetime

class Logger:
    """
    Simple CSV-based logger that writes scores, predictions, and
    acquisition values to separate CSV files in a specified log directory.
    Each method appends to a file.
    """
    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: Directory where CSV log files will be written.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Flags to track if we've written a header to each file yet
        self._scores_header_written = False
        self._predictions_header_written = False
        self._acquisition_header_written = False

        # Define file paths
        self._scores_file = os.path.join(self.log_dir, "scores.csv")
        self._predictions_file = os.path.join(self.log_dir, "predictions.csv")
        self._acquisition_file = os.path.join(self.log_dir, "acquisition.csv")

    def log_scores(self, scores: dict):
        """
        Logs peptide scores to scores.csv.
        Each key in `scores` is a peptide, and each value is the score.
        
        Args:
            scores: dict of {peptide: score}.
        """
        timestamp = datetime.now().isoformat()

        with open(self._scores_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._scores_header_written:
                writer.writerow(["timestamp", "peptide", "score"])
                self._scores_header_written = True

            for peptide, score in scores.items():
                writer.writerow([timestamp, peptide, score])

    def log_predictions(self, embeddings: dict, means, stds):
        """
        Logs model predictions (mean, std) for each peptide to predictions.csv.
        
        Args:
            embeddings: dict {peptide: embedding}, used here to get a consistent peptide order.
            means: array-like of same length as the number of peptides in `embeddings`.
            stds: array-like of same length as the number of peptides in `embeddings`.
        """
        timestamp = datetime.now().isoformat()
        peptides = embeddings.keys()

        with open(self._predictions_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._predictions_header_written:
                writer.writerow(["timestamp", "peptide", "mean", "std"])
                self._predictions_header_written = True

            for i, pep in enumerate(peptides):
                mean_val = float(means[i])
                std_val = float(stds[i])
                writer.writerow([timestamp, pep, mean_val, std_val])

    def log_acquisition(self, embeddings: dict, acquisition_values, acquisition_name: str):
        """
        Logs acquisition function values for each peptide to acquisition.csv.
        
        Args:
            embeddings: dict {peptide: embedding}, used for consistent ordering.
            acquisition_values: array-like, same length as `embeddings`.
            acquisition_name: e.g., "expected_improvement" or "standard_deviation".
        """
        timestamp = datetime.now().isoformat()
        peptides = embeddings.keys()

        with open(self._acquisition_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._acquisition_header_written:
                writer.writerow(["timestamp", "peptide", "acquisition_name", "acquisition_value"])
                self._acquisition_header_written = True

            for i, pep in enumerate(peptides):
                writer.writerow([
                    timestamp, 
                    pep, 
                    acquisition_name, 
                    float(acquisition_values[i])
                ])
