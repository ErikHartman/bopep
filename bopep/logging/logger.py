import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List


class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self._scores_header_written = False
        self._model_losses_header_written = False
        self._predictions_header_written = False
        self._acquisition_header_written = False
        self._objectives_header_written = False

        # Define base filenames
        scores_base = "scores.csv"
        model_losses_base = "model_losses.csv"
        predictions_base = "predictions.csv"
        acquisition_base = "acquisition.csv"
        objectives_base = "objectives.csv"
        hyperparameters_base = "hyperparameters.csv"
        
        # Check if any of the files already exist
        base_files = [scores_base, model_losses_base, predictions_base, acquisition_base, objectives_base, hyperparameters_base]
        file_exists = any(os.path.exists(os.path.join(self.log_dir, f)) for f in base_files)
        
        # If files exist, ask user about overwriting
        overwrite = True
        if file_exists:
            response = input("Log files already exist. Do you want to overwrite them? (y/n): ").lower()
            overwrite = response == 'y' or response == 'yes'
        
        # Either use base filenames or find alternative names
        if overwrite:
            self._scores_file = os.path.join(self.log_dir, scores_base)
            self._model_losses_file = os.path.join(self.log_dir, model_losses_base)
            self._predictions_file = os.path.join(self.log_dir, predictions_base)
            self._acquisition_file = os.path.join(self.log_dir, acquisition_base)
            self._objectives_file = os.path.join(self.log_dir, objectives_base)
            self._hyperparameter_file = os.path.join(self.log_dir, hyperparameters_base)
            
            # Delete existing files if overwriting
            for file_path in [self._scores_file, self._model_losses_file, 
                             self._predictions_file, self._acquisition_file, self._objectives_file, self._hyperparameter_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            self._scores_file = self._get_unique_filename(self.log_dir, scores_base)
            self._model_losses_file = self._get_unique_filename(self.log_dir, model_losses_base)
            self._predictions_file = self._get_unique_filename(self.log_dir, predictions_base)
            self._acquisition_file = self._get_unique_filename(self.log_dir, acquisition_base)
            self._objectives_file = self._get_unique_filename(self.log_dir, objectives_base)
            self._hyperparameter_file = self._get_unique_filename(self.log_dir, hyperparameters_base)
    
    def _get_unique_filename(self, directory, base_filename):
        """Generate a unique filename by adding an incremental number if file exists."""
        name, ext = os.path.splitext(base_filename)
        counter = 1
        new_filename = base_filename
        
        while os.path.exists(os.path.join(directory, new_filename)):
            new_filename = f"{name}_{counter}{ext}"
            counter += 1
            
        return os.path.join(directory, new_filename)

    def log_scores(self, scores: dict, iteration: int, acquisition_name: str = "unknown"):
        """
        Log scores for peptides where each peptide has multiple score types.
        """
        timestamp = datetime.now().isoformat()
        
        score_types = set()
        for peptide, peptide_scores in scores.items():
            score_types.update(peptide_scores.keys())
        score_types = sorted(score_types)
        
        with open(self._scores_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._scores_header_written:
                header = ["timestamp", "iteration", "peptide", "acquisition_phase_when_added"] + score_types
                writer.writerow(header)
                self._scores_header_written = True
            
            for peptide, peptide_scores in scores.items():
                row = [timestamp, iteration, peptide, acquisition_name]
                for score_type in score_types:
                    row.append(peptide_scores.get(score_type, None))
                writer.writerow(row)
                
    def log_objectives(self, objectives: dict, iteration: int, acquisition_name: str = "unknown"):
        """
        Log objective values for peptides along with the acquisition phase they were added in.
        
        Args:
            objectives: Dictionary mapping peptides to their objective values
            iteration: Current iteration number
            acquisition_name: Name of the acquisition function used when selecting this peptide
        """
        timestamp = datetime.now().isoformat()
        
        with open(self._objectives_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._objectives_header_written:
                writer.writerow(["timestamp", "iteration", "peptide", "objective_value", "acquisition_phase_when_added"])
                self._objectives_header_written = True
            
            for peptide, objective_value in objectives.items():
                writer.writerow([timestamp, iteration, peptide, objective_value, acquisition_name])

    def log_model_loss(self, loss: float, iteration: int, r2: Optional[float] = None):
        """
        losses: a float
        r2: coefficient of determination (optional)
        """
        timestamp = datetime.now().isoformat()
        with open(self._model_losses_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not self._model_losses_header_written:
                writer.writerow(["timestamp", "iteration", "loss", "r2"])
                self._model_losses_header_written = True

            writer.writerow([timestamp, iteration, loss, r2])

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
    def log_hyperparameters(self, iteration: int, hyperparams: dict, model_type: str, network_type: str):
        """
        Log the hyperparameters of the surrogate model.
        """
        
        hyper_file = os.path.join(self.log_dir, "hyperparameters.csv")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        flat_params = {}
        for key, value in hyperparams.items():
            if isinstance(value, (list, tuple)):
                flat_params[key] = str(value)
            else:
                flat_params[key] = value
        
        # Write to CSV
        header_needed = not os.path.exists(hyper_file)
        with open(hyper_file, "a", newline="") as f:
            fieldnames = ["timestamp", "iteration", "model_type", "network_type"]
            fieldnames.extend(sorted(flat_params.keys()))
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                writer.writeheader()
            
            # Prepare row data
            row_data = {
                "timestamp": timestamp,
                "iteration": iteration,
                "model_type": model_type,
                "network_type": network_type,
                **flat_params
            }
            writer.writerow(row_data)
        