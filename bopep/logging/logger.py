import gzip
import os
import csv
from datetime import datetime
from typing import Optional, Dict, Any


class Logger:
    def __init__(
        self,
        log_dir: str = "logs",
        overwrite_logs: Optional[bool] = None,
        continue_from_checkpoint: bool = False,
    ):
        # Ensure log directory exists
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Remember base filenames for unique-name logic
        self._scores_base        = "scores.csv"
        self._model_losses_base  = "model_losses.csv"
        self._predictions_base   = "predictions.csv.gz"
        self._acquisition_base   = "acquisition.csv.gz"
        self._objectives_base    = "objectives.csv"
        self._hyperparams_base   = "hyperparameters.csv"

        # Flags to track if we've already written headers
        self._scores_header_written       = False
        self._model_losses_header_written = False
        self._predictions_header_written  = False
        self._acquisition_header_written  = False
        self._objectives_header_written   = False

        # Build full paths
        self._scores_file        = os.path.join(self.log_dir, self._scores_base)
        self._model_losses_file  = os.path.join(self.log_dir, self._model_losses_base)
        self._predictions_file   = os.path.join(self.log_dir, self._predictions_base)
        self._acquisition_file   = os.path.join(self.log_dir, self._acquisition_base)
        self._objectives_file    = os.path.join(self.log_dir, self._objectives_base)
        self._hyperparameter_file= os.path.join(self.log_dir, self._hyperparams_base)

        # Detect any existing logs
        all_paths = [
            self._scores_file,
            self._model_losses_file,
            self._predictions_file,
            self._acquisition_file,
            self._objectives_file,
            self._hyperparameter_file,
        ]
        files_exist = any(os.path.exists(p) for p in all_paths)

        if continue_from_checkpoint:
            # Pick up any existing headers
            self._scores_header_written       = self._file_has_header(self._scores_file)
            self._model_losses_header_written = self._file_has_header(self._model_losses_file)
            self._predictions_header_written  = self._file_has_header(self._predictions_file)
            self._acquisition_header_written  = self._file_has_header(self._acquisition_file)
            self._objectives_header_written   = self._file_has_header(self._objectives_file)

        elif files_exist:
            # Fresh start but logs exist: ask or obey overwrite_logs
            if overwrite_logs is None:
                resp = input(f"Log files already exist in {self.log_dir}. Overwrite? (y/n): ").lower()
                overwrite = resp in ("y","yes")
            else:
                overwrite = overwrite_logs

            if overwrite:
                for p in all_paths:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
            else:
                # Generate unique filenames for each
                self._scores_file        = self._get_unique_filename(self.log_dir, self._scores_base)
                self._model_losses_file  = self._get_unique_filename(self.log_dir, self._model_losses_base)
                self._predictions_file   = self._get_unique_filename(self.log_dir, self._predictions_base)
                self._acquisition_file   = self._get_unique_filename(self.log_dir, self._acquisition_base)
                self._objectives_file    = self._get_unique_filename(self.log_dir, self._objectives_base)
                self._hyperparameter_file= self._get_unique_filename(self.log_dir, self._hyperparams_base)


    def _open(self, path: str, mode: str = "at"):
        """Open text-mode UTF-8, newline=''—gzip if filename ends in .gz."""
        if path.endswith(".gz"):
            return gzip.open(path, mode, encoding="utf-8", newline="")
        return open(path, mode, encoding="utf-8", newline="")


    def _file_has_header(self, file_path: str) -> bool:
        """Return True if file exists and its first line is nonempty."""
        if not os.path.exists(file_path):
            return False
        try:
            with self._open(file_path, "rt") as f:
                return bool(f.readline().strip())
        except Exception:
            return False


    def _get_unique_filename(self, directory: str, base_filename: str) -> str:
        """If base_filename exists, append _1, _2, … until it's unique."""
        name, ext = os.path.splitext(base_filename)
        candidate = base_filename
        i = 1
        while os.path.exists(os.path.join(directory, candidate)):
            candidate = f"{name}_{i}{ext}"
            i += 1
        return os.path.join(directory, candidate)


    def log_scores(self, scores: Dict[str, Dict[str, Any]], iteration: int, acquisition_name: str = "unknown"):
        timestamp = datetime.now().isoformat()
        # collect all score types once
        types = sorted({t for sc in scores.values() for t in sc})
        with open(self._scores_file, "a", newline="") as f:
            wr = csv.writer(f)
            if not self._scores_header_written:
                wr.writerow(["timestamp","iteration","peptide","phase"] + types)
                self._scores_header_written = True
            for pep, sc in scores.items():
                row = [timestamp, iteration, pep, acquisition_name] + [sc.get(t) for t in types]
                wr.writerow(row)


    def log_objectives(self, objectives: Dict[str, float], iteration: int, acquisition_name: str = "unknown"):
        timestamp = datetime.now().isoformat()
        with open(self._objectives_file, "a", newline="") as f:
            wr = csv.writer(f)
            if not self._objectives_header_written:
                wr.writerow(["timestamp","iteration","peptide","objective_value","phase"])
                self._objectives_header_written = True
            for pep, val in objectives.items():
                wr.writerow([timestamp, iteration, pep, val, acquisition_name])


    def log_model_metrics(self, loss: float, iteration: int, metrics: Optional[Dict[str, float]] = None):
        timestamp = datetime.now().isoformat()
        metrics = metrics or {}
        keys = list(metrics.keys())
        with open(self._model_losses_file, "a", newline="") as f:
            wr = csv.writer(f)
            if not self._model_losses_header_written:
                wr.writerow(["timestamp","iteration","loss"] + keys)
                self._model_losses_header_written = True
            wr.writerow([timestamp, iteration, loss] + [metrics[k] for k in keys])


    def log_predictions(self, predictions: Dict[str, tuple], iteration: int):
        timestamp = datetime.now().isoformat()
        with self._open(self._predictions_file, "at") as f:
            wr = csv.writer(f)
            if not self._predictions_header_written:
                wr.writerow(["timestamp","iteration","peptide","mean","std"])
                self._predictions_header_written = True
            for pep, (m, s) in predictions.items():
                wr.writerow([timestamp, iteration, pep, m, s])


    def log_acquisition(self, acquisition_values: Dict[str, float], acquisition_name: str, iteration: int):
        timestamp = datetime.now().isoformat()
        with self._open(self._acquisition_file, "at") as f:
            wr = csv.writer(f)
            if not self._acquisition_header_written:
                wr.writerow(["timestamp","iteration","peptide","phase","acquisition_value"])
                self._acquisition_header_written = True
            for pep, val in sorted(acquisition_values.items()):
                wr.writerow([timestamp, iteration, pep, acquisition_name, val])


    def log_hyperparameters(self, iteration: int, hyperparams: Dict[str, Any], model_type: str, network_type: str):
        hyper_file = os.path.join(self.log_dir, self._hyperparams_base)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        flat = {k: (str(v) if isinstance(v, (list,tuple)) else v) for k,v in hyperparams.items()}
        cols = sorted(flat.keys())
        header_needed = not os.path.exists(hyper_file)
        with open(hyper_file, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["timestamp","iteration","model_type","network_type"]+cols)
            if header_needed:
                wr.writeheader()
            row = {"timestamp":timestamp,"iteration":iteration,
                   "model_type":model_type,"network_type":network_type, **flat}
            wr.writerow(row)
