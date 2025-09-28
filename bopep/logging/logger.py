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
        
        # Track all metric keys encountered for model losses
        self._model_losses_all_keys = set()

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


    def log_objectives(self, objectives: Dict[str, Any], iteration: int, acquisition_name: str = "unknown"):
        timestamp = datetime.now().isoformat()
        
        # Check if we have single or multi-objective case
        if not objectives:
            return
            
        sample_obj = next(iter(objectives.values()))
        is_multiobjective = isinstance(sample_obj, dict)
        
        with open(self._objectives_file, "a", newline="") as f:
            wr = csv.writer(f)
            
            if is_multiobjective:
                # Multi-objective case: collect all objective names and use as columns
                all_objective_names = sorted({name for obj_dict in objectives.values() for name in obj_dict.keys()})
                
                if not self._objectives_header_written:
                    wr.writerow(["timestamp", "iteration", "peptide", "phase"] + all_objective_names)
                    self._objectives_header_written = True
                
                for pep, obj_dict in objectives.items():
                    row = [timestamp, iteration, pep, acquisition_name] + [obj_dict.get(name) for name in all_objective_names]
                    wr.writerow(row)
            else:
                # Single objective case: original format
                if not self._objectives_header_written:
                    wr.writerow(["timestamp", "iteration", "peptide", "objective_value", "phase"])
                    self._objectives_header_written = True
                
                for pep, val in objectives.items():
                    wr.writerow([timestamp, iteration, pep, val, acquisition_name])

    def log_model_metrics(self, loss: float, iteration: int, metrics: Optional[Dict[str, float]] = None):
        timestamp = datetime.now().isoformat()
        metrics = metrics or {}
        
        # Track all keys we've encountered
        current_keys = set(metrics.keys())
        all_keys_before = self._model_losses_all_keys.copy()
        self._model_losses_all_keys.update(current_keys)
        
        # Check if we need to update the header (new keys found)
        need_header_update = not self._model_losses_header_written or (current_keys - all_keys_before)
        
        if need_header_update:
            # Read existing data if header needs updating
            existing_data = []
            if self._model_losses_header_written:
                # File exists, we need to preserve existing data and update header
                try:
                    with open(self._model_losses_file, "r", newline="") as f:
                        reader = csv.reader(f)
                        existing_data = list(reader)
                except FileNotFoundError:
                    existing_data = []
            
            # Prepare new header with all keys (logically ordered)
            ordered_keys = sorted(self._model_losses_all_keys)
            new_header = ["timestamp", "iteration", "loss"] + ordered_keys
            
            # Rewrite the file with updated header
            with open(self._model_losses_file, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(new_header)
                
                # If we had existing data, rewrite it with expanded columns
                if existing_data and len(existing_data) > 1:  # More than just header
                    old_header = existing_data[0] if existing_data else []
                    
                    # Create mapping from old positions to new positions
                    old_key_positions = {}
                    if len(old_header) > 3:  # Has metric columns
                        for i, key in enumerate(old_header[3:], start=3):  # Skip timestamp, iteration, loss
                            old_key_positions[key] = i
                    
                    # Rewrite existing data rows with new column structure
                    for row in existing_data[1:]:  # Skip old header
                        if len(row) >= 3:  # Has at least timestamp, iteration, loss
                            new_row = row[:3]  # timestamp, iteration, loss
                            
                            # Add metric values in new ordered sequence
                            for key in ordered_keys:
                                if key in old_key_positions and old_key_positions[key] < len(row):
                                    new_row.append(row[old_key_positions[key]])
                                else:
                                    new_row.append("")  # Missing metric from old data (use empty string for CSV compatibility)
                            
                            wr.writerow(new_row)
            
            self._model_losses_header_written = True
        
        # Append current data row
        ordered_keys = sorted(self._model_losses_all_keys)
        current_row = [timestamp, iteration, loss]
        
        # Add metric values in logical order, empty string for missing metrics (CSV compatible)
        for key in ordered_keys:
            value = metrics.get(key, "")
            # Convert None to empty string for CSV compatibility
            if value is None:
                value = ""
            current_row.append(value)
        
        # Append to file
        with open(self._model_losses_file, "a", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(current_row)


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
