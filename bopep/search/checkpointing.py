import csv
from pathlib import Path
import datetime
import shutil
import json
import pickle
from typing import Optional
from bopep.search.utils import _save_model
import logging

def _next_checkpoint_dir(self) -> Path:
    """Find the next available checkpoint_{i} directory under self.log_dir."""
    base = Path(self.log_dir)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")]
    # Extract suffix numbers
    idxs = []
    for d in existing:
        try:
            idxs.append(int(d.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    next_idx = max(idxs) + 1 if idxs else 0
    return base / f"checkpoint_{next_idx}"

def _save_checkpoint(self, global_iteration: int, force_embeddings: bool = False):
    """
    Save checkpoint with incremental updates.
    
    Args:
        global_iteration: Current iteration number
        force_embeddings: If True, force saving embeddings even if already saved
    """
    checkpoint_dir = self.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and metadata (always updated)
    _save_model(str(checkpoint_dir / "model.pt"), model=self.surrogate_manager.model, surrogate_model_kwargs=self.surrogate_model_kwargs, best_hyperparams=self.best_hyperparams)
    meta_json_path = checkpoint_dir / "metadata.json"
    
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "global_iteration": global_iteration,
        "surrogate_model_kwargs": self.surrogate_model_kwargs,
        "target_structure_path": self.target_structure_path,
        "binding_site_residue_indices": self.binding_site_residue_indices,
        "docker_kwargs": self.docker_kwargs,
        "hpo_kwargs": self.hpo_kwargs,
        "objective_function_kwargs": self.objective_function_kwargs,
        "scoring_kwargs": self.scoring_kwargs,
        "num_docked_sequences": len(self.docked_sequences),
        "num_remaining_sequences": len(self.not_docked_sequences),
    }

    if self.checkpoint_path:
        meta["checkpoint_path"] = self.checkpoint_path
    
    with open(meta_json_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    # Save embeddings only if not saved yet or forced
    embeddings_path = checkpoint_dir / "embeddings.pkl"
    if not self.embeddings_saved or force_embeddings:
        if hasattr(self, 'checkpoint_path') and self.checkpoint_path and not force_embeddings:
            # Copy embeddings from source checkpoint instead of rewriting
            source_embeddings = Path(self.checkpoint_path) / "embeddings.pkl"
            if source_embeddings.exists():
                shutil.copy2(source_embeddings, embeddings_path)
                logging.debug("Copied embeddings from source checkpoint")
            else:
                # Fallback to saving current embeddings
                with open(embeddings_path, "wb") as f:
                    pickle.dump(self.embeddings, f)
                logging.debug("Saved embeddings (fallback)")
        else:
            # Fresh run - save embeddings
            with open(embeddings_path, "wb") as f:
                pickle.dump(self.embeddings, f)
            logging.debug("Saved embeddings")
        
        self.embeddings_saved = True

    # Always update log files
    self._copy_logs_to_checkpoint(checkpoint_dir)
    logging.info("=" * 60)
    logging.info(f"Saved checkpoint at iteration {global_iteration} to {checkpoint_dir}")
    logging.info(f"Checkpoint metadata saved to {meta_json_path}")
    logging.info("=" * 60)

def _copy_logs_to_checkpoint(self, checkpoint_dir: Path):
    """Copy all log files to the checkpoint directory."""
    
    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    log_files = [
        "scores.csv",
        "objectives.csv", 
        "model_losses.csv",
        "predictions.csv.gz",
        "acquisition.csv.gz",
        "hyperparameters.csv"
    ]
    
    logs_copied = 0
    for log_file in log_files:
        source_path = Path(self.log_dir) / log_file
        if source_path.exists():
            dest_path = results_dir / log_file
            shutil.copy2(source_path, dest_path)
            logs_copied += 1
            logging.debug(f"Copied {log_file} to checkpoint")
    
    logging.info(f"Copied {logs_copied} log files to checkpoint")

def _setup_checkpoint_dir(self, continue_from_checkpoint: bool):
    """
    Setup the checkpoint directory for this run.
    
    For fresh runs: Use checkpoint_0
    For continued runs: Increment the checkpoint number (e.g., checkpoint_0 -> checkpoint_1)
    """
    base = Path(self.log_dir)
    
    if not continue_from_checkpoint:
        self.checkpoint_dir = base / "checkpoint_0"
        self.embeddings_saved = False
    else:
        checkpoint_path = Path(self.checkpoint_path)
        checkpoint_name = checkpoint_path.name
        
        if checkpoint_name.startswith("checkpoint_"):
            try:
                current_num = int(checkpoint_name.split("_", 1)[1])
                next_num = current_num + 1
                self.checkpoint_dir = base / f"checkpoint_{next_num}"
            except (IndexError, ValueError):
                self.checkpoint_dir = self._next_checkpoint_dir()
        else:
            self.checkpoint_dir = self._next_checkpoint_dir()
        self.embeddings_saved = False 
    
    logging.info(f"Checkpoints for this run will be saved to: {self.checkpoint_dir}")



def _rebuild_logs_from_csvs(self, checkpoint_path: Optional[Path] = None):
    """
    Rebuild the optimization state from CSV log files.
    
    Args:
        checkpoint_path: If provided, read logs from checkpoint/results/ directory
                        Otherwise read from current log_dir
    """
    if checkpoint_path:
        # Read from checkpoint results directory
        log_base = checkpoint_path / "results"
    else:
        # Read from current log directory
        log_base = Path(self.log_dir)
        
    scores_path = log_base / "scores.csv"
    self.scores = {}
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pep = row["sequence"]
            score_cols = [c for c in reader.fieldnames
                            if c not in ("timestamp", "iteration", "sequence", "phase")]
            sc = {}
            for col in score_cols:
                val = row[col]
                if val in (None, ""):
                    sc[col] = None
                elif val in ("True", "False"):
                    sc[col] = val == "True"
                else:
                    sc[col] = float(val)
            self.scores[pep] = sc

    obj_path = log_base / "objectives.csv"
    self.all_logged_objectives = set()
    with open(obj_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            self.all_logged_objectives.add(row["sequence"])

    self.docked_sequences = set(self.scores.keys())
    self.not_docked_sequences = set(self.embeddings.keys()) - self.docked_sequences


def _validate_checkpoint(checkpoint_path: Path):
    """Validate checkpoint integrity."""
    required_files = [
        "metadata.json",
        "embeddings.pkl",
        "model.pt",
        "results/scores.csv",
        "results/objectives.csv"
    ]
    optional_files = [
        "results/model_losses.csv",
        "results/predictions.csv.gz",
        "results/acquisition.csv.gz",
        "results/hyperparameters.csv"
    ]
    
    missing = [f for f in required_files if not (checkpoint_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint incomplete. Missing required files: {missing}"
        )
    missing_optional = [f for f in optional_files if not (checkpoint_path / f).exists()]
    if missing_optional:
        logging.warning(f"Checkpoint missing optional files: {missing_optional}")
