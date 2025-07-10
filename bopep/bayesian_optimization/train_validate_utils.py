import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Optional, Union
import logging

def _compute_model_metrics(self, predictions_dict: dict, objectives: dict):
        peptides = list(predictions_dict.keys())
        actual = np.array([objectives[p] for p in peptides])
        predicted = np.array([predictions_dict[p][0] for p in peptides])
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        return {"r2": r2, "mae": mae}

def _compute_split_indices(
    self, total_samples: int, n_validate: Union[int, float]
) -> Optional[int]:
    """Return num_validate or None if split is infeasible."""
    if isinstance(n_validate, float):
        num = int(total_samples * n_validate)
    else:
        num = n_validate

    if (
        num < self.MIN_VALIDATION_SAMPLES
        or (total_samples - num) < self.MIN_TRAINING_SAMPLES
    ):
        logging.warning(
            f"Cannot split {total_samples} samples into "
            f"{self.MIN_TRAINING_SAMPLES} train + "
            f"{self.MIN_VALIDATION_SAMPLES} val; training on all."
        )
        return None

    return num

def _train_and_validate(self, train_emb, train_obj, val_emb, val_obj):
    """Train on train set, evaluate on both splits."""
    loss = self.model.fit_dict(
        embedding_dict=train_emb,
        objective_dict=train_obj,
        val_embedding_dict=val_emb,
        val_objective_dict=val_obj,
        epochs=self.best_hyperparams.get("epochs", 100),
        learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
        batch_size=self.best_hyperparams.get("batch_size", 16),
        device=self.device,
    )
    train_pred = self.model.predict_dict(train_emb, device=self.device)
    val_pred = self.model.predict_dict(val_emb, device=self.device)
    train_m = self._compute_model_metrics(train_pred, train_obj)
    val_m = self._compute_model_metrics(val_pred, val_obj)

    metrics = {
        "train_r2": train_m["r2"],
        "train_mae": train_m["mae"],
        "val_r2": val_m["r2"],
        "val_mae": val_m["mae"],
    }
    logging.info(
        f"Loss {loss:.4f}, train R2 {train_m['r2']:.4f}, "
        f"val R2 {val_m['r2']:.4f} "
        f"(N_train={len(train_emb)}, N_val={len(val_emb)})"
    )
    return loss, metrics

def _train_on_all(self, embeddings, objectives):
    """Train on the entire dataset (no validation)."""
    loss = self.model.fit_dict(
        embedding_dict=embeddings,
        objective_dict=objectives,
        epochs=self.best_hyperparams.get("epochs", 100),
        learning_rate=self.best_hyperparams.get("learning_rate", 1e-3),
        batch_size=self.best_hyperparams.get("batch_size", 16),
        device=self.device,
    )
    preds = self.model.predict_dict(embeddings, device=self.device)
    m = self._compute_model_metrics(preds, objectives)
    metrics = {"r2": m["r2"], "mae": m["mae"]}
    logging.info(f"Loss {loss:.4f}, R2 {m['r2']:.4f}, N={len(embeddings)}")
    return loss, metrics

def _split_train_validation(
    self, docked_embeddings: dict, objectives: dict, num_validate: int
):
    """
    Split the available data into training and validation sets.
    """
    peptides = list(objectives.keys())
    val_indices = np.random.choice(len(peptides), num_validate, replace=False)
    val_peptides = [peptides[i] for i in val_indices]
    train_peptides = [p for p in peptides if p not in val_peptides]
    train_embeddings = {p: docked_embeddings[p] for p in train_peptides}
    train_objectives = {p: objectives[p] for p in train_peptides}
    val_embeddings = {p: docked_embeddings[p] for p in val_peptides}
    val_objectives = {p: objectives[p] for p in val_peptides}

    logging.info(
        f"Split data into {len(train_peptides)} training and {len(val_peptides)} validation samples"
    )

    return train_embeddings, train_objectives, val_embeddings, val_objectives

