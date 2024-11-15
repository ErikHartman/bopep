import optuna
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Ensure Optuna's verbosity is set appropriately
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_stratified_folds(X, y, n_splits, random_state, bin_edges=None):
    if bin_edges is None:
        # Default bin edges: 0-0.1, 0.1-0.4, over 0.4
        bin_edges = np.array([0, 0.1, 0.4, np.inf])
    y_binned = np.digitize(y, bins=bin_edges, right=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y_binned)

def optimize_hyperparameters(
    X_scaled,
    y_train,
    previous_study=None,
    n_layers_range=(1, 5),
    n_units_l1_range=(64, 1024),
    alpha_range=(1e-5, 1e-3),
    learning_rate_init_range=(1e-4, 1e-2),
    n_splits=5,
    random_state=42,
    n_trials=50,
    n_jobs=1,
    pruner_n_warmup_steps=5,
    direction='maximize',
    pruner_type='MedianPruner',
    sampler_type='TPESampler',
    max_iter=3000,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    hidden_layer_decrease_factor=2,
    min_hidden_layer_size=8,
    bin_edges=None
):
    """
    Optimize hyperparameters for an MLPRegressor using Optuna.

    Parameters:
    - X_scaled: Scaled input features.
    - y_train: Target values.
    - previous_study: An existing Optuna study to continue optimization.
    - n_layers_range: Tuple specifying the range of layers (min_layers, max_layers).
    - n_units_l1_range: Tuple specifying the range for the first hidden layer units.
    - alpha_range: Tuple specifying the range for the regularization parameter alpha.
    - learning_rate_init_range: Tuple specifying the range for the initial learning rate.
    - n_splits: Number of splits for cross-validation.
    - random_state: Random state for reproducibility.
    - n_trials: Number of trials for the optimization.
    - n_jobs: Number of parallel jobs for Optuna.
    - pruner_n_warmup_steps: Number of warm-up steps for the pruner.
    - direction: Direction of optimization ('maximize' or 'minimize').
    - pruner_type: Type of pruner to use ('MedianPruner', etc.).
    - sampler_type: Type of sampler to use ('TPESampler', etc.).
    - max_iter: Maximum number of iterations for the MLPRegressor.
    - tol: Tolerance for the optimizer.
    - early_stopping: Whether to use early stopping.
    - validation_fraction: Fraction of data for validation if early stopping is used.
    - n_iter_no_change: Number of iterations with no improvement to wait before stopping.
    - hidden_layer_decrease_factor: Factor by which to decrease hidden layer sizes.
    - min_hidden_layer_size: Minimum size of hidden layers.
    - bin_edges: Custom bin edges for stratification.

    Returns:
    - best_params: Best hyperparameters found.
    - study: The Optuna study object.
    """

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', *n_layers_range)
        hidden_layer_sizes = []

        first_layer_size = trial.suggest_int('n_units_l1', *n_units_l1_range, log=True)
        hidden_layer_sizes.append(first_layer_size)

        for _ in range(1, n_layers):
            next_layer_size = max(int(hidden_layer_sizes[-1] / hidden_layer_decrease_factor), min_hidden_layer_size)
            hidden_layer_sizes.append(next_layer_size)
        hidden_layer_sizes = tuple(hidden_layer_sizes)

        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': trial.suggest_float('alpha', *alpha_range, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', *learning_rate_init_range, log=True),
            'learning_rate': 'adaptive',
            'random_state': random_state,
            'max_iter': max_iter,
            'tol': tol,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'verbose': False,
        }

        skf = get_stratified_folds(X_scaled, y_train, n_splits=n_splits, random_state=random_state, bin_edges=bin_edges)
        val_scores = []

        for step, (train_index, val_index) in enumerate(skf):
            X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            model = MLPRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            val_score = model.score(X_val_fold, y_val_fold)  # R^2 score
            val_scores.append(val_score)

            # Prune unpromising trials
            trial.report(np.mean(val_scores), step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(val_scores)

    # Initialize sampler
    if sampler_type == 'TPESampler':
        sampler = optuna.samplers.TPESampler()
    else:
        # Add other sampler options as needed
        sampler = optuna.samplers.TPESampler()

    # Initialize pruner
    if pruner_type == 'MedianPruner':
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruner_n_warmup_steps)
    else:
        # Add other pruner options as needed
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=pruner_n_warmup_steps)

    if previous_study is None:
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    else:
        study = previous_study

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params = study.best_params
    return best_params, study
