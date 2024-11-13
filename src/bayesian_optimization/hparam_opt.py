import optuna
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import StratifiedKFold


optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_stratified_folds(X, y, n_splits, random_state):
    # Define bin edges manually: 0-0.1, 0.1-0.4, over 0.4
    bin_edges = np.array([0, 0.1, 0.4, np.inf]) 
    y_binned = np.digitize(y, bins=bin_edges, right=False) 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf.split(X, y_binned)

def optimize_hyperparameters(X_scaled, y_train, previous_study=None):

    def objective(trial):

        n_layers = trial.suggest_int('n_layers', 1, 8)  # Reduced from 10 to 5
        hidden_layer_sizes = []

        first_layer_size = trial.suggest_int('n_units_l1', 64, 1024, log=True)
        hidden_layer_sizes.append(first_layer_size)

        for _ in range(1, n_layers):
            next_layer_size = max(int(hidden_layer_sizes[-1] / 2), 8)
            hidden_layer_sizes.append(next_layer_size)
        hidden_layer_sizes = tuple(hidden_layer_sizes)

        
        params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-3, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'learning_rate': 'adaptive',
            'random_state': 42,
            'max_iter': 3000,
            'tol': 1e-4,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
        }


        skf = get_stratified_folds(X_scaled, y_train, n_splits=5, random_state=42)
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

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    if previous_study is None:
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    else:
        study = previous_study

    study.optimize(objective, n_trials=100, n_jobs=25)

    best_params = study.best_params
    return best_params, study
