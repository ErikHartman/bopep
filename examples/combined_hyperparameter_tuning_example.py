import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
import os
from sklearn.preprocessing import StandardScaler

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
    MVE
)
from bopep.surrogate_model.hyperparameter_tuner import tune_hyperparams

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cubic_data(n_samples: int = 100, 
                        noise_std: float = 3.0, 
                        x_range: Tuple[float, float] = (-3, 3)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a cubic function with Gaussian noise.
    """
    # Generate x values with good coverage over the range
    x_segments = np.linspace(x_range[0], x_range[1], num=10)
    x_values = np.array([])
    for i in range(len(x_segments)-1):
        segment_samples = n_samples // (len(x_segments)-1)
        if i == len(x_segments)-2:
            segment_samples = n_samples - len(x_values)
        segment_x = np.random.uniform(x_segments[i], x_segments[i+1], segment_samples)
        x_values = np.append(x_values, segment_x)
    
    # Generate y values with noise
    y_values = x_values**3 + np.random.normal(0, noise_std, n_samples)
    
    return x_values, y_values

def plot_model_predictions(model, model_name, x_train, y_train, x_test, test_embedding_tensor,
                           x_scaler, y_scaler, ax, params=None):
    """
    Plot model predictions with uncertainty.
    """
    # Get model predictions
    model.eval()
    with torch.no_grad():
        means, stds = model.forward_predict(test_embedding_tensor.to(device))
    
    # Convert to numpy and scale back
    means = means.cpu().numpy().flatten()
    stds = stds.cpu().numpy().flatten()
    means = y_scaler.inverse_transform(means.reshape(-1, 1)).flatten()
    stds = stds * y_scaler.scale_
    
    # Plot true function
    x_true = np.linspace(min(x_test), max(x_test), 1000)
    y_true = x_true**3
    ax.plot(x_true, y_true, 'k-', linewidth=1, label='True function')
    
    # Plot training data
    ax.scatter(x_train, y_train, alpha=0.3, s=10, color='k', label='Training data')
    
    # Plot predictions
    ax.plot(x_test, means, 'r-', linewidth=1.5, label='Prediction')
    
    # Plot uncertainty bands
    n_stds = 4
    for i, k in enumerate(np.linspace(0, n_stds, 4)):
        ax.fill_between(
            x_test,
            means - k * stds,
            means + k * stds,
            alpha=0.3,
            edgecolor=None,
            facecolor="#983ed4",
            linewidth=0,
            zorder=1,
            label="Unc." if i == 0 else None
        )
    
    # Set plot limits and labels
    ax.set_xlim(-7, 7)
    ax.set_ylim(-150, 150)
    
    # Add a subtitle with key parameters if provided
    if params:
        subtitle = []
        if "hidden_dims" in params:
            subtitle.append(f"Layers: {params['hidden_dims']}")
        if "uncertainty_param" in params:
            subtitle.append(f"Unc param: {params['uncertainty_param']:.4f}")
        if "learning_rate" in params:
            subtitle.append(f"LR: {params['learning_rate']:.4f}")
        if subtitle:
            subtitle_str = '\n'.join(subtitle)
            ax.set_title(f"{model_name}\n{subtitle_str}", fontsize=10)
        else:
            ax.set_title(model_name)
    else:
        ax.set_title(model_name)
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize='small', frameon=False)
    
    return means, stds  # Return predictions for individual plotting


def save_individual_plot(model_name, x_train, y_train, x_test, means, stds):

    fig, ax = plt.figure(figsize=(3,3)), plt.gca()
    
    # Plot true function
    x_true = np.linspace(min(x_test), max(x_test), 1000)
    y_true = x_true**3
    ax.plot(x_true, y_true, 'k-', linewidth=1, label='True function')
    
    # Plot training data
    ax.scatter(x_train, y_train, alpha=0.8, s=6, color='k', label='Training data')
    
    # Plot predictions
    ax.plot(x_test, means, '-', color="#ff8b50", linewidth=1.5, label='Prediction')
    
    # Plot uncertainty bands
    n_stds = 4
    for i, k in enumerate(np.linspace(0, n_stds, 4)):
        ax.fill_between(
            x_test,
            means - k * stds,
            means + k * stds,
            alpha=0.3,
            edgecolor=None,
            facecolor="#ac69da",
            linewidth=0,
            zorder=1,
            label="Unc." if i == 0 else None
        )
    
    # Set plot limits and labels
    ax.set_xlim(-7, 7)
    ax.set_ylim(-150, 150)
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize='small')
    
    # Save as SVG
    file_path = f"examples/figures/{model_name.lower()}.svg"
    plt.tight_layout()
    #plt.savefig(file_path, format='svg')
    plt.close(fig)
    print(f"Individual plot saved as '{file_path}'")



def create_and_train_model(model_type, best_params, embedding_dict, objective_dict):
    """
    Create and train a model with the best parameters from the tuner.
    """
    # Extract common parameters
    hidden_dims = best_params.get("hidden_dims", [64,64])
    learning_rate = best_params.get("learning_rate", 0.001)
    epochs = best_params.get("epochs", 100)
    uncertainty_param = best_params.get("uncertainty_param", 0.1)
    
    # Create model based on type
    if model_type == "mve":
        model = MVE(
            input_dim=1,
            hidden_dims=hidden_dims,
            mve_regularization=uncertainty_param
        )
    elif model_type == "deep_evidential":
        model = DeepEvidentialRegression(
            input_dim=1,
            hidden_dims=hidden_dims,
            evidential_regularization=uncertainty_param
        )
    elif model_type == "nn_ensemble":
        model = NeuralNetworkEnsemble(
            input_dim=1,
            hidden_dims=hidden_dims,
            n_networks=int(uncertainty_param)
        )
    elif model_type == "mc_dropout":
        model = MonteCarloDropout(
            input_dim=1,
            hidden_dims=hidden_dims,
            dropout_rate=uncertainty_param,
            mc_samples=20
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.to(device)
    model.fit_dict(
        embedding_dict=embedding_dict,
        objective_dict=objective_dict,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=False
    )
    
    return model

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    train_range = (-4, 4)
    x_values, y_values = generate_cubic_data(n_samples=150, noise_std=3.0, x_range=train_range)
    test_x = np.linspace(-7, 7, 200)
    
    # Scale the data
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_values.reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.fit_transform(y_values.reshape(-1, 1)).flatten()
    test_x_scaled = x_scaler.transform(test_x.reshape(-1, 1)).flatten()
    
    # Convert to tensors for test prediction
    test_x_tensor = torch.tensor(test_x_scaled.reshape(-1, 1), dtype=torch.float32)
    
    # Create embedding and scores dictionaries for tuning
    embedding_dict = {}
    objective_dict = {}
    for i in range(len(x_train_scaled)):
        key = f"sample_{i}"
        # Store as single feature arrays
        embedding_dict[key] = np.array([x_train_scaled[i]], dtype=np.float32)
        objective_dict[key] = float(y_train_scaled[i])
    
    # Set up plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # List of model types to tune
    model_types = ["mve", "deep_evidential", "nn_ensemble", "mc_dropout"]
    results = {}

    # fixed_params = {"mve": 0.3,  "deep_evidential": 1, "nn_ensemble": 10, "mc_dropout": 0.3}
    
    # Run hyperparameter tuning for each model type
    for i, model_type in enumerate(model_types):
        print(f"\n\nTuning {model_type.upper()} model with combined hyperparameter search...")
        
        best_params, study = tune_hyperparams(
            model_type=model_type,
            embedding_dict=embedding_dict,
            objective_dict=objective_dict,
            n_trials=10,
            n_splits=3,
            random_state=SEED,
            hidden_dim_max=32,

        )
        
        
        print(f"Best {model_type.upper()} parameters: {best_params}")
        
        results[model_type] = best_params
        
        # Create and train model with best parameters
        model = create_and_train_model(
            model_type=model_type,
            best_params= best_params, # {"uncertainty_param": fixed_params[model_type], "epochs":500}, #
            embedding_dict=embedding_dict,
            objective_dict=objective_dict
        )
        
        # Plot results
        means, stds = plot_model_predictions(
            model=model,
            model_name=model_type.upper(),
            x_train=x_values,
            y_train=y_values,
            x_test=test_x,
            test_embedding_tensor=test_x_tensor,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            ax=axes[i],
            params=best_params
        )
        
        # Create and save individual SVG plot
        save_individual_plot(
            model_name=model_type.upper(),
            x_train=x_values,
            y_train=y_values,
            x_test=test_x,
            means=means,
            stds=stds,
        )
    
    plt.tight_layout()
    
    # Save the combined figure
    os.makedirs("examples/figures", exist_ok=True)
    plt.savefig("examples/figures/combined_hyperparameter_tuning.png", dpi=300)
    print("Figure saved as 'examples/figures/combined_hyperparameter_tuning.png'")
    
    # Print summary of all optimized parameters
    print("\n\n===== Optimal Parameters Summary =====")
    for model_type, params in results.items():
        print(f"\n{model_type.upper()}:")
        for param_name, param_value in params.items():
            print(f"  {param_name}: {param_value}")

if __name__ == "__main__":
    main()
