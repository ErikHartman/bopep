import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os
from sklearn.preprocessing import StandardScaler

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
    MVE,
    tune_uncertainty_parameter
)

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
                           x_scaler, y_scaler, ax):
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
    ax.scatter(x_train, y_train, alpha=0.3, s=10, label='Training data')
    
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
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if i == 0 else None
        )
    
    # Set plot limits and labels
    ax.set_xlim(-7, 7)
    ax.set_ylim(-150, 150)
    ax.set_title(model_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)

def plot_tuned_model(model_type, x_train, y_train, x_test, test_embedding_tensor,
                     input_dim, optimal_param, x_scaler, y_scaler, ax):
    """
    Train and plot a model with tuned parameters.
    
    Args:
        model_type: Type of model to plot
        x_train, y_train: Training data
        x_test: Test x values
        test_embedding_tensor: Tensor of test embeddings
        input_dim: Input dimension
        optimal_param: Optimal parameter value from tuning
        x_scaler, y_scaler: Scalers for data transformation
        ax: Axes for plotting
    """
    # Common parameters
    common_params = {
        "input_dim": input_dim,
        "hidden_dims": [32, 32],
        "network_type": "mlp",
    }
    
    # Convert training data to dictionary format for fit_dict
    # Scale the data
    x_train_scaled = x_scaler.transform(x_train.reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Create embedding and scores dictionaries
    embedding_dict = {}
    scores_dict = {}
    for i in range(len(x_train_scaled)):
        key = f"sample_{i}"
        embedding_dict[key] = np.array([x_train_scaled[i]], dtype=np.float32)
        scores_dict[key] = float(y_train_scaled[i])
    
    # Create model with tuned parameter
    if model_type == "mve":
        model = MVE(mve_regularization=optimal_param, **common_params).to(device)
        param_name = "mve_regularization"
    elif model_type == "der":
        model = DeepEvidentialRegression(evidential_regularization=optimal_param, **common_params).to(device)
        param_name = "evidential_regularization"
    elif model_type == "nn_ensemble":
        model = NeuralNetworkEnsemble(n_networks=int(optimal_param), **common_params).to(device)
        param_name = "n_networks"
    elif model_type == "mc_dropout":
        model = MonteCarloDropout(dropout_rate=optimal_param, mc_samples=20, **common_params).to(device)
        param_name = "dropout_rate"
    
    # Train model using fit_dict
    model.fit_dict(
        embedding_dict=embedding_dict,
        scores_dict=scores_dict,
        epochs=1000,
        batch_size=16,
        learning_rate=0.001,
        device=device,
        verbose=False
    )
    
    # Plot tuned model predictions
    plot_model_predictions(
        model, 
        f"{model_type.upper()} ({param_name}={optimal_param:.4f})", 
        x_train, y_train, x_test, test_embedding_tensor,
        x_scaler, y_scaler, ax
    )

def main():
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
    
    # Convert to tensors
    x_train_tensor = torch.tensor(x_train_scaled.reshape(-1, 1), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x_scaled.reshape(-1, 1), dtype=torch.float32)
    
    # Input dimension
    input_dim = 1
    
    # Create a single figure with 2x2 grid for all models
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Tune MVE model
    print("\n\nTuning MVE model...")
    mve_optimal = tune_uncertainty_parameter(
        model_type="mve",
        X=x_train_tensor,
        y=y_train_tensor,
        input_dim=input_dim,
        hidden_dims=[32, 32],
        network_type="mlp",
        epochs=200,
        batch_size=16,
        n_splits=3,
        param_values=np.logspace(-3, 0, 8),  # Test 8 values from 0.001 to 1.0
    )
    print(f"Optimal MVE regularization: {mve_optimal:.4f}")
    plot_tuned_model("mve", x_values, y_values, test_x, test_x_tensor, 
                    input_dim, mve_optimal, x_scaler, y_scaler, axes[0])
    
    # Tune DER model
    print("\n\nTuning Deep Evidential Regression model...")
    der_optimal = tune_uncertainty_parameter(
        model_type="der",
        X=x_train_tensor,
        y=y_train_tensor,
        input_dim=input_dim,
        hidden_dims=[32, 32],
        network_type="mlp",
        epochs=200,
        batch_size=16,
        n_splits=3,
        param_values=np.logspace(-3, 0, 8),  # Test 8 values from 0.001 to 1.0
    )
    print(f"Optimal Evidential regularization: {der_optimal:.4f}")
    plot_tuned_model("der", x_values, y_values, test_x, test_x_tensor, 
                    input_dim, der_optimal, x_scaler, y_scaler, axes[1])
    
    # Tune Neural Network Ensemble
    print("\n\nTuning Neural Network Ensemble...")
    nn_ensemble_optimal = tune_uncertainty_parameter(
        model_type="nn_ensemble",
        X=x_train_tensor,
        y=y_train_tensor,
        input_dim=input_dim,
        hidden_dims=[32, 32],
        network_type="mlp",
        epochs=200,
        batch_size=16,
        n_splits=3,
        param_values=np.arange(2, 15, 2),  # Test [2, 4, 6, 8, 10, 12, 14]
    )
    print(f"Optimal number of networks: {int(nn_ensemble_optimal)}")
    plot_tuned_model("nn_ensemble", x_values, y_values, test_x, test_x_tensor, 
                    input_dim, nn_ensemble_optimal, x_scaler, y_scaler, axes[2])
    
    # Tune Monte Carlo Dropout
    print("\n\nTuning Monte Carlo Dropout...")
    mc_dropout_optimal = tune_uncertainty_parameter(
        model_type="mc_dropout",
        X=x_train_tensor,
        y=y_train_tensor,
        input_dim=input_dim,
        hidden_dims=[32, 32],
        network_type="mlp",
        epochs=200,
        batch_size=16,
        n_splits=3,
        param_values=np.linspace(0.05, 0.5, 8),  # Test 8 values from 0.05 to 0.5
        mc_samples=20  # Set MC samples for tuning
    )
    print(f"Optimal dropout rate: {mc_dropout_optimal:.4f}")
    plot_tuned_model("mc_dropout", x_values, y_values, test_x, test_x_tensor, 
                    input_dim, mc_dropout_optimal, x_scaler, y_scaler, axes[3])
    
    plt.tight_layout()
    
    # Save the combined figure
    os.makedirs("examples/figures", exist_ok=True)
    plt.savefig("examples/figures/uncertainty_tuning_comparison.png", dpi=300)
    print(f"Figure saved as 'examples/figures/uncertainty_tuning_comparison.png'")
    plt.show()
    
    # Summarize all optimal parameters
    print("\n\n===== Optimal Parameters Summary =====")
    print(f"MVE regularization:          {mve_optimal:.4f}")
    print(f"Evidential regularization:   {der_optimal:.4f}")
    print(f"Neural Network Ensemble size: {int(nn_ensemble_optimal)}")
    print(f"Monte Carlo Dropout rate:    {mc_dropout_optimal:.4f}")

if __name__ == "__main__":
    main()
