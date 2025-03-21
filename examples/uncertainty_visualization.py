import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import os
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from bopep.surrogate_model import (
    MonteCarloDropout,
    NeuralNetworkEnsemble,
    DeepEvidentialRegression,
    MVE
)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def generate_cubic_data(n_samples: int = 100, 
                        noise_std: float = 3.0, 
                        x_range: Tuple[float, float] = (-3, 3)) -> Tuple[Dict[str, np.ndarray], Dict[str, float], np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a cubic function with Gaussian noise.
    
    Args:
        n_samples: Number of data points to generate
        noise_std: Standard deviation of the Gaussian noise
        x_range: Range of x values (min, max)
        
    Returns:
        embedding_dict: Dictionary of input values {point_id: [x]}
        scores_dict: Dictionary of output values {point_id: y}
        x_values: Raw x values for plotting
        y_values: Raw y values for plotting
    """
    # Generate x values - use stratified sampling across the range for better coverage
    x_segments = np.linspace(x_range[0], x_range[1], num=10)
    x_values = np.array([])
    for i in range(len(x_segments)-1):
        segment_samples = n_samples // (len(x_segments)-1)
        if i == len(x_segments)-2:  # Make sure we get exactly n_samples
            segment_samples = n_samples - len(x_values)
        segment_x = np.random.uniform(x_segments[i], x_segments[i+1], segment_samples)
        x_values = np.append(x_values, segment_x)
    
    # Generate y values: y = x^3 + N(0, noise_std)
    y_values = x_values**3 + np.random.normal(0, noise_std, n_samples)
    
    # Create dictionaries expected by surrogate models
    embedding_dict = {f"point_{i}": np.array([x], dtype=np.float32) for i, x in enumerate(x_values)}
    scores_dict = {f"point_{i}": float(y) for i, y in enumerate(y_values)}
    
    return embedding_dict, scores_dict, x_values, y_values

def compare_all_hyperparameters(embedding_dict, scores_dict, test_x):
    """
    Train different surrogate models with varying hyperparameters and visualize their predictions
    in a comprehensive 4x4 grid.
    
    Args:
        embedding_dict: Dictionary of input values for training
        scores_dict: Dictionary of output values for training
        test_x: Array of x values for testing/visualization
    """
    # Extract training data for plotting
    x_train = np.array([embedding_dict[key][0] for key in embedding_dict.keys()])
    y_train = np.array([scores_dict[key] for key in scores_dict.keys()])
    
    # Standardize inputs and outputs
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Create standardized training dictionaries
    embedding_dict_scaled = {f"point_{i}": np.array([x], dtype=np.float32) 
                            for i, x in enumerate(x_train_scaled)}
    scores_dict_scaled = {f"point_{i}": float(y) for i, y in enumerate(y_train_scaled)}
    
    # Standardize test data
    test_x_scaled = x_scaler.transform(test_x.reshape(-1, 1)).flatten()
    test_embedding_dict_scaled = {f"test_{i}": np.array([x], dtype=np.float32) 
                                for i, x in enumerate(test_x_scaled)}
    
    # Define common model parameters
    common_params = {
        "input_dim": 1,  # 1-dimensional input (x)
        "hidden_dims": [32, 32],  # Reasonable network size for this task
    }
    
    # Define hyperparameters to vary for each model
    mc_dropout_rates = [0.1, 0.3, 0.5, 0.7]
    ensemble_sizes = [1, 5, 10, 20]
    evidential_reg_values = [0, 0.1, 0.5, 1.0]
    mve_reg_values = [0, 0.01, 0.1, 0.9]
    
    # Create a large figure with 4x4 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    
    # True function (without noise)
    x_true = np.linspace(min(test_x), max(test_x), 1000)
    y_true = x_true**3
    
    # 1. Train and plot MC Dropout with varying dropout rates (top-left 2x2)
    for i, dropout_rate in enumerate(mc_dropout_rates):
        row, col = i // 2, i % 2  # Position in the top-left 2x2 grid
        model_name = f"MC Dropout (rate={dropout_rate})"
        print(f"Training {model_name}...")
        
        model = MonteCarloDropout(
            dropout_rate=dropout_rate,
            mc_samples=20,
            network_type="mlp",
            **common_params
        ).to(device)
        
        try:
            # Train model and generate predictions
            train_and_plot_model(
                model=model,
                model_name=model_name,
                embedding_dict_scaled=embedding_dict_scaled,
                scores_dict_scaled=scores_dict_scaled,
                test_embedding_dict_scaled=test_embedding_dict_scaled,
                x_train=x_train,
                y_train=y_train,
                test_x=test_x,
                x_true=x_true,
                y_true=y_true,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                ax=axes[row, col]
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            axes[row, col].text(0.5, 0.5, f"Error: {type(e).__name__}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[row, col].transAxes, fontsize=12, color='red')
            axes[row, col].set_title(f"{model_name} (Failed)")
    
    # 2. Train and plot Neural Network Ensemble with varying ensemble sizes (top-right 2x2)
    for i, n_networks in enumerate(ensemble_sizes):
        row, col = i // 2, (i % 2) + 2  # Position in the top-right 2x2 grid
        model_name = f"NN Ensemble (size={n_networks})"
        print(f"Training {model_name}...")
        
        model = NeuralNetworkEnsemble(
            n_networks=n_networks,
            network_type="mlp",
            **common_params
        ).to(device)
        
        try:
            # Train model and generate predictions
            train_and_plot_model(
                model=model,
                model_name=model_name,
                embedding_dict_scaled=embedding_dict_scaled,
                scores_dict_scaled=scores_dict_scaled,
                test_embedding_dict_scaled=test_embedding_dict_scaled,
                x_train=x_train,
                y_train=y_train,
                test_x=test_x,
                x_true=x_true,
                y_true=y_true,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                ax=axes[row, col]
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            axes[row, col].text(0.5, 0.5, f"Error: {type(e).__name__}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[row, col].transAxes, fontsize=12, color='red')
            axes[row, col].set_title(f"{model_name} (Failed)")
    
    # 3. Train and plot Deep Evidential Regression with varying regularization (bottom-left 2x2)
    for i, reg_value in enumerate(evidential_reg_values):
        row, col = (i // 2) + 2, i % 2  # Position in the bottom-left 2x2 grid
        model_name = f"Evidential (reg={reg_value})"
        print(f"Training {model_name}...")
        
        model = DeepEvidentialRegression(
            network_type="mlp",
            evidential_regularization=reg_value,
            **common_params
        ).to(device)
        
        try:
            # Train model and generate predictions
            train_and_plot_model(
                model=model,
                model_name=model_name,
                embedding_dict_scaled=embedding_dict_scaled,
                scores_dict_scaled=scores_dict_scaled,
                test_embedding_dict_scaled=test_embedding_dict_scaled,
                x_train=x_train,
                y_train=y_train,
                test_x=test_x,
                x_true=x_true,
                y_true=y_true,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                ax=axes[row, col],
                is_evidential=True
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            axes[row, col].text(0.5, 0.5, f"Error: {type(e).__name__}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[row, col].transAxes, fontsize=12, color='red')
            axes[row, col].set_title(f"{model_name} (Failed)")
    
    # 4. Train and plot MVE with varying regularization (bottom-right 2x2)
    for i, reg_value in enumerate(mve_reg_values):
        row, col = (i // 2) + 2, (i % 2) + 2  # Position in the bottom-right 2x2 grid
        model_name = f"MVE (reg={reg_value})"
        print(f"Training {model_name}...")
        
        model = MVE(
            network_type="mlp",
            mve_regularization=reg_value,
            **common_params
        ).to(device)
        
        try:
            # Train model and generate predictions
            train_and_plot_model(
                model=model,
                model_name=model_name,
                embedding_dict_scaled=embedding_dict_scaled,
                scores_dict_scaled=scores_dict_scaled,
                test_embedding_dict_scaled=test_embedding_dict_scaled,
                x_train=x_train,
                y_train=y_train,
                test_x=test_x,
                x_true=x_true,
                y_true=y_true,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                ax=axes[row, col],
            )
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            axes[row, col].text(0.5, 0.5, f"Error: {type(e).__name__}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[row, col].transAxes, fontsize=12, color='red')
            axes[row, col].set_title(f"{model_name} (Failed)")
    
    # Add overall titles to each quadrant
    fig.text(0.25, 0.95, "MC Dropout (varying dropout rate)", ha='center', va='center', fontsize=16)
    fig.text(0.75, 0.95, "Neural Network Ensemble (varying ensemble size)", ha='center', va='center', fontsize=16)
    fig.text(0.25, 0.48, "Deep Evidential Regression (varying regularization)", ha='center', va='center', fontsize=16)
    fig.text(0.75, 0.48, "Mean Variance Estimation (varying regularization)", ha='center', va='center', fontsize=16)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the overall titles
    
    # Create output directory if it doesn't exist
    os.makedirs("examples/figures", exist_ok=True)
    
    # Save the figure
    plt.savefig("examples/figures/hyperparameter_comparison.png", dpi=300)
    print(f"Figure saved as 'examples/figures/hyperparameter_comparison.png'")
    
    # Also display it if in an interactive environment
    plt.show()

def train_and_plot_model(model, model_name, embedding_dict_scaled, scores_dict_scaled, 
                         test_embedding_dict_scaled, x_train, y_train, test_x, x_true, y_true,
                         x_scaler, y_scaler, ax, is_evidential=False):
    """
    Helper function to train a model and plot its predictions.
    """
    # Increase epochs for better training, especially for evidential models
    epochs = 1000 if is_evidential else 500
    
    # Train the model
    model.fit_dict(
        embedding_dict=embedding_dict_scaled,
        scores_dict=scores_dict_scaled,
        epochs=epochs,  # Increased for better uncertainty estimates
        batch_size=16,
        learning_rate=0.001,
        device=device,
        verbose=False,  # Reduce output noise
    )
    
    # Generate predictions
    predictions_scaled = model.predict_dict(
        embedding_dict=test_embedding_dict_scaled,
        batch_size=32,
        device=device
    )
    
    # Extract predictions
    means_scaled = np.array([predictions_scaled[key][0] for key in test_embedding_dict_scaled.keys()])
    stds_scaled = np.array([predictions_scaled[key][1] for key in test_embedding_dict_scaled.keys()])
    
    # Transform predictions back to original scale
    means = y_scaler.inverse_transform(means_scaled.reshape(-1, 1)).flatten()
    stds = stds_scaled * y_scaler.scale_
    
    # Plot true function
    ax.plot(x_true, y_true, 'k-', linewidth=1, label='True function')
    
    # Plot training data
    ax.scatter(x_train, y_train, alpha=0.3, s=10, label='Training data')
    
    # Plot model prediction
    ax.plot(test_x, means, 'r-', linewidth=1.5, label='Prediction')

    # Remove the old ±2σ code and replace with a loop for multiple std bands
    n_stds = 4
    for i, k in enumerate(np.linspace(0, n_stds, 4)):
        ax.fill_between(
            test_x,
            means - k * stds,
            means + k * stds,
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if i == 0 else None
        )

    ax.set_xlim(-6, 6)
    ax.set_ylim(-150, 150)
    
    # Calculate MSE
    interp_func = interp1d(test_x, means, bounds_error=False, fill_value="extrapolate")
    pred_at_train = interp_func(x_train)
    mse = np.mean((y_train - pred_at_train)**2)
    
    # Set title and labels
    ax.set_title(f"{model_name}\nMSE: {mse:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    return mse

def main():
    print("Generating synthetic data...")
    embedding_dict, scores_dict, x_values, y_values = generate_cubic_data(n_samples=150, noise_std=3.0, x_range=(-4,4))
    
    # Generate test points for prediction
    test_x = np.linspace(-6, 6, 200)
    
    print("Comparing all models with different hyperparameters...")
    compare_all_hyperparameters(embedding_dict, scores_dict, test_x)


if __name__ == "__main__":
    main()
