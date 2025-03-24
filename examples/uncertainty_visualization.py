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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cubic_data(n_samples: int = 100, 
                        noise_std: float = 3.0, 
                        x_range: Tuple[float, float] = (-3, 3)) -> Tuple[Dict[str, np.ndarray], Dict[str, float], np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a cubic function with Gaussian noise.
    """
    x_segments = np.linspace(x_range[0], x_range[1], num=10)
    x_values = np.array([])
    for i in range(len(x_segments)-1):
        segment_samples = n_samples // (len(x_segments)-1)
        if i == len(x_segments)-2:  #
            segment_samples = n_samples - len(x_values)
        segment_x = np.random.uniform(x_segments[i], x_segments[i+1], segment_samples)
        x_values = np.append(x_values, segment_x)
    
    y_values = x_values**3 + np.random.normal(0, noise_std, n_samples)
    embedding_dict = {f"point_{i}": np.array([x], dtype=np.float32) for i, x in enumerate(x_values)}
    scores_dict = {f"point_{i}": float(y) for i, y in enumerate(y_values)}
    
    return embedding_dict, scores_dict, x_values, y_values

def compare_all_hyperparameters(embedding_dict, scores_dict, test_x):
    x_train = np.array([embedding_dict[key][0] for key in embedding_dict.keys()])
    y_train = np.array([scores_dict[key] for key in scores_dict.keys()])
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_train_scaled = x_scaler.fit_transform(x_train.reshape(-1, 1)).flatten()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    embedding_dict_scaled = {f"point_{i}": np.array([x], dtype=np.float32) 
                            for i, x in enumerate(x_train_scaled)}
    scores_dict_scaled = {f"point_{i}": float(y) for i, y in enumerate(y_train_scaled)}
    
    test_x_scaled = x_scaler.transform(test_x.reshape(-1, 1)).flatten()
    test_embedding_dict_scaled = {f"test_{i}": np.array([x], dtype=np.float32) 
                                for i, x in enumerate(test_x_scaled)}
    
    common_params = {
        "input_dim": 1,
        "hidden_dims": [32, 32],
    }
    
    mc_dropout_rates = [0.1, 0.3, 0.5, 0.7]
    ensemble_sizes = [1, 5, 10, 20]
    evidential_reg_values = [0, 0.1, 0.5, 1.0]
    mve_reg_values = [0.01, 0.1, 0.3, 0.4]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    x_true = np.linspace(min(test_x), max(test_x), 1000)
    y_true = x_true**3
    
    for i, dropout_rate in enumerate(mc_dropout_rates):
        row, col = i // 2, i % 2
        model_name = f"MC Dropout (rate={dropout_rate})"
        print(f"Training {model_name}...")
        
        model = MonteCarloDropout(
            dropout_rate=dropout_rate,
            mc_samples=20,
            network_type="mlp",
            **common_params
        ).to(device)
        
        try:
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
            axes[row, col].text(0.5, 0.5, f"Error: {type(e).__name__}", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[row, col].transAxes, fontsize=12, color='red')
            axes[row, col].set_title(f"{model_name} (Failed)")
    
    for i, n_networks in enumerate(ensemble_sizes):
        row, col = i // 2, (i % 2) + 2
        model_name = f"NN Ensemble (size={n_networks})"
        print(f"Training {model_name}...")
        
        model = NeuralNetworkEnsemble(
            n_networks=n_networks,
            network_type="mlp",
            **common_params
        ).to(device)
        
        try:
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
    
    for i, reg_value in enumerate(evidential_reg_values):
        row, col = (i // 2) + 2, i % 2
        model_name = f"Evidential (reg={reg_value})"
        print(f"Training {model_name}...")
        
        model = DeepEvidentialRegression(
            network_type="mlp",
            evidential_regularization=reg_value,
            **common_params
        ).to(device)
        
        try:
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
    
    for i, reg_value in enumerate(mve_reg_values):
        row, col = (i // 2) + 2, (i % 2) + 2
        model_name = f"MVE (reg={reg_value})"
        print(f"Training {model_name}...")
        
        model = MVE(
            network_type="mlp",
            mve_regularization=reg_value,
            **common_params
        ).to(device)
        
        try:
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
    
    fig.text(0.25, 0.95, "MC Dropout (varying dropout rate)", ha='center', va='center', fontsize=16)
    fig.text(0.75, 0.95, "Neural Network Ensemble (varying ensemble size)", ha='center', va='center', fontsize=16)
    fig.text(0.25, 0.48, "Deep Evidential Regression (varying regularization)", ha='center', va='center', fontsize=16)
    fig.text(0.75, 0.48, "Mean Variance Estimation (varying regularization)", ha='center', va='center', fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("examples/figures", exist_ok=True)
    plt.savefig("examples/figures/hyperparameter_comparison.png", dpi=300)
    print(f"Figure saved as 'examples/figures/hyperparameter_comparison.png'")
    plt.show()

def train_and_plot_model(model, model_name, embedding_dict_scaled, scores_dict_scaled, 
                         test_embedding_dict_scaled, x_train, y_train, test_x, x_true, y_true,
                         x_scaler, y_scaler, ax, is_evidential=False):
    """
    Helper function to train a model and plot its predictions.
    """
    epochs = 1000
    
    model.fit_dict(
        embedding_dict=embedding_dict_scaled,
        scores_dict=scores_dict_scaled,
        epochs=epochs,
        batch_size=16,
        learning_rate=0.001,
        device=device,
        verbose=False,
    )
    
    predictions_scaled = model.predict_dict(
        embedding_dict=test_embedding_dict_scaled,
        batch_size=32,
        device=device
    )
    
    means_scaled = np.array([predictions_scaled[key][0] for key in test_embedding_dict_scaled.keys()])
    stds_scaled = np.array([predictions_scaled[key][1] for key in test_embedding_dict_scaled.keys()])
    
    means = y_scaler.inverse_transform(means_scaled.reshape(-1, 1)).flatten()
    stds = stds_scaled * y_scaler.scale_

    ax.plot(x_true, y_true, 'k-', linewidth=1, label='True function')
    ax.scatter(x_train, y_train, alpha=0.3, s=10, label='Training data')
    ax.plot(test_x, means, 'r-', linewidth=1.5, label='Prediction')

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

    ax.set_xlim(-7,7)
    ax.set_ylim(-150, 150)
    
    interp_func = interp1d(test_x, means, bounds_error=False, fill_value="extrapolate")
    pred_at_train = interp_func(x_train)
    mse = np.mean((y_train - pred_at_train)**2)
    
    ax.set_title(f"{model_name}\nMSE: {mse:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    
    return mse

def main():
    print("Generating synthetic data...")
    embedding_dict, scores_dict, x_values, y_values = generate_cubic_data(n_samples=150, noise_std=3.0, x_range=(-4,4))
    test_x = np.linspace(-7,7, 200)
    print("Comparing all models with different hyperparameters...")
    compare_all_hyperparameters(embedding_dict, scores_dict, test_x)


if __name__ == "__main__":
    main()
