import torch
import numpy as np
from bopep.surrogate_model.base_models import MLPNetwork, BiLSTMNetwork
from bopep.surrogate_model.hparam_opt import OptunaOptimizer, CustomPredictionModel

def generate_sample_data(num_samples=100, embedding_dim=20, seq_length_range=(10, 30), seed=42):
    """
    Generate synthetic data for testing the optimizer.
    
    Args:
        num_samples: Number of samples to generate
        embedding_dim: Dimension of each embedding vector
        seq_length_range: Range of sequence lengths (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        embedding_dict: Dictionary of embeddings {peptide_id: embedding_array}
        scores_dict: Dictionary of scores {peptide_id: score}
    """
    np.random.seed(seed)
    
    embedding_dict = {}
    scores_dict = {}
    
    # Generate random embeddings and scores
    for i in range(num_samples):
        peptide_id = f"peptide_{i}"
        
        # Random sequence length
        seq_length = np.random.randint(seq_length_range[0], seq_length_range[1])
        
        # Random embedding - simulating variable length sequences
        embedding = np.random.randn(seq_length, embedding_dim)
        
        # Create a simple relationship between embedding and score
        score = np.mean(embedding[:, 0]) + 0.5 * np.random.randn()
        
        embedding_dict[peptide_id] = embedding
        scores_dict[peptide_id] = score
    
    return embedding_dict, scores_dict

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sample data
    print("Generating sample data...")
    embedding_dim = 20
    embedding_dict, scores_dict = generate_sample_data(
        num_samples=200, 
        embedding_dim=embedding_dim
    )
    
    # Check a few of the embeddings to confirm variable length
    keys = list(embedding_dict.keys())[:3]
    print("\nSample embedding shapes:")
    for key in keys:
        print(f"{key}: {embedding_dict[key].shape}")
    
    # Example with MLPNetwork
    print("\n--- Optimizing MLPNetwork ---")
    mlp_optimizer = OptunaOptimizer(
        model_class=MLPNetwork,
        embedding_dict=embedding_dict,
        scores_dict=scores_dict,
        n_trials=10,  # Using fewer trials for this example
        test_size=0.2,
        random_state=42,
        early_stopping_rounds=5,
    )
    
    best_mlp_params = mlp_optimizer.optimize()
    print(f"Best MLP parameters: {best_mlp_params}")
    
    # Example with BiLSTMNetwork
    print("\n--- Optimizing BiLSTMNetwork ---")
    bilstm_optimizer = OptunaOptimizer(
        model_class=BiLSTMNetwork,
        embedding_dict=embedding_dict,
        scores_dict=scores_dict,
        n_trials=10,  # Using fewer trials for this example
        test_size=0.2,
        random_state=42,
        early_stopping_rounds=5,
    )
    
    best_bilstm_params = bilstm_optimizer.optimize()
    print(f"Best BiLSTM parameters: {best_bilstm_params}")
    
    # Using the best parameters to create and use a model
    print("\n--- Creating models with best parameters ---")
    
    # For MLPNetwork
    hidden_dims = best_mlp_params.get("hidden_dims", [64, 32])
    mlp_network = MLPNetwork(input_dim=embedding_dim, hidden_dims=hidden_dims)
    mlp_model = CustomPredictionModel(mlp_network)
    print(f"Created MLP model with hidden dimensions: {hidden_dims}")
    
    # For BiLSTMNetwork
    hidden_dim = best_bilstm_params.get("hidden_dim", 64)
    num_layers = best_bilstm_params.get("num_layers", 1)
    bilstm_network = BiLSTMNetwork(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    bilstm_model = CustomPredictionModel(bilstm_network)
    print(f"Created BiLSTM model with hidden dimension: {hidden_dim}, layers: {num_layers}")


if __name__ == "__main__":
    main()
