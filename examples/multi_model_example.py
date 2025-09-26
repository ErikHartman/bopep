"""
Simple example demonstrating the MultiModelWrapper for multi-objective optimization.

This example shows how to use separate models per objective instead of a single 
model with multiple outputs, which can help avoid gradient conflicts.
"""

import numpy as np
from bopep.surrogate_model.manager import SurrogateModelManager

def main():
    
    # Generate synthetic multi-objective data
    np.random.seed(42)
    embeddings = {
        f'peptide_{i}': np.random.randn(64).astype(np.float32)
        for i in range(20)
    }
    
    objectives = {
        peptide: {
            'binding_affinity': np.random.uniform(-10, 0),  # Lower is better
            'stability': np.random.uniform(0, 100),         # Higher is better
            'solubility': np.random.uniform(0, 1)           # Higher is better
        }
        for peptide in embeddings.keys()
    }
    

    # Compare single model vs multi-model approaches
    for approach, use_multi_model in [("Single Model", False), ("Multi-Model", True)]:
        print(f"\n=== {approach} Approach ===")
        
        surrogate_kwargs = {
            'model_type': 'deep_evidential',
            'network_type': 'mlp',
            'multi_model': use_multi_model
        }
        
        # Create manager and optimize hyperparameters
        manager = SurrogateModelManager(surrogate_kwargs)
        manager.optimize_hyperparameters(embeddings, objectives, n_trials=3)
        
        # Initialize and train model
        manager.initialize_model(embeddings=embeddings, objectives=objectives)
        manager.train(embeddings, objectives)
        
        # Make predictions
        predictions = manager.predict(embeddings)
        
        # Show sample prediction
        sample_peptide = list(predictions.keys())[0]
        sample_pred = predictions[sample_peptide]
        print(f"Sample prediction for {sample_peptide}:")
        for obj_name, (mean, std) in sample_pred.items():
            print(f"  {obj_name}: {mean:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()