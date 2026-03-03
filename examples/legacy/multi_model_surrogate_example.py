"""
Simple example demonstrating the MultiModelWrapper for multi-objective optimization.

This example shows how to use separate models per objective instead of a single 
model with multiple outputs, which can help avoid gradient conflicts.
"""

import numpy as np
import matplotlib.pyplot as plt
from bopep.surrogate_model.manager import SurrogateModelManager

def plot_loss_comparison(loss_histories):
    """Plot basic loss curves comparing approaches."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    # Plot 1: Total loss comparison
    ax1 = axes[0]
    ax1.set_title('Training Loss Comparison')
    for approach, history in loss_histories.items():
        if 'train_loss' in history and 'epoch' in history and len(history['train_loss']) > 0:
            epochs = history['epoch'][:len(history['train_loss'])]
            train_loss = history['train_loss']
            ax1.plot(epochs, train_loss, label=f'{approach}', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.legend()
    
    # Plot 2: Per-objective losses for Multi-Model
    ax2 = axes[1]
    ax2.set_title('Per-Objective Losses (Multi-Model)')
    
    # Only plot Multi-Model objective-specific losses
    if 'Multi-Model' in loss_histories:
        multi_model_history = loss_histories['Multi-Model']
        objective_names = [k for k in multi_model_history.keys() 
                          if k not in ['epoch', 'train_loss', 'total_loss', 'val_loss', 'val_total_loss']
                          and not k.startswith('val_')]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, obj_name in enumerate(objective_names):
            if obj_name in multi_model_history and 'epoch' in multi_model_history:
                if len(multi_model_history[obj_name]) > 0:
                    epochs = multi_model_history['epoch'][:len(multi_model_history[obj_name])]
                    obj_losses = multi_model_history[obj_name]
                    color = colors[i % len(colors)]
                    ax2.plot(epochs, obj_losses, label=obj_name, color=color, linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Multi-Model data not available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('examples/figures/multi_model_loss_comparison.png', dpi=300, bbox_inches='tight')


def main():
    
    # Generate synthetic multi-objective data
    embeddings = {
        f'sequence_{i}': np.random.randn(64).astype(np.float32)
        for i in range(20)
    }
    
    objectives = {
        sequence: {
            'binding_affinity': np.random.uniform(-10, 0),  # Lower is better
            'stability': np.random.uniform(0, 100),         # Higher is better
            'solubility': np.random.uniform(0, 1)           # Higher is better
        }
        for sequence in embeddings.keys()
    }
    

    # Compare single model vs multi-model approaches
    loss_histories = {}
    
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
        
        # Ensure we use a reasonable number of epochs for plotting
        if 'epochs' in manager.best_hyperparams:
            manager.best_hyperparams['epochs'] = min(manager.best_hyperparams['epochs'], 50)  # Limit for demo
        
        # Initialize and train model
        manager.initialize_model(embeddings=embeddings, objectives=objectives)
        manager.train(embeddings, objectives)
        
        # Store loss history for plotting
        loss_histories[approach] = manager.model.loss_history
        print(f"Loss history keys: {list(manager.model.loss_history.keys())}")
        print(f"Train loss entries: {len(manager.model.loss_history.get('train_loss', []))}")
        
        # Make predictions
        predictions = manager.predict(embeddings)
        
        # Show sample prediction
        sample_sequence = list(predictions.keys())[0]
        sample_pred = predictions[sample_sequence]
        print(f"Sample prediction for {sample_sequence}:")
        for obj_name, (mean, std) in sample_pred.items():
            print(f"  {obj_name}: {mean:.3f} ± {std:.3f}")
    
    # Plot loss comparison
    plot_loss_comparison(loss_histories)

if __name__ == "__main__":
    main()