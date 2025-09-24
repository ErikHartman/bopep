#!/usr/bin/env python3
"""
Simple Multi-Objective Optimization Demo

This is a streamlined example demonstrating the key concepts of multi-objective
optimization with BoPep. It focuses on the essential components:

1. Multi-objective scoring function (mo_objective)
2. Multi-objective surrogate models 
3. ParEGO acquisition functions
4. Basic Pareto front analysis

Perfect for understanding the basics before diving into the full example.
"""

import os
import sys
import numpy as np
import logging
from typing import Dict, Any

# Add the parent directory to the path to import bopep
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bopep.search.optimization import BoPep
from bopep.scoring.scores_to_objective import mo_objective

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_simple_multiobjective_data():
    """Create a small dataset for demonstrating multi-objective optimization."""
    
    # Simple peptide sequences for demonstration
    peptides = [
        "ARNPRYDGL", "GLGAMDYAR", "KLLNEPMRT", "VWFQPKDMT", "IPSAGLQVR",
        "MRTVALSKD", "QDFPKLLNE", "HYGVWSNPR", "LDSTQKFMV", "PAKRGLVDT",
        "FQMTIPSKL", "VYGHRWPMD", "NQVALFKDT", "SKPRMGVLY", "DTQVALGKM",
    ]
    
    # Create simple embeddings (random for demo)
    np.random.seed(42)
    embeddings = {}
    for peptide in peptides:
        embeddings[peptide] = np.random.randn(64).astype(np.float32)
    
    # Create multi-objective scores
    scores = {}
    for i, peptide in enumerate(peptides):
        # Create some realistic multi-objective trade-offs
        base_quality = np.random.normal(0, 1)
        
        # Objective 1: interface_dG (binding energy, lower is better)
        interface_dG = -20 + base_quality * 10 + np.random.normal(0, 3)
        
        # Objective 2: iptm (confidence, higher is better) 
        # Anti-correlated with binding energy for realistic trade-off
        iptm = 0.65 - base_quality * 0.15 + np.random.normal(0, 0.05)
        iptm = np.clip(iptm, 0.2, 0.9)
        
        scores[peptide] = {
            'interface_dG': interface_dG,
            'iptm': iptm,
            'peptide_pae': np.random.uniform(5, 25),
            'distance_score': np.random.uniform(6, 7),
            'rosetta_score': np.random.uniform(-500, -100),
            'in_binding_site': True,  # All in binding site for simplicity
        }
    
    logging.info(f"Created dataset with {len(peptides)} peptides")
    logging.info(f"Interface dG range: [{min(s['interface_dG'] for s in scores.values()):.2f}, {max(s['interface_dG'] for s in scores.values()):.2f}]")
    logging.info(f"iPTM range: [{min(s['iptm'] for s in scores.values()):.3f}, {max(s['iptm'] for s in scores.values()):.3f}]")
    
    return embeddings, scores


def demonstrate_mo_objective():
    """Show how the mo_objective function works."""
    
    logging.info("\\n=== Demonstrating Multi-Objective Function ===")
    
    # Create sample scores
    sample_scores = {
        "PEPTIDE1": {
            'interface_dG': -25.0,
            'iptm': 0.8,
            'in_binding_site': True
        },
        "PEPTIDE2": {
            'interface_dG': -15.0,
            'iptm': 0.9,
            'in_binding_site': True
        },
    }
    
    # Apply mo_objective
    multi_objectives = mo_objective(sample_scores)
    
    logging.info("Sample input scores:")
    for peptide, scores in sample_scores.items():
        logging.info(f"  {peptide}: interface_dG={scores['interface_dG']}, iptm={scores['iptm']}")
    
    logging.info("Multi-objective output:")
    for peptide, objectives in multi_objectives.items():
        logging.info(f"  {peptide}: {objectives}")
    
    return multi_objectives


def simple_multiobjective_optimization():
    """Run a simple multi-objective optimization."""
    
    logging.info("\\n=== Running Simple Multi-Objective Optimization ===")
    
    # Create data
    embeddings, scores = create_simple_multiobjective_data()
    
    logging.info("\\n=== Demonstrating Multi-Objective Surrogate Model ===")
    
    # Create a simple multi-objective surrogate model example
    from bopep.surrogate_model.manager import SurrogateModelManager
    
    # Configure for multi-objective
    surrogate_manager = SurrogateModelManager({
        'model_type': 'mve',
        'network_type': 'mlp',
        'n_objectives': 2,  # This is the key for multi-objective!
    })
    
    # Convert scores to multi-objective format
    mo_scores = mo_objective(scores)
    
    # Optimize hyperparameters and train
    logging.info("Optimizing hyperparameters for multi-objective model...")
    surrogate_manager.optimize_hyperparameters(embeddings, mo_scores, n_trials=3)
    
    logging.info("Initializing and training multi-objective model...")
    surrogate_manager.initialize_model(hyperparams=None, embeddings=embeddings, objectives=mo_scores)
    surrogate_manager.train(embeddings, mo_scores)
    
    # Make predictions
    logging.info("Making multi-objective predictions...")
    predictions = surrogate_manager.predict(embeddings)
    
    logging.info("\\n=== Multi-Objective Predictions ===")
    logging.info("Prediction format for multi-objective models:")
    
    # Show a few predictions
    count = 0
    for peptide, pred_dict in predictions.items():
        if count < 3:
            logging.info(f"\\n{peptide}:")
            for objective, (mean, std) in pred_dict.items():
                logging.info(f"  {objective}: mean={mean:.3f}, std={std:.3f}")
            count += 1
    
    # Demonstrate multi-objective acquisition
    logging.info("\\n=== Multi-Objective Acquisition Functions ===")
    
    from bopep.bayes.acquisition import AcquisitionFunction
    acq_func = AcquisitionFunction()
    
    # Test ParEGO acquisition functions
    try:
        ei_values = acq_func.compute_acquisition(
            predictions=predictions,
            acquisition_function="parego_chebyshev_ei",
            objective_order=['interface_dG', 'iptm'],
            objective_directions={'interface_dG': 'min', 'iptm': 'max'},
            rho=0.05,
            n_mc=64
        )
        
        logging.info("ParEGO Chebyshev EI acquisition values (sample):")
        for i, (peptide, value) in enumerate(list(ei_values.items())[:3]):
            logging.info(f"  {peptide}: {value:.4f}")
            
    except Exception as e:
        logging.warning(f"ParEGO EI failed: {e}")
    
    try:
        ucb_values = acq_func.compute_acquisition(
            predictions=predictions,
            acquisition_function="parego_chebyshev_ucb", 
            objective_order=['interface_dG', 'iptm'],
            objective_directions={'interface_dG': 'min', 'iptm': 'max'},
            rho=0.05,
            kappa=2.0
        )
        
        logging.info("\\nParEGO Chebyshev UCB acquisition values (sample):")
        for i, (peptide, value) in enumerate(list(ucb_values.items())[:3]):
            logging.info(f"  {peptide}: {value:.4f}")
            
    except Exception as e:
        logging.warning(f"ParEGO UCB failed: {e}")
    
    # Demonstrate the new extended schedule format
    logging.info("\\n=== Extended Schedule Format for Multi-Objective ===")
    logging.info("Example schedule with acquisition_kwargs:")
    
    example_schedule = [
        {
            "acquisition": "standard_deviation",
            "iterations": 1
        },
        {
            "acquisition": "parego_chebyshev_ei",
            "iterations": 2,
            "acquisition_kwargs": {
                "objective_order": ['interface_dG', 'iptm'],
                "objective_directions": {'interface_dG': 'min', 'iptm': 'max'},
                "rho": 0.05,
                "n_mc": 128
            }
        },
        {
            "acquisition": "parego_chebyshev_ucb",
            "iterations": 1,
            "acquisition_kwargs": {
                "objective_order": ['interface_dG', 'iptm'],
                "objective_directions": {'interface_dG': 'min', 'iptm': 'max'},
                "rho": 0.05,
                "kappa": 2.0
            }
        }
    ]
    
    logging.info(f"Schedule: {example_schedule}")
    logging.info("This schedule would:")
    logging.info("1. Do 1 iteration of exploration with standard_deviation")
    logging.info("2. Do 2 iterations of multi-objective EI optimization")  
    logging.info("3. Do 1 iteration of multi-objective UCB optimization")
    
    # Clean up
    surrogate_manager.cleanup_model()
    
    return mo_scores
    


def analyze_pareto_front(objectives: Dict[str, Dict[str, float]]):
    """Simple Pareto front analysis."""
    
    logging.info("\\n=== Pareto Front Analysis ===")
    
    # Extract objective values
    peptides = []
    interface_dG_values = []
    iptm_values = []
    
    for peptide, obj_dict in objectives.items():
        if isinstance(obj_dict, dict) and 'interface_dG' in obj_dict and 'iptm' in obj_dict:
            peptides.append(peptide)
            interface_dG_values.append(obj_dict['interface_dG'])
            iptm_values.append(obj_dict['iptm'])
    
    if not peptides:
        logging.warning("No multi-objective data found")
        return
    
    # Find Pareto optimal points
    pareto_indices = []
    for i in range(len(peptides)):
        is_pareto = True
        for j in range(len(peptides)):
            if i != j:
                # Point j dominates point i if j is better in both objectives
                if (interface_dG_values[j] <= interface_dG_values[i] and 
                    iptm_values[j] >= iptm_values[i] and 
                    (interface_dG_values[j] < interface_dG_values[i] or iptm_values[j] > iptm_values[i])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    logging.info(f"Found {len(pareto_indices)} Pareto optimal solutions out of {len(peptides)} total")
    
    if pareto_indices:
        logging.info("\\nPareto optimal peptides:")
        for idx in pareto_indices:
            peptide = peptides[idx]
            dG = interface_dG_values[idx]
            iptm = iptm_values[idx]
            logging.info(f"  {peptide}: interface_dG={dG:.3f}, iptm={iptm:.3f}")
    
    return pareto_indices


def main():
    """Main function demonstrating multi-objective optimization concepts."""
    
    logging.info("=== BoPep Multi-Objective Optimization Demo ===")
    
    # Step 1: Demonstrate the mo_objective function
    demonstrate_mo_objective()
    
    # Step 2: Run simple multi-objective optimization
    try:
        results = simple_multiobjective_optimization()
        
        # Step 3: Analyze the Pareto front
        analyze_pareto_front(results)
        
        logging.info("\\n=== Demo Complete ===")
        logging.info("Key takeaways:")
        logging.info("1. Use mo_objective for multi-objective scoring")
        logging.info("2. Set n_objectives in surrogate_model_kwargs")  
        logging.info("3. Use parego_chebyshev_ei/ucb acquisition functions")
        logging.info("4. Use extended schedule format with acquisition_kwargs for multi-objective")
        logging.info("5. BoPep and BoGA now support acquisition_kwargs in schedule")
        logging.info("6. Analyze results using Pareto front concepts")
        
    except Exception as e:
        logging.error(f"Error in optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()