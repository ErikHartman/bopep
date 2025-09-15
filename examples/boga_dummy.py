#!/usr/bin/env python3
"""
BoGA (Genetic Algorithm) Example with Mocked Docking/Scoring

This example demonstrates:
1. Basic BoGA usage with fast surrogate models and mocked operations
2. Continuation from previous runs using log-based checkpointing

The example uses patching to simulate expensive docking/scoring operations,
making it suitable for testing and development without Docker dependencies.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import random
import numpy as np

# Add the parent directory to the path so we can import bopep
sys.path.insert(0, str(Path(__file__).parent.parent))

from bopep.genetic_algorithm.generate import BoGA


def mock_dock_peptides(self, peptides):
    """Mock docking function that returns fake directories."""
    mock_dirs = []
    for peptide in peptides:
        # Create a fake directory name
        mock_dir = f"/fake/dock/dir/{peptide}"
        mock_dirs.append(mock_dir)
    return mock_dirs


def mock_score_batch(self, scores_to_include, inputs, input_type, **kwargs):
    """Mock scoring function that returns fake scores based on peptide properties."""
    scores = {}
    
    for input_path in inputs:
        # Extract peptide from the fake path
        peptide = input_path.split('/')[-1]
        
        # Generate fake scores based on peptide properties
        # This creates a somewhat realistic landscape for testing
        length_score = 1.0 - abs(len(peptide) - 12) / 20.0  # Prefer ~12 residues
        
        # Favor certain amino acids (simplified binding preference)
        positive_aas = {'K', 'R', 'H'}  # Basic residues
        hydrophobic_aas = {'F', 'W', 'Y', 'L', 'I', 'V'}  # Hydrophobic
        
        positive_count = sum(1 for aa in peptide if aa in positive_aas)
        hydrophobic_count = sum(1 for aa in peptide if aa in hydrophobic_aas)
        
        # Simple scoring function that favors balanced peptides
        balance_score = 1.0 - abs(positive_count - hydrophobic_count) / len(peptide)
        
        # Add some noise for realism
        noise = random.gauss(0, 0.1)
        
        fake_scores = {
            'iptm': min(1.0, max(0.0, 0.6 + length_score * 0.3 + balance_score * 0.1 + noise)),
            'pae': max(1.0, min(30.0, 15.0 - length_score * 5.0 - balance_score * 3.0 + noise * 5)),
            'dG': random.gauss(-5.0, 2.0),  # Binding energy
            'rosetta_score': random.gauss(-50.0, 20.0),  # Rosetta energy
        }
        
        scores[peptide] = fake_scores
    
    return scores


def custom_objective(scores_dict):
    """Custom objective function that maximizes iptm and minimizes pae."""
    objectives = {}
    for peptide, scores in scores_dict.items():
        # Simple multi-objective: maximize iptm, minimize pae
        iptm = scores.get('iptm', 0.5)
        pae = scores.get('pae', 15.0)
        
        # Normalize and combine (higher is better)
        obj = iptm - (pae / 30.0)  # Scale pae to roughly same range as iptm
        objectives[peptide] = obj
    
    return objectives


def run_basic_example():
    """Run a basic BoGA example with mocked functions."""
    print("="*60)
    print("BASIC BoGA EXAMPLE")
    print("="*60)
    
    # Create temporary directory for logs
    temp_dir = tempfile.mkdtemp(prefix="boga_basic_")
    log_dir = os.path.join(temp_dir, "logs")
    
    # Create a dummy target structure file
    dummy_pdb = os.path.join(temp_dir, "target.pdb")
    with open(dummy_pdb, 'w') as f:
        f.write("HEADER    DUMMY PDB FOR TESTING\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    
    try:
        # Mock the expensive operations
        with patch('bopep.docking.docker.Docker.dock_peptides', mock_dock_peptides), \
             patch('bopep.scoring.scorer.Scorer.score_batch', mock_score_batch):
            
            # Create BoGA instance
            boga = BoGA(
                target_structure_path=dummy_pdb,  # Use dummy PDB file
                schedule=[
                    {"acquisition": "expected_improvement", "generations": 3, "m_select": 5, "k_pool": 20},
                    {"acquisition": "upper_confidence_bound", "generations": 2, "m_select": 3, "k_pool": 15},
                ],
                initial_sequences=["ACDEFGHIKLMN", "FYWLIV"],  # Starter sequences
                n_init=8,
                min_sequence_length=8,
                max_sequence_length=15,
                
                # Fast surrogate model settings
                surrogate_model_kwargs={
                    'model_type': 'mc_dropout',
                    'network_type': 'mlp',
                    'hidden_sizes': [64, 32],
                    'dropout_rate': 0.1,
                    'n_epochs': 10,  # Fast training
                },
                
                # Custom objective
                objective_function=custom_objective,
                
                # Scoring configuration
                scoring_kwargs={
                    'scores_to_include': ['iptm', 'pae', 'dG', 'rosetta_score'],
                    'n_jobs': 1,
                },
                
                # Docker configuration (for mocking)
                docker_kwargs={'models': ['alphafold'], 'output_dir': '/tmp/fake_output'},
                
                # Fast embedding
                embed_method='aaindex',  # Faster than ESM
                pca_n_components=20,
                
                # Validation
                n_validate=0.2,  # Use 20% for validation
                
                # Logging
                log_dir=log_dir,
                
                # Hyperparameter tuning
                hpo_interval=3,  # Tune every 3 generations
            )
            
            print(f"Starting basic BoGA run...")
            print(f"Logs will be saved to: {log_dir}")
            
            # Run the optimization
            results = boga.run()
            
            print(f"\n=== BASIC EXAMPLE RESULTS ===")
            print(f"Total sequences evaluated: {len(results)}")
            
            # Show top 5 results
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 sequences:")
            for i, (seq, obj) in enumerate(sorted_results, 1):
                print(f"  {i}. {seq} - Objective: {obj:.4f}")
            
            return log_dir
            
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir)
        raise e


def run_continuation_example(previous_log_dir):
    """Run a continuation example using logs from previous run."""
    print("\n" + "="*60)
    print("CONTINUATION EXAMPLE")
    print("="*60)
    
    # Create new temporary directory for continuation logs
    temp_dir = tempfile.mkdtemp(prefix="boga_continue_")
    continue_log_dir = os.path.join(temp_dir, "logs")
    
    # Create a dummy target structure file
    dummy_pdb = os.path.join(temp_dir, "target.pdb")
    with open(dummy_pdb, 'w') as f:
        f.write("HEADER    DUMMY PDB FOR TESTING\nATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    
    try:
        # Mock the expensive operations
        with patch('bopep.docking.docker.Docker.dock_peptides', mock_dock_peptides), \
             patch('bopep.scoring.scorer.Scorer.score_batch', mock_score_batch):
            
            # Create BoGA instance for continuation
            boga_continue = BoGA(
                target_structure_path=dummy_pdb,  # Use dummy PDB file
                schedule=[
                    {"acquisition": "probability_of_improvement", "generations": 2, "m_select": 4, "k_pool": 15},  # Different acquisition strategy
                    {"acquisition": "expected_improvement", "generations": 2, "m_select": 3, "k_pool": 12},
                ],
                initial_sequences=["DUMMY"],  # Won't be used when continuing
                n_init=5,  # Won't be used when continuing
                
                # Same model configuration for consistency
                surrogate_model_kwargs={
                    'model_type': 'mc_dropout',
                    'network_type': 'mlp',
                    'hidden_sizes': [64, 32],
                    'dropout_rate': 0.1,
                    'n_epochs': 10,
                },
                
                # Custom objective
                objective_function=custom_objective,
                
                # Scoring configuration
                scoring_kwargs={
                    'scores_to_include': ['iptm', 'pae', 'dG', 'rosetta_score'],
                    'n_jobs': 1,
                },
                
                # Docker configuration (for mocking)
                docker_kwargs={'models': ['alphafold'], 'output_dir': '/tmp/fake_output'},
                
                # Fast embedding
                embed_method='aaindex',
                pca_n_components=20,
                
                # Validation
                n_validate=0.2,
                
                # Continuation settings
                continue_from_logs=previous_log_dir,  # Load from previous run
                log_dir=continue_log_dir,  # New log directory
                
                # Hyperparameter tuning
                hpo_interval=2,
            )
            
            print(f"Starting continuation run...")
            print(f"Loading from: {previous_log_dir}")
            print(f"New logs will be saved to: {continue_log_dir}")
            
            # Run the continued optimization
            results = boga_continue.run()
            
            print(f"\n=== CONTINUATION RESULTS ===")
            print(f"Total sequences evaluated: {len(results)}")
            
            # Show top 5 results
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 sequences after continuation:")
            for i, (seq, obj) in enumerate(sorted_results, 1):
                print(f"  {i}. {seq} - Objective: {obj:.4f}")
            
            return continue_log_dir
            
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir)
        raise e


def main():
    """Run all examples."""
    print("BoGA Examples with Mocked Docking/Scoring")
    print("This demonstrates the genetic algorithm functionality without expensive operations.")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 1. Basic example
        basic_log_dir = run_basic_example()
        
        # 2. Continuation example
        continue_log_dir = run_continuation_example(basic_log_dir)
        
        print(f"\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Basic run logs: {basic_log_dir}")
        print(f"Continuation logs: {continue_log_dir}")
        print("\nNote: Temporary directories will be cleaned up automatically.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise
    
    finally:
        # Clean up temporary directories
        try:
            if 'basic_log_dir' in locals():
                shutil.rmtree(Path(basic_log_dir).parent)
            if 'continue_log_dir' in locals():
                shutil.rmtree(Path(continue_log_dir).parent)
        except:
            pass


if __name__ == "__main__":
    main()
