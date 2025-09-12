#!/usr/bin/env python3

import os
from bopep.genetic_algorithm.generate import BoGA
from bopep.scoring.scores_to_objective import benchmark_objective

def main():
    # File paths
    target_structure_path = "./data/ply1.pdb"
    
    # Verify target file exists
    if not os.path.exists(target_structure_path):
        raise FileNotFoundError(f"Target structure not found: {target_structure_path}")
    
    # Starting sequence
    starting_sequence = "ARNPRYDGLGAMDY"
    
    # Binding site residues (370-465)
    binding_site_residues = list(range(370, 466))  # 466 to include 465
    
    # Define acquisition schedule
    schedule = [
        {
            'acquisition': 'expected_improvement',
            'generations': 5,
            'm_select': 5,
            'k_pool': 10_000
        },
        {
            'acquisition': 'upper_confidence_bound', 
            'generations': 5,
            'm_select': 5,
            'k_pool': 10_000
        }
    ]
    
    # BoGA configuration
    boga = BoGA(
        # Target and sequence parameters
        target_structure_path=target_structure_path,
        schedule=schedule,    # New schedule parameter
        initial_sequences=starting_sequence,  # Single sequence to mutate
        min_sequence_length=8,
        max_sequence_length=25,
        
        # Population and evolution parameters  
        n_init=10,           # Initial population size (reduced for testing)
        mutation_rate=0.05,  # Higher mutation rate for exploration
        
        # Surrogate model configuration (BiGRU + DER)
        surrogate_model_kwargs={
            'model_type': 'deep_evidential',  # Deep Evidential Regression (DER)
            'network_type': 'mlp',          # MLP network (will auto-detect embed_average=True)
            'n_trials': 10,                   # Hyperparameter optimization trials (reduced for testing)
            'n_splits': 3,                    # Cross-validation splits
            'random_state': 42
        },
        
        # Scoring configuration
        scoring_kwargs={
            'scores_to_include': [
                # Required scores for benchmark_objective function
                'rosetta_score',           # Rosetta energy score
                'interface_dG',            # Binding free energy
                'distance_score',          # Distance-based score  
                'iptm',                    # Interface predicted template modeling score
                'peptide_pae',            # Peptide positional average error
                'in_binding_site',         # Binding site occupancy (boolean)
                
                # Additional useful scores
                'interface_sasa',          # Solvent accessible surface area
                'n_contacts',              # Number of contacts
                'peptide_plddt',          # Peptide confidence score
            ],
            'binding_site_residue_indices': binding_site_residues,
            'required_n_contact_residues': 5,  # Number of contacts as requested
            'binding_site_distance_threshold': 5.0,
            'n_jobs': 12  # Parallel scoring jobs
        },
        
        # Docker configuration for docking (not used in dummy mode)
        docker_kwargs={
            'models': ['alphafold'],  # Use AlphaFold for docking
            'output_dir': './examples/boga_output_docking',
            'num_models': 2,            # Number of docking models to generate
            'num_recycles': 1,          # Number of recycles in AlphaFold
        },
        
        # Objective function - use benchmark objective
        objective_function=benchmark_objective,
        objective_function_kwargs={},  # benchmark_objective doesn't need additional kwargs
        
        # Embedding configuration
        embed_method='esm',           # ESM protein language model
        # embed_average will be auto-detected: False for BiGRU, True for MLP
        embed_batch_size=32,          # Batch size for embedding
        pca_n_components=10,          # PCA components to retain (less than n_init)

        # Other parameters
        hpo_interval=5,               # Hyperparameter optimization every 10 generations
        random_seed=42,
        
        # Logging configuration
        log_dir="./examples/boga_logs",
        
        # Testing configuration - use dummy scoring for quick testing
        use_dummy_scoring=False,
    )
    
    print("="*60)
    print("BoGA Peptide Binder Discovery")
    print("="*60)
    print(f"Target structure: {target_structure_path}")
    print(f"Starting sequence: {starting_sequence}")
    print(f"Binding site residues: {min(binding_site_residues)}-{max(binding_site_residues)}")
    print(f"Required contacts: 5")
    print(f"Surrogate model: MLP + Deep Evidential Regression")
    print(f"Objective function: benchmark_objective")
    print(f"Initial population: {boga.n_init} sequences")
    print(f"Schedule phases: {len(boga.schedule)}")
    for i, phase in enumerate(boga.schedule, 1):
        print(f"  Phase {i}: {phase['acquisition']} for {phase['generations']} generations (m_select={phase['m_select']}, k_pool={phase['k_pool']})")
    print(f"Dummy scoring mode: {boga.use_dummy_scoring}")
    print(f"Logging enabled: {boga.logger is not None}")
    if boga.logger:
        print(f"Log directory: {boga.logger.log_dir}")
    print("="*60)
    
    # Run the genetic algorithm
    print("Starting BoGA optimization...")
    final_results = boga.run()
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Sort by objective value (highest first)
    sorted_results = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 sequences found:")
    for i, (sequence, objective) in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. {sequence:20s} | Objective: {objective:.4f}")
    
    print(f"\nTotal sequences evaluated: {len(final_results)}")
    print(f"Best objective value: {sorted_results[0][1]:.4f}")
    print(f"Best sequence: {sorted_results[0][0]}")
    
    # Save results
    output_file = "./examples/boga_results.txt"
    with open(output_file, 'w') as f:
        f.write("BoGA Results\n")
        f.write("="*50 + "\n")
        f.write(f"Target: {target_structure_path}\n")
        f.write(f"Starting sequence: {starting_sequence}\n")
        f.write(f"Binding site: {min(binding_site_residues)}-{max(binding_site_residues)}\n")
        f.write(f"Model: MLP + Deep Evidential Regression\n")
        f.write(f"Objective: benchmark_objective\n\n")
        
        for i, (sequence, objective) in enumerate(sorted_results, 1):
            f.write(f"{i:3d}. {sequence:25s} | {objective:.6f}\n")
    
    print(f"\nResults saved to: {output_file}")
    if boga.logger:
        print(f"Detailed logs saved to: {boga.logger.log_dir}")
        print("Log files created:")
        print("  - scores.csv: Raw scores for all sequences")
        print("  - objectives.csv: Objective values for all sequences") 
        print("  - model_losses.csv: Surrogate model accuracy metrics")
        print("  - hyperparameters.csv: Hyperparameter optimization history")
    print("BoGA optimization completed!")

if __name__ == "__main__":
    main()
