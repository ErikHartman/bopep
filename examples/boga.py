"""
Example: Bayesian-optimized Genetic Algorithm (BoGA)
"""

import random
from bopep.scoring.scores_to_objective import bopep_objective_v1
from bopep.genetic_algorithm.generate import BoGA


def main():
    # Path to the target protein structure (.cif or .pdb)
    target_structure_path = "/path/to/target_structure.cif"

    # Generate random starting sequences
    random.seed(42)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    starting_sequences = [
        ''.join(random.choices(amino_acids, k=random.randint(8, 25)))
        for _ in range(100)
    ]

    # Binding site residue indices 
    # Identify these from your structure using a tool such as PyMOL or ChimeraX.
    binding_site_residues = [15, 17, 18, 19, 20, 44, 45, 46, 66, 67, 68, 69, 75, 76, 77, 78, 79, 80]

    # Optimization schedule: one or more stages with different acquisition functions.
    # Stages run sequentially; the surrogate model is updated between each generation.
    schedule = [
        {
            'acquisition': 'expected_improvement',  # Acquisition function: 'expected_improvement', 'standard_deviation', or 'greedy'
            'generations': 100,                     # Number of GA generations to run in this stage
            'm_select': 10,                         # Sequences selected for docking each generation
            'k_pool': 5000,                         # Candidate pool size scored by the surrogate before selection
            'acquisition_kwargs': {
                'top_fraction': 50,             # Top-N sequences kept as parents for the next generation
                'selection_method': 'uniform',  # Parent sampling strategy: 'uniform' or 'softmax'
            },
        },
    ]

    boga = BoGA(
        target_structure_path=target_structure_path,  # Path to the target protein structure (.cif or .pdb)
        initial_sequences=starting_sequences,          # Seed sequences for the genetic algorithm
        min_sequence_length=8,                         # Minimum peptide length (residues)
        max_sequence_length=25,                        # Maximum peptide length (residues)
        mode='binding',                                # 'binding' (requires target structure), 'unconditional', or 'sequence'

        n_init=100,          # Number of sequences docked during the initialisation round
        mutation_rate=0.05,  # Per-residue probability of a random mutation each generation

        surrogate_model_kwargs={
            'model_type': 'deep_evidential',  # Uncertainty-aware model: 'deep_evidential', 'nn_ensemble', 'mc_dropout', or 'mve'
            'network_type': 'bigru',          # Sequence encoder: 'bigru', 'bilstm', or 'mlp'
            'n_trials': 50,                   # Optuna HPO trials per HPO round
            'n_splits': 5,                    # Cross-validation folds used during HPO
            'hpo_interval': 200,              # Re-run HPO every N generations (set high to disable)
        },

        scoring_kwargs={
            'scores_to_include': [
                'interface_dG',    # Rosetta interface binding free energy (kcal/mol)
                'iptm',            # Interface predicted TM-score from the docking model
                'in_binding_site', # Boolean: peptide centre-of-mass is within the binding site
                'distance_score',  # Distance from the peptide to the binding site centroid
                'peptide_pae',     # Mean predicted aligned error for peptide residues
                'rosetta_score',   # Rosetta full-complex total score
            ],
            'binding_site_residue_indices': binding_site_residues,  # 0-indexed label_seq_id residues defining the binding site
            'required_n_contact_residues': 5,                        # Minimum receptor residues in contact to count as 'in binding site'
            'n_jobs': 4,                                             # Parallel workers for scoring
        },

        docker_kwargs={
            'models': ['boltz'],                  # Docking backend(s): 'boltz' and/or 'alphafold'
            'output_dir': '/path/to/docking_output',  # Directory where docking results are written
            'num_models': 3,                      # Number of structure models generated per sequence (Boltz: diffusion_samples)
            'num_recycles': 5,                    # Recycling iterations for AlphaFold (ignored by Boltz)
            'diffusion_samples': 3,               # Boltz diffusion samples per sequence
            'recycling_steps': 5,                 # Boltz recycling steps
            'gpu_ids': ['0'],                     # GPU device IDs for docking (list of strings)
        },

        objective_function=bopep_objective_v1,  # Maps score dict -> scalar objective; bopep_objective_v1 is the recommended default
        objective_function_kwargs={},            # Extra keyword arguments forwarded to the objective function

        embed_method='esm',      # Embedding method: 'esm' (ESM-2) or 'aaindex'
        embed_batch_size=64,     # Sequences embedded per forward pass
        pca_n_components=100,    # PCA output dimensionality; must be set when use_pca=True
        n_validate=0.2,          # Fraction (or count) of docked sequences held out for surrogate validation

        log_dir='./logs/boga_run',  # Directory for logs, scores, and checkpoints
    )

    boga.run(schedule=schedule)


if __name__ == "__main__":
    main()
