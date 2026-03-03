"""
Example: PeptidomeSearch
"""

from bopep import PeptidomeSearch, bopep_objective_v1

def main():

    peptidome = [] # List of sequences

    # --- Binding site residue indices (0-indexed) ---
    # Identify these from your structure using PyMOL, ChimeraX, or similar.
    BINDING_SITE_RESIDUE_INDICES = [
        22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 104, 105, 106, 107, 108, 109, 110,
    ]

    DOCKING_KWARGS = {
        "models": ["alphafold"],               # Docking backend(s): 'alphafold' and/or 'boltz'
        "num_models": 5,                       # AlphaFold: number of model replicates per sequence
        "num_recycles": 10,                    # AlphaFold: structure-refinement recycling iterations
        "recycle_early_stop_tolerance": 0.1,  # AlphaFold: stop recycling early if Cα RMSD change < threshold
        "amber": True,                         # AlphaFold: apply AMBER force-field relaxation
        "num_relax": 1,                        # AlphaFold: number of models to relax (ranked by pLDDT)
        "gpu_ids": ["0"],                      # GPU device IDs (list of strings); one thread per GPU
        "save_raw": False,                     # Keep raw ColabFold/Boltz output files after processing
        "output_dir": "/path/to/docking_output",  # Directory where docking results are written
    }

    SCORING_KWARGS = {
        "scores_to_include": [
            "iptm",            # Interface predicted TM-score from the docking model
            "interface_dG",    # Rosetta interface binding free energy (kcal/mol)
            "sequence_pae",    # Mean predicted aligned error for peptide residues
            "rosetta_score",   # Rosetta full-complex total score
            "distance_score",  # Distance from the peptide centre-of-mass to the binding site centroid
            "in_binding_site", # Boolean: peptide is within binding_site_distance_threshold of the site
        ],
        "binding_site_residue_indices": BINDING_SITE_RESIDUE_INDICES,  # 0-indexed residues defining the binding site
        "binding_site_distance_threshold": 5.0,   # Å cutoff for counting a residue contact with the binding site
        "required_n_contact_residues": 5,          # Minimum contacts needed to mark a peptide as 'in binding site'
        "n_jobs": 4,                               # Parallel workers for scoring
    }

    SURROGATE_MODEL_KWARGS = {
        "model_type": "deep_evidential",  # Uncertainty-aware model: 'deep_evidential', 'nn_ensemble', 'mc_dropout', or 'mve'
        "network_type": "bigru",          # Sequence encoder: 'bigru', 'bilstm', or 'mlp'
        "hpo_interval": 50,               # Re-run Optuna HPO every N iterations
        "n_trials": 50,                   # Optuna trials per HPO round
        "n_splits": 3,                    # Cross-validation folds used during HPO
    }

    ps = PeptidomeSearch(
        surrogate_model_kwargs=SURROGATE_MODEL_KWARGS,  # Surrogate model architecture and HPO configuration
        objective_function=bopep_objective_v1,          # Maps score dict -> scalar; bopep_objective_v1 is the recommended default
        scoring_kwargs=SCORING_KWARGS,                  # Which scores to compute and binding site definition
        docker_kwargs=DOCKING_KWARGS,                   # Docking backend and hardware configuration
        log_dir="./logs/peptidome_search",              # Directory for logs, scores, and checkpoints
    )

    # Optimization schedule: stages run sequentially with surrogate updates between iterations.
    schedule = [
        {
            "acquisition": "standard_deviation",  # Exploration phase: select sequences with highest predicted uncertainty
            "generations": 25,                    # Iterations in this stage
        },
        {
            "acquisition": "expected_improvement", # Exploitation phase: select sequences most likely to beat current best
            "generations": 250,                    # Iterations in this stage
        },
    ]

    ps.run(
        target_structure_path="/path/to/target_structure.cif",  # Path to the target protein structure
        schedule=schedule,                                       # Optimization schedule defined above
        peptidome=peptidome,                                     # List of candidate peptide sequences to search over
        n_init=500,                                              # Sequences randomly selected and docked for initialisation
        k_propose=50000,                                         # Candidate sequences evaluated by the surrogate each iteration
        m_select=10,                                             # Top candidates forwarded to the docking engine each iteration
        min_peptide_length=8,                                    # Minimum peptide length (residues)
        max_peptide_length=35,                                   # Maximum peptide length (residues)
        binding_site_residue_indices=BINDING_SITE_RESIDUE_INDICES,  # Passed to the scorer for binding site evaluation
        embed_method="esm",          # Embedding method: 'esm' (ESM-2 language model) or 'aaindex'
        pca_n_components=100,        # PCA output dimensionality applied after embedding
        n_validate=0.2,              # Fraction (or count) of docked sequences held out for surrogate validation
    )

if __name__ == "__main__":
    main()