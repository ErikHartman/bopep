"""
Example: ProteomeSearch
"""
import torch
from bopep import ProteomeSearch, bopep_objective_v1


def main():
    # --- Load proteome ---
    proteome = {} # Load your proteome here as a dictionary {protein_id: sequence}

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

    ps = ProteomeSearch(
        proteome=proteome,                             # Dict mapping protein IDs to sequences
        target_structure_path="/path/to/target_structure.cif",  # Target protein structure (.cif or .pdb)

        min_peptide_length=8,            # Minimum peptide fragment length (residues)
        max_peptide_length=35,           # Maximum peptide fragment length (residues)
        length_distribution="uniform",   # Fragment-length sampling distribution: 'uniform' or 'normal'

        n_init=500,      # Peptides randomly sampled and docked in the initialisation round

        k_propose=50000, # Peptide fragments sampled from the proteome per iteration
        m_select=10,     # Top surrogate-ranked candidates forwarded to docking each iteration

        surrogate_model_kwargs=SURROGATE_MODEL_KWARGS,  # Surrogate model architecture and HPO configuration
        use_pca=True,      # Apply PCA dimensionality reduction to embeddings before training
        pca_fit_data=None, # External sequences used to fit the PCA; if None, fitted on docked sequences

        objective_function=bopep_objective_v1,  # Maps score dict -> scalar; bopep_objective_v1 is the recommended default

        scoring_kwargs=SCORING_KWARGS,   # Which scores to compute and binding site definition
        docker_kwargs=DOCKING_KWARGS,    # Docking backend and hardware configuration

        embed_method="esm",                                       # Embedding method: 'esm' (ESM-2) or 'aaindex'
        pca_n_components=100,                                     # PCA output dimensionality; must match across runs
        embed_batch_size=64,                                      # Sequences embedded per forward pass
        embed_device="cuda" if torch.cuda.is_available() else "cpu",  # Device used for ESM inference

        n_validate=0.2,             # Fraction (or count) of docked sequences held out for surrogate validation
        min_validation_samples=10,  # Skip validation if fewer than this many validation sequences are available
        min_training_samples=50,    # Skip surrogate training if fewer than this many training sequences are available

        log_dir="./logs/proteome_search",  # Directory for logs, scores, and checkpoints
        # continue_from_logs="./logs/proteome_search",  # Uncomment to resume a previous run
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

    ps.run(schedule)

if __name__ == "__main__":
    main()