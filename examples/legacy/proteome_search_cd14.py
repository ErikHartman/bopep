"""
Proteome search example for CD14 target.
Searches human proteome for peptide binders.
"""
import logging
import numpy as np
from bopep import ProteomeSearch, bopep_objective_v1
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    logging.info("Using CUDA")
    torch.cuda.manual_seed(SEED)


def fasta_to_dict(fasta_path):
    """Convert a FASTA file to a dictionary of {header: sequence}."""
    sequences = {}
    with open(fasta_path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = ''.join(seq_lines)
                header = line[1:]  # Remove '>'
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            sequences[header] = ''.join(seq_lines)
    return sequences


if __name__ == "__main__":
    
    # Load proteome
    proteome_path = "/home/er8813ha/docking-peptide/data/proteomes/human_proteome_13_12_2025.fasta"
    logging.info(f"Loading proteome from {proteome_path}")
    proteome = fasta_to_dict(proteome_path)
    logging.info(f"Loaded {len(proteome)} proteins from proteome")
    
    # For testing, you might want to use a subset first
    # proteome = dict(list(proteome.items())[:100])  # Use first 100 proteins
    
    # CD14 binding site residues (0-indexed)
    BINDING_SITE_RESIDUE_INDICES = [
        22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
        50, 51, 52, 53, 69, 70, 71, 72, 73, 74, 75, 
        76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
        90, 104, 105, 106, 107, 108, 109, 110
    ]
    
    # Docking configuration
    DOCKING_KWARGS = {
        "models": ["alphafold"],  # or ["alphafold"] depending on what you want
        "num_models": 5,
        "num_recycles": 10,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["1", "2", "3"],
        "save_raw": False,
        "force": False,
        "output_dir": "/srv/data1/er8813ha/bopep/docked/cd14_proteome_search",
    }
    
    # Scoring configuration - required for bopep_objective_v1
    SCORING_KWARGS = {
        "scores_to_include":["iptm", "interface_dG", "sequence_pae", "rosetta_score", "distance_score", "in_binding_site"],
        "binding_site_residue_indices": BINDING_SITE_RESIDUE_INDICES,
        "binding_site_distance_threshold": 5,
        "required_n_contact_residues": 8,
        "n_jobs": 12,
    }
    
    # Surrogate model configuration
    SURROGATE_MODEL_KWARGS = {
        "model_type": "deep_evidential",
        "network_type": "bigru",
        "hpo_interval": 500, # No HPO
        "n_trials": 100,
        "n_splits": 5,
    }
    
    # Initialize ProteomeSearch
    logging.info("Initializing ProteomeSearch")
    ps = ProteomeSearch(
        proteome=proteome,
        target_structure_path="/home/er8813ha/docking-peptide/data/target_structures/2glf.pdb",
        
        # Peptide sampling parameters
        min_peptide_length=8,
        max_peptide_length=35,
        length_distribution='uniform',
        
        # Initial sampling
        n_init=100,  # Start with 100 random peptides from proteome
        
        # Search parameters
        k_propose=10000,  # Sample 10k peptides from proteome each iteration
        m_select=10,       # Dock top 10 per iteration
        
        # Surrogate model
        surrogate_model_kwargs=SURROGATE_MODEL_KWARGS,
        
        # Objective function
        objective_function=bopep_objective_v1,
        
        # Scoring and docking
        scoring_kwargs=SCORING_KWARGS,
        docker_kwargs=DOCKING_KWARGS,
        
        # Embedding
        embed_method='esm',
        pca_n_components=100,
        embed_batch_size=64,
        embed_device='cuda' if torch.cuda.is_available() else 'cpu',
        
        # Validation
        n_validate=0.2,
        min_validation_samples=10,
        min_training_samples=50,
        
        # Logging
        log_dir="/home/er8813ha/docking-peptide/results/proteome_search_cd14",
    )
    
    # Define optimization schedule
    schedule = [
        {
            'acquisition': 'ei',
            'generations': 100,
        }
    ]
    
    # Run the search
    logging.info("Starting proteome search")
    logging.info(f"Proteome size: {len(proteome)} proteins")
    logging.info(f"Peptide length range: 8-15")
    logging.info(f"Will sample {ps.k_propose} peptides per iteration, dock top {ps.m_select}")
    
    final_objectives = ps.run(schedule)
    
    # Print top results
    logging.info("\n=== Top 10 Results ===")
    sorted_results = sorted(final_objectives.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (seq, obj) in enumerate(sorted_results, 1):
        logging.info(f"{rank}. {seq}: {obj:.4f}")
    
    logging.info(f"Results saved to {ps.log_dir}")
