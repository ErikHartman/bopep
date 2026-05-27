# General
ALL_PEPTIDES_PATH = (
    "/home/er8813ha/docking-peptide/data/2_filtered/peptide_data_filtered.csv"
)
ESM_PATH = "/srv/data1/general/esm/esm2_t33_650M_UR50D.pt"

INITIAL_PEPTIDES_DIR = (
    "/home/er8813ha/docking-peptide/src/benchmark/benchmark_peptides.csv"
)

EMBEDDING_DATA_PATH = "/srv/data1/er8813ha/bopep/embedding/cd14/embedding_benchmark_data.pkl"

OUTPUT_RESULTS_DIR = "/home/er8813ha/docking-peptide/results"
OUTPUT_DOCKING_DIR = (
    "/srv/data1/er8813ha/bopep/docked/cd14"
)
TARGET_STRUCTURE_PATH = "/home/er8813ha/docking-peptide/data/target_structures/4glf.pdb"



BINDING_SITE_RESIDUE_INDICES = [22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
                                50, 51, 52, 53, 69, 70, 71, 72,
                                 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                                 88, 89, 90, 104, 105, 106, 107, 108, 109, 110]
DOCKING_KWARGS = {
    "num_models": 5,
    "num_recycles": 10,
    "recycle_early_stop_tolerance": 0.1,
    "amber": True,
    "num_relax": 1,
    "gpu_ids": ["0", "1", "2"],
    "overwrite_results": False,
    "output_dir": OUTPUT_DOCKING_DIR,
}

model_type = "deep_evidential"
network_type = "bigru"

HPO_KWARGS = {"n_trials": 20, "hpo_interval": 25, "n_splits": 3}

N_VALIDATE = 0.2

SEED = 42
BATCH_SIZE = 10

BO_SCHEDULE = [{"acquisition": "standard_deviation", "iterations": 25}, {"acquisition": "expected_improvement", "iterations": 250}, {"acquisition": "mean", "iterations": 10}]
# 500 initial
# 250 standard_deviation
# 2500 expected_improvement
# 100 mean
# Total = 3350 iterations dockings

SCORES_TO_INCLUDE = ["iptm", "interface_dG", "peptide_pae", "rosetta_score", "distance_score", "in_binding_site"]