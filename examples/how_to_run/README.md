# How to Run PeptidomeSearch

This guide explains how to set up and run the PeptidomeSearch framework for peptide optimization using Bayesian optimization.

## Basic Setup

1. Install bopep and its dependencies (see main project README)
2. Prepare your target protein structure (PDB format)
3. Define a list of peptide sequences to optimize (your search space)
4. Configure your optimization parameters

## Example Script

To run PeptidomeSearch, you can use a script similar to this example:

```python
from bopep import PeptidomeSearch, benchmark_objective

# Define peptides to optimize
peptides = ["KLLPNLLEQH", "SLLPNLLEQY", "ALLPNLLEQH", "TLLPNLLEQH"]

# Configure PeptidomeSearch
peptidome_search = PeptidomeSearch(
    surrogate_model_kwargs={"model_type": "deep_evidential", "network_type": "bigru"},
    objective_function=benchmark_objective,
    scoring_kwargs={"scores_to_include": ["iptm", "interface_dG", "peptide_pae", "rosetta_score"]},
    hpo_kwargs={"n_trials": 20},
    docker_kwargs={"num_models": 5, "output_dir": "docking_results"},
    log_dir="optimization_logs",
)

# Run optimization
peptidome_search.optimize(
    peptides=peptides,
    target_structure_path="target_protein.pdb",
    batch_size=4,
    binding_site_residue_indices=[23, 24, 27, 28, 31],
    num_validate=10  # Optional: use validation set
)
```

## Key Parameters

### PeptidomeSearch Initialization

- **surrogate_model_kwargs**: Configure the surrogate model
  - `model_type`: "nn_ensemble", "mc_dropout", "deep_evidential", or "mve"
  - `network_type`: "mlp", "bilstm", or "bigru"
- **objective_function**: Function to convert raw scores to optimization objectives
- **scoring_kwargs**: Configure scoring metrics
  - `scores_to_include`: List of scores to compute. Available scores:
  
- **hpo_kwargs**: Hyperparameter optimization settings
- **docker_kwargs**: Configuration for the docking process
- **log_dir**: Directory to save logs and results

### Optimization Method

- **peptides**: List of peptide sequences to optimize
- **target_structure_path**: Path to target protein structure (PDB)
- **batch_size**: Number of peptides to evaluate per iteration
- **binding_site_residue_indices**: Residue indices of binding site on target
- **num_validate**: Number of samples to use for validation (optional)
- **schedule**: List of optimization phases with acquisition functions
- **embeddings**: Path to pre-computed embeddings (optional)
- **initial_peptides**: Specific peptides to start with (optional)

## Validation

When `num_validate` is specified, PeptidomeSearch will:

1. Split docked peptides into training and validation sets
2. Train the surrogate model on the training set
3. Evaluate on both training and validation sets
4. Report R² and MAE metrics for both sets

This helps assess model generalization performance during optimization.

## Advanced Configuration

See the example_main.py file for a more comprehensive configuration example including:

- Multi-phase optimization schedule
- GPU configuration
- Custom docking parameters
- ESM embeddings

## Outputs

PeptidomeSearch logs detailed information to the specified `log_dir`, including:

- Training and validation metrics
- Model predictions
- Acquisition values
- Best peptides found
