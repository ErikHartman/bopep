"""
Example: BoRF (Bopep RFdiffusion + ProteinMPNN + FastRelax) pipeline

Prerequisites:
  - RFdiffusion installation: https://github.com/RosettaCommons/RFdiffusion
  - ProteinMPNN installation: https://github.com/dauparas/ProteinMPNN
  - We recommend using separate conda environments for each tool.
"""

import os
import pandas as pd
from bopep.diffusion.borf import BoRF

# --- Configure paths to external tools ---
# Point these to the Python executables inside each tool's conda environment.
RFD_ENV_PATH = "/path/to/rfdiffusion/env/bin/python"
MPNN_ENV_PATH = "/path/to/proteinmpnn/env/bin/python"

# Root directories of the RFdiffusion and ProteinMPNN repositories.
RFD_PATH = "/path/to/RFdiffusion"
MPNN_PATH = "/path/to/ProteinMPNN"

# Output directory for generated structures and sequences.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "design_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_run():
    # Path to the target protein structure (.pdb)
    pdb_path = "/path/to/target_structure.pdb"

    borf = BoRF(
        output_dir=OUTPUT_DIR,        # Root directory for all generated structures and sequences
        rfdiffusion_path=RFD_PATH,    # Path to the RFdiffusion repository root
        protein_mpnn_path=MPNN_PATH,  # Path to the ProteinMPNN repository root
        pdb_path=pdb_path,            # Target protein PDB file used to condition diffusion (mandatory)
        rfd_env_path=RFD_ENV_PATH,    # Python executable inside the RFdiffusion conda environment
        mpnn_env=MPNN_ENV_PATH,       # Python executable inside the ProteinMPNN conda environment
        mpnn_chains="A",              # Receptor chain(s) ProteinMPNN will design sequences for
    )

    # CSV with a 'peptide' column containing seed sequences.
    samples_csv = pd.read_csv("/path/to/peptide_sequences.csv")

    results = borf.run(
        samples_csv=samples_csv,  # DataFrame (or path) with seed peptides; determines the diffusion targets
        temperature=0.1,          # ProteinMPNN softmax sampling temperature; lower = more conservative
        relax_cycles=4,           # Alternating MPNN + FastRelax passes; more cycles increase diversity
        threads=4,                # Parallel CPU threads for MPNN/FastRelax
        limited_run=1,            # Process only this many samples (0 = all); useful for testing
    )
    print("Pipeline results:", results)


if __name__ == "__main__":
    example_run()