import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bopep.design.borf import Borf

# Adjust these paths according to your environment and installation. We recommend separating the environments for RFdiffusion and ProteinMPNN.

RFD_ENV_PATH = "/srv/data1/general/RFdiffusion/env/rf_env/bin/python" # Point this to your RFdiffusion environment
MPNN_ENV_PATH = "/srv/data1/general/proteinMPNN/mpnn_env/bin/python" # Point this to your ProteinMPNN environment
RFD_PATH = "/srv/data1/general/RFdiffusion" # Path to RFdiffusion installation
MPNN_PATH = "/srv/data1/general/ProteinMPNN" # Path to ProteinMPNN installation

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "design_output") # If running on a cluster, you may want to set this to a dedicated storage partition
os.makedirs(OUTPUT_DIR, exist_ok=True) # Creates output directory if it does not exist

def example_run():

    output_dir = OUTPUT_DIR 
    rfdiffusion_path = RFD_PATH 
    protein_mpnn_path = MPNN_PATH     
    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "plo1.pdb")

    borf = Borf(
        output_dir=output_dir,
        rfdiffusion_path=rfdiffusion_path,
        protein_mpnn_path=protein_mpnn_path,
        pdb_path=pdb_path,
        rfd_env_path=RFD_ENV_PATH,
        mpnn_env=MPNN_ENV_PATH
    )

    samples_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "peptide_sequences.csv"))

    results = borf.run(
        samples_csv=samples_csv, # Path to your peptide samples CSV file
        temperature=0.1, # Temperature for MPNN sampling, read the MPNN paper for details
        relax_cycles=1, # Number of MPNN + FastRelax cycles to run. We found 4 cycles to be a good trade-off between diversity and time. 
        threads=1, # Set this to the number of threads you want to use. For testing one should be sufficient. 
        limited_run=1 # If set to N, the run will only execute on N of the samples in the CSV file. This is useful for testing.
    )
    print("Pipeline results:", results)

if __name__ == "__main__":
    example_run()