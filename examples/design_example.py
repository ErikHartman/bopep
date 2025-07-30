import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from bopep.design.borf import Borf

# Adjust these paths according to your environment and installation

RFD_ENV_PATH = "/srv/data1/general/RFdiffusion/env/rf_env/bin/python"
MPNN_ENV_PATH = "/srv/data1/general/proteinMPNN/mpnn_env/bin/python"
RFD_PATH = "/srv/data1/general/RFdiffusion"
MPNN_PATH = "/srv/data1/general/ProteinMPNN" 

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "design_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_minimal_complete_pipeline():

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

    samples_csv = borf.create_sample_data(os.path.join(OUTPUT_DIR, "peptide_samples.csv"))
    results = borf.run(
        samples_csv=samples_csv,
        temperature=0.1,
        relax_cycles=1,
        threads=1,
        limited_run=1 
    )
    print("Pipeline results:", results)

if __name__ == "__main__":
    run_minimal_complete_pipeline()