import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from bopep.design.borf import Borf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "1"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "design_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_minimal_complete_pipeline():

    output_dir = OUTPUT_DIR
    rfdiffusion_path = "/srv/data1/general/RFdiffusion"      # <-- set this to your RFdiffusion install
    protein_mpnn_path = "/srv/data1/general/ProteinMPNN"     # <-- set this to your ProteinMPNN install
    pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "plo1.pdb")

    # Che

    borf = Borf(
        output_dir=output_dir,
        rfdiffusion_path=rfdiffusion_path,
        protein_mpnn_path=protein_mpnn_path,
        pdb_path=pdb_path,
        rfd_env_path="/srv/data1/general/RFdiffusion/env/rf_env/bin/python",
        mpnn_env="/srv/data1/ma7631si/proteinMPNN/mpnn_env/bin/python"
    )

    samples_csv = borf.create_sample_data(os.path.join(OUTPUT_DIR, "peptide_samples.csv"))
    results = borf.run_complete_pipeline(
        samples_csv=samples_csv,
        temperature=0.1,
        relax_cycles=1,
        threads=1,
        limited_run=1
    )
    print("Pipeline results:", results)

if __name__ == "__main__":
    run_minimal_complete_pipeline()