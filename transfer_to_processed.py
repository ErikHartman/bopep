#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add bopep to path
sys.path.append(str(Path(__file__).parent))

from bopep.docking.alphafold_docker import AlphaFoldDocker

def main():
    input_dir = "/srv/data1/er8813ha/bopep/docked/cd14" # Set this to directory with raw output
    output_dir = "/srv/data1/er8813ha/bopep/docked/cd14_processed" # Your new dir
    protein_code = "4glf" # your protein code


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    peptides_in_input_dir = [f.split("_")[1] for f in os.listdir(input_dir)]
    afd = AlphaFoldDocker(output_dir=output_dir)
    for peptide in peptides_in_input_dir:
        print("Processing peptide:", peptide, "for protein:", protein_code)
        afd.process_raw_output(f"{input_dir}/{protein_code}_{peptide}", peptide, protein_code)
        #os.rmdir(f"{output_dir}/raw/alphafold")
        #os.rmdir(f"{output_dir}/raw")

if __name__ == "__main__":
    main()
