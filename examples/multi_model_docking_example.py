import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bopep.docking.docker import Docker
from bopep.scoring.scorer import Scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Run the complete multi-model docking and scoring pipeline."""
    
    peptide_sequence = "NYLSELSEHV"
    target_pdb_path = "./data/4glf.cif" 
    output_dir = "./examples/docking/"


    docker_kwargs = {
        "models": ["boltz"],
        "output_dir": output_dir,
        "num_models": 1,
        "num_recycles": 0,  
        "recycle_early_stop_tolerance": 0.01,
        "amber": True,
        "num_relax": 1,
        "diffusion_samples": 2,  
        "sampling_steps": 200,  
        "output_format": "pdb",
        "step_scale": 1.638,
        "force": True,
        "threshold": 2,
        "gpu_ids": [0],
    }

    docker = Docker(
        docker_kwargs
    )

    docker.set_target_structure(target_pdb_path)

    results = docker.dock_peptides([peptide_sequence])

    print(results)
        
    
    scorer = Scorer()

    scores_to_compute = [
        #"alphafold_iptm",
        "boltz_iptm",
        
        "molecular_weight",
        "gravy",
        "instability_index",
        "aromaticity",
        "isoelectric_point",
        "helix_fraction",
        "turn_fraction",
        "sheet_fraction",
        "hydrophobic_aa_percent",
        "polar_aa_percent",

        "boltz_distance_score",
        #"alphafold_distance_score",
        #"alphafold_rosetta_score",
        "boltz_rosetta_score",
        #"alphafold_interface_sasa",
        "boltz_interface_sasa",
        #"alphafold_interface_dG",
        "boltz_interface_dG",
        #"alphafold_packstat",
        "boltz_packstat",

        #"alphafold_interface_peptide_plddt",
        "boltz_interface_peptide_plddt",
        #"alphafold_peptide_plddt",
        "boltz_peptide_plddt",

        "boltz_in_binding_site",
        #"alphafold_in_binding_site",
       # "inter_model_rmsd"
    ]
    
    


    results = scorer.score(
        scores_to_include=scores_to_compute,
        processed_dir=results[0],
        peptide_sequence=peptide_sequence,
        binding_site_residue_indices=[22, 23, 24, 42, 43, 44, 45, 46, 47, 48, 49, 
            50, 51, 52, 53, 69, 70, 71, 72,
                73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 
                88, 89, 90, 104, 105, 106, 107, 108, 109, 110] 
    )

    print(results)

if __name__ == "__main__":
    main()
