import os
import shutil
import logging
from typing import List, Tuple
from Bio.PDB import PDBParser
from colabfold.batch import run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def dock_peptides(
    peptide_list: List[Tuple[str, str]],
    peptide_results: dict, 
    target_structure: str,
    output_dir: str = "/content/output",
    num_recycles: int = 1,
    num_relax: int = 1,
    num_models: int = 1,
    amber: bool = True,
    recycle_early_stop_tolerance=0.3,
    target_sequence: str = None,
   
) -> None:

    # Process peptides one by one

    for peptide_sequence in peptide_list:
        combined_sequence = f"{target_sequence}:{peptide_sequence}"
        jobname = (
            f"{os.path.basename(target_structure).replace('.pdb', '')}_{peptide_sequence}"
        )
        peptide_output_dir = os.path.join(output_dir, jobname)
        os.makedirs(peptide_output_dir, exist_ok=True)

        # Copy target_structure to peptide_output_dir
        target_structure_copy = os.path.join(
            peptide_output_dir, os.path.basename(target_structure)
        )
        shutil.copy2(target_structure, target_structure_copy)

        # Prepare custom_template_path
        custom_template_path = peptide_output_dir
        queries = [(jobname, combined_sequence)]
        logging.info(f"Docking peptide {peptide_sequence}...")

        result = run(
            queries=queries,
            is_complex=True,
            result_dir=peptide_output_dir,
            use_templates=True,
            custom_template_path=custom_template_path,
            model_type="alphafold2_multimer_v3",
            msa_mode="single_sequence",
            num_recycles=num_recycles,
            recycle_early_stop_tolerance=recycle_early_stop_tolerance,
            num_relax=num_relax,
            num_models=num_models,
            rank_by="iptm",
            use_amber=amber,
            pair_mode="unpaired",
            save_recycles=False,
        )
        peptide_results[peptide_sequence] = result
    return peptide_results


def extract_sequence_from_pdb(pdb_file, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_file)
    aa_dict = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        "SEC": "U",
        "PYL": "O",
    }
    sequence = "".join(
        aa_dict.get(residue.get_resname(), "X")
        for model in structure
        for chain in model
        if chain.id == chain_id
        for residue in chain
        if residue.id[0] == " "
    )
    return sequence
