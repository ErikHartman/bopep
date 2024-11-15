import os
import glob
import re
import json
import zipfile
from io import StringIO

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_peptide(
    peptide,
    target_structure,
    binding_site_residue_indices,
    proximity_threshold,
    agreeing_models,
    output_dir,
):
    """
    Process and score a single peptide.

    Parameters:
    - peptide: Peptide to score.
    - docking_params: Dictionary containing docking parameters.
    - binding_site_residue_indices: List of residue indices for receptor binding site.

    Returns:
    - peptide: The peptide string (for identification).
    - scores: Calculated or loaded scores for the peptide.
    """
    target_name = os.path.basename(target_structure).replace(
        ".pdb", ""
    )
    output_subdir = os.path.join(
        output_dir, f"{target_name}_{peptide}"
    )

    results_file = os.path.join(output_subdir, "results.json")

    # Check if results.json exists and has all required scores
    if os.path.exists(results_file):
        logging.info(f"Loading scores from {results_file} for peptide: {peptide}")
        with open(results_file, "r") as f:
            json_contents = json.load(f)
            required_scores = [
                "iptm_score",
                "interface_sasa",
                "interface_dG",
                "rosetta_score",
                "interface_delta_hbond_unsat",
                "packstat",
                "is_proximate",
            ]

            if all(score in json_contents for score in required_scores):
                return peptide, json_contents  # Return the peptide and loaded scores

    logging.info(f"Calculating scores for peptide: {peptide}")

    # Calculate the scores if not available or incomplete
    iptm_score = get_ipTM_from_zip(output_subdir)

    # Check if the peptide is proximate in multiple models
    if binding_site_residue_indices: 
        is_proximate = evaluate_binding_site_proximity_multiple_models(
            output_subdir,
            binding_site_residue_indices,
            threshold=proximity_threshold,
            required_matches=agreeing_models + 1,  # +1 since top model has to match
        )
    else:
        is_proximate = True

    # Get PDB content from the highest-ranked model
    pdb_content = get_pdb_content_from_zip(output_subdir, rank_num=1, relaxed=True)

    # Calculate interface scores
    (
        interface_sasa,
        interface_dG,
        rosetta_score,
        interface_delta_hbond_unsat,
        packstat,
    ) = compute_interface_scores(pdb_content)

    # Store the scores
    scores = {
        "iptm_score": iptm_score,
        "interface_sasa": interface_sasa,
        "interface_dG": interface_dG,
        "rosetta_score": rosetta_score,
        "interface_delta_hbond_unsat": interface_delta_hbond_unsat,
        "packstat": packstat,
        "is_proximate": is_proximate,
    }

    # Save the results to results.json for future use
    logging.info(f"Saving results to {results_file} for peptide: {peptide}")
    os.makedirs(output_subdir, exist_ok=True)  # Ensure the directory exists
    with open(results_file, "w") as f:
        json.dump(scores, f)

    return peptide, scores


def extract_scores(
    peptides,
    target_structure,
    binding_site_residue_indices,
    proximity_threshold,
    agreeing_models,
):
    logging.info(f"Processing {len(peptides)} peptides...")
    scores = {}
    for peptide in peptides:
        peptide_name, score = process_peptide(
            peptide,
            target_structure,
            binding_site_residue_indices,
            proximity_threshold,
            agreeing_models,
            output_dir = "/content/output"
        )
        scores[peptide_name] = score
    logging.info(f"All scores have been processed.")
    return scores


def is_peptide_within_threshold(
    pdb_content, binding_site_residue_indices, threshold=5.0
):
    """
    Determines if the peptide in the given PDB content is within the threshold distance
    to the receptor's binding site.

    Parameters:
    - pdb_content: PDB content as a string.
    - binding_site_residue_indices: List of residue indices representing the receptor binding site.
    - threshold: Distance threshold (in Ångstroms) to consider proximity.

    Returns:
    - True if within threshold, False otherwise.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("docked", StringIO(pdb_content))
    except Exception as e:
        print(f"Error parsing PDB content: {e}")
        return False

    try:
        model = structure[0]
        receptor_chain = model["A"]
        peptide_chain = model["B"]

        receptor_binding_site_atoms = [
            atom
            for residue in receptor_chain
            if residue.id[1] in binding_site_residue_indices
            for atom in residue.get_atoms()
        ]

        if not receptor_binding_site_atoms:
            print(f"No atoms found in specified receptor binding site residues.")
            return False

        peptide_atoms = list(peptide_chain.get_atoms())
        if not peptide_atoms:
            print(f"No atoms found in peptide chain.")
            return False

        # Find atoms within distance threshold
        receptor_coords = np.array([atom.coord for atom in receptor_binding_site_atoms])
        peptide_coords = np.array([atom.coord for atom in peptide_atoms])

        tree = cKDTree(peptide_coords)
        distances, _ = tree.query(receptor_coords, distance_upper_bound=threshold)
        return np.any(distances != float("inf"))

    except KeyError:
        print(f"Chains A or B not found in structure.")
        return False


def evaluate_binding_site_proximity(
    output_subdir, binding_site_residue_indices, threshold=5.0, rank_num=1
):
    pdb_content = get_pdb_content_from_zip(output_subdir, rank_num=rank_num)
    if not pdb_content:
        return False

    return is_peptide_within_threshold(
        pdb_content, binding_site_residue_indices, threshold
    )


def evaluate_binding_site_proximity_multiple_models(
    output_subdir, binding_site_residue_indices, threshold=5.0, required_matches=3
):
    """
    Evaluates if the docked peptide is within a given proximity to the receptor binding site
    across multiple models (ranks). It checks if at least `required_matches` of the models
    have the peptide close to the binding site.
    """
    matches_within_threshold = 0

    for rank_num in range(1, 6):  #  5 models

        pdb_content = get_pdb_content_from_zip(output_subdir, rank_num=rank_num)
        if not pdb_content:
            continue

        peptide_within_threshold = is_peptide_within_threshold(
            pdb_content, binding_site_residue_indices, threshold
        )

        if (
            rank_num == 1 and peptide_within_threshold == False
        ):  # the top model must be within threshold
            return False

        if peptide_within_threshold:
            matches_within_threshold += 1

        if matches_within_threshold >= required_matches:
            return True

    return matches_within_threshold >= required_matches


def get_ipTM_from_zip(zip_dir):
    """
    Extracts the ipTM score from a zipped docking result.

    Parameters:
    - zip_dir: Directory containing the zipped docking result.

    Returns:
    - ipTM score as a float, or None if not found.
    """
    if not os.path.isdir(zip_dir):
        logging.error(f"Directory {zip_dir} does not exist.")
        return None

    zip_pattern = os.path.join(zip_dir, "*.result.zip")
    zip_files = glob.glob(zip_pattern)

    if not zip_files:
        logging.error(f"No result zip file found in {zip_pattern}")
        return None

    zip_file_path = zip_files[0]
    json_pattern = re.compile(r".*_scores_rank_001_.*\.json")

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            json_files = [f for f in zip_ref.namelist() if json_pattern.search(f)]
            if not json_files:
                logging.error(f"JSON file not found in {zip_pattern}")
                return None

            json_content = zip_ref.read(json_files[0]).decode("utf-8")
            return json.loads(json_content).get("iptm")

    except zipfile.BadZipFile:
        logging.error(f"Cannot open {zip_file_path}")
        return None


def get_pdb_content_from_zip(zip_dir, rank_num=1, relaxed=False):
    """
    Retrieves the PDB content from the zipped docking results.

    Parameters:
    - zip_dir: Directory containing the zipped docking result.

    Returns:
    - PDB content as a string or None if not found.
    """
    zip_pattern = os.path.join(zip_dir, "*.result.zip")
    zip_files = glob.glob(zip_pattern)

    if not zip_files:
        print(f"Zip file not found in {zip_dir}.")
        return None

    zip_file_path = zip_files[0]
    if relaxed:
        pattern = rf"relaxed_rank_00{rank_num}_.*\.pdb"
    else:
        pattern = rf"unrelaxed_rank_00{rank_num}_.*\.pdb"

    pdb_pattern = re.compile(pattern)

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            pdb_files = [f for f in zip_ref.namelist() if pdb_pattern.search(f)]
            if not pdb_files:
                print(f"PDB file not found in zip for rank {rank_num}.")
                return None

            return zip_ref.read(pdb_files[0]).decode("utf-8")

    except zipfile.BadZipFile:
        print(f"Cannot open zip file for {zip_dir}.")
        return None


def compute_interface_scores(pdb_content, receptor_chain_id="A", peptide_chain_id="B"):
    # Create a Pose from the PDB content
    pose = Pose()
    pose_from_pdbstring(pose, pdb_content)

    # Prepare the score function
    scorefxn = pyrosetta.get_fa_scorefxn()  # Full-atom scoring function

    # Score the pose before analysis
    rosetta_score = scorefxn(pose)

    # Define the interface between receptor and peptide chains
    interface = receptor_chain_id + "_" + peptide_chain_id

    # Create an InterfaceAnalyzerMover
    ia = InterfaceAnalyzerMover(interface)
    ia.set_compute_packstat(True)
    ia.apply(pose)
    interface_sasa = ia.get_interface_delta_sasa()
    interface_dG = ia.get_interface_dG()
    interface_delta_hbond_unsat = ia.get_interface_delta_hbond_unsat()
    packstat = ia.get_interface_packstat()

    return (
        interface_sasa,
        interface_dG,
        rosetta_score,
        interface_delta_hbond_unsat,
        packstat,
    )
