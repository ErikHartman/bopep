from bopep.scoring.pep_prot_distance import distance_score_from_pdb
from bopep.scoring.rosetta_scorer import RosettaScorer
from bopep.scoring.iptm import get_ipTM_from_dir
from bopep.scoring.is_peptide_in_binding_site import (
    is_peptide_in_binding_site,
    n_peptides_in_binding_site_colab_dir,
)
from bopep.scoring.peptide_properties import PeptideProperties
import os
import re


class Scorer:

    def __init__(self):
        pass

    def score(
        self,
        scores_to_include: list,
        pdb_file: str = None,
        colab_dir: str = None,
        binding_site_residue_indices: list = None,
    ) -> dict:

        available_scores = [
            "all_rosetta_scores",
            "rosetta_score",
            "interface_sasa",
            "interface_dG",
            "interface_delta_hbond_unsat",
            "packstat",
            "distance_score",
            "iptm_score",
            "in_binding_site",
            "peptide_properties",
            "molecular_weight",
            "aromaticity",
            "instability_index",
            "isoelectric_point",
            "gravy",
            "helix_fraction",
            "turn_fraction",
            "sheet_fraction",
            "hydrophobic_aa_percent",
            "polar_aa_percent",
            "positively_charged_aa_percent",
            "negatively_charged_aa_percent",
            "delta_net_charge_frac",
            "uHrel",
        ]
        for score in scores_to_include:
            if score not in available_scores:
                print(f"WARNING: {score} is not a valid score")

        scores = {}

        if colab_dir and not pdb_file:
            pdb_pattern = re.compile(
                r".*_rank_001_.*\.pdb"
            )  # Regex for the top scoring docking result
            pdb_file = os.path.join(
                colab_dir,
                [f for f in os.listdir(colab_dir) if pdb_pattern.search(f)][0],
            )

        rosetta_scorer = RosettaScorer(pdb_file)
        peptide_properties = PeptideProperties(pdb_file)

        if "all_rosetta_scores" in scores_to_include:
            rosetta_scores = rosetta_scorer.get_all_scores()
            scores.update(rosetta_scores)
        if "rosetta_score" in scores_to_include:
            scores["rosetta_score"] = rosetta_scorer.get_rosetta_score()
        if "interface_sasa" in scores_to_include:
            scores["interface_sasa"] = rosetta_scorer.get_interface_sasa()
        if "interface_dG" in scores_to_include:
            scores["interface_dG"] = rosetta_scorer.get_interface_dG()
        if "interface_delta_hbond_unsat" in scores_to_include:
            scores["interface_delta_hbond_unsat"] = (
                rosetta_scorer.get_interface_delta_hbond_unsat()
            )
        if "packstat" in scores_to_include:
            scores["packstat"] = rosetta_scorer.get_packstat()
        if "distance_loss" in scores_to_include:
            scores["distance_loss"] = distance_score_from_pdb(pdb_file)
        if "iptm_score" in scores_to_include:
            if not colab_dir:
                print("WARNING: ipTM score needs a docking result directory.")
            else:
                scores["iptm_score"] = get_ipTM_from_dir(colab_dir)
        if "in_binding_site" in scores_to_include:
            if colab_dir:
                scores["in_binding_site"] = n_peptides_in_binding_site_colab_dir(
                    colab_dir, binding_site_residue_indices=binding_site_residue_indices
                )
            else:
                scores["in_binding_site"] = is_peptide_in_binding_site(
                    pdb_file, binding_site_residue_indices=binding_site_residue_indices
                )
        if "peptide_properties" in scores_to_include:
            peptide_properties = peptide_properties.get_all_properties()
            scores.update(peptide_properties)
        if "molecular_weight" in scores_to_include:
            scores["molecular_weight"] = peptide_properties.get_molecular_weight()
        if "aromaticity" in scores_to_include:
            scores["aromaticity"] = peptide_properties.get_aromaticity()
        if "instability_index" in scores_to_include:
            scores["instability_index"] = peptide_properties.get_instability_index()
        if "isoelectric_point" in scores_to_include:
            scores["isoelectric_point"] = peptide_properties.get_isoelectric_point()
        if "gravy" in scores_to_include:
            scores["gravy"] = peptide_properties.get_gravy()
        if "helix_fraction" in scores_to_include:
            scores["helix_fraction"] = peptide_properties.get_helix_fraction()
        if "turn_fraction" in scores_to_include:
            scores["turn_fraction"] = peptide_properties.get_turn_fraction()
        if "sheet_fraction" in scores_to_include:
            scores["sheet_fraction"] = peptide_properties.get_sheet_fraction()
        if "hydrophobic_aa_percent" in scores_to_include:
            scores["hydrophobic_aa_percent"] = (
                peptide_properties.get_hydrophobic_aa_percent()
            )
        if "polar_aa_percent" in scores_to_include:
            scores["polar_aa_percent"] = peptide_properties.get_polar_aa_percent()
        if "positively_charged_aa_percent" in scores_to_include:
            scores["positively_charged_aa_percent"] = (
                peptide_properties.get_positively_charged_aa_percent()
            )
        if "negatively_charged_aa_percent" in scores_to_include:
            scores["negatively_charged_aa_percent"] = (
                peptide_properties.get_negatively_charged_aa_percent()
            )
        if "delta_net_charge_frac" in scores_to_include:
            scores["delta_net_charge_frac"] = (
                peptide_properties.get_delta_net_charge_frac()
            )
        if "uHrel" in scores_to_include:
            scores["uHrel"] = peptide_properties.get_uHrel()

        return scores


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    scorer = Scorer()
    scores = scorer.score(scores_to_include=["rosetta_score"], pdb_file=pdb_file_path)
    print(f"Rosetta score for {pdb_file_path}: {scores}")
