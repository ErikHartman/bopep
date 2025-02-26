from bopep.scoring.loss_distance import evobind_loss_from_pdb
from bopep.scoring.loss_rosetta import RosettaScorer
from bopep.scoring.loss_iptm import get_ipTM_from_dir
from bopep.scoring.loss_binding_site import (
    is_peptide_in_binding_site,
    n_peptides_in_binding_site_colab_dir,
)
import os


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
            "rosetta_score",
            "interface_sasa",
            "interface_dG",
            "interface_delta_hbond_unsat",
            "packstat",
            "evobind_loss",
            "iptm_score",
            "in_binding_site",
        ]
        for score in scores_to_include:
            if score not in available_scores:
                print(f"WARNING: {score} is not a valid score")

        scores = {}

        if colab_dir and not pdb_file:
            pdb_file = os.path.join(
                colab_dir, "top_scoring_complex.pdb"
            )  # the top 1 docked pdb. It is not named like this in reality.

        rosetta_scorer = RosettaScorer(pdb_file)

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
        if "evobind_loss" in scores_to_include:
            scores["evobind_loss"] = evobind_loss_from_pdb(pdb_file)
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

        return scores


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    scorer = Scorer()
    scores = scorer.score(scores_to_include=["rosetta_score"], pdb_file=pdb_file_path)
    print(f"Composite bopep loss for {pdb_file_path}: {scores}")
