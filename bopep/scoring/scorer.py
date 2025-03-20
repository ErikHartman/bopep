from bopep.scoring.pep_prot_distance import distance_score_from_pdb
from bopep.scoring.rosetta_scorer import RosettaScorer
from bopep.scoring.iptm import get_ipTM_from_dir
from bopep.scoring.is_peptide_in_binding_site import (
    is_peptide_in_binding_site_pdb_file,
    n_peptides_in_binding_site_colab_dir,
)
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.docking.utils import extract_sequence_from_pdb
import os
import re


class Scorer:

    def __init__(self):
        self.available_scores = [
            "all_rosetta_scores",
            "rosetta_score",
            "interface_sasa",
            "interface_dG",
            "interface_delta_hbond_unsat",
            "packstat",
            "distance_score",
            "iptm",
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
        pass

    def score(
        self,
        scores_to_include: list,
        pdb_file: str = None,
        colab_dir: str = None,
        binding_site_residue_indices: list = None,
        peptide_sequence: str = None,
    ) -> dict:
        """
        Calculate and return selected scores for a peptide.
        
        This function calculates various scores for a peptide based on the selected
        metrics. The available data sources determine which scores can be calculated:
        - With only a peptide sequence: Only peptide property scores are available
        - With a PDB file: Rosetta scores and peptide property scores are available
        - With a colab_dir: All scores including ipTM and binding site metrics are available
        
        Parameters
        ----------
        scores_to_include : list
            List of score names to include in the output. Valid options include:
            
            Rosetta scores (requires PDB file):
            - "all_rosetta_scores": Include all Rosetta-based scores
            - "rosetta_score": Overall energy score (lower is better)
            - "interface_sasa": Buried surface area at the interface (higher means larger interface)
            - "interface_dG": Binding energy (more negative indicates stronger binding)
            - "interface_delta_hbond_unsat": Unsatisfied hydrogen bonds at the interface (lower is better)
            - "packstat": Interface packing quality (higher is better, range 0-1)
            
            Other structural scores (requires PDB file):
            - "distance_score": Distance-based scoring of peptide-protein interactions
            - "in_binding_site": Whether peptide is in the defined binding site (requires binding_site_residue_indices)
            
            ColabFold specific scores (requires colab_dir):
            - "iptm": Interface predicted TM-score from ColabFold
            
            Peptide property scores (requires peptide_sequence or PDB file):
            - "peptide_properties": Include all peptide property scores
            - "molecular_weight": Molecular weight of the peptide
            - "aromaticity": Relative frequency of aromatic amino acids
            - "instability_index": Estimate of peptide stability (>40 indicates instability)
            - "isoelectric_point": pH at which the peptide has no net charge
            - "gravy": Grand average of hydropathy (positive = hydrophobic)
            - "helix_fraction": Predicted fraction of residues in alpha helix
            - "turn_fraction": Predicted fraction of residues in turns
            - "sheet_fraction": Predicted fraction of residues in beta sheets
            - "hydrophobic_aa_percent": Percentage of hydrophobic amino acids
            - "polar_aa_percent": Percentage of polar amino acids
            - "positively_charged_aa_percent": Percentage of positively charged amino acids
            - "negatively_charged_aa_percent": Percentage of negatively charged amino acids
            - "delta_net_charge_frac": Net charge as a fraction of peptide length
            - "uHrel": Relative hydrophobic moment (measure of amphipathicity)
            
        pdb_file : str, optional
            Path to a PDB file for structure-based scores)
        colab_dir : str, optional
            Path to a ColabFold output directory for additional scores
        binding_site_residue_indices : list, optional
            List of residue indices defining the binding site (required for in_binding_site score)
        peptide_sequence : str, optional
            Direct peptide sequence (if no PDB file is available)
            
        Returns
        -------
        dict
            Dictionary with peptide sequence as key and a dictionary of scores as value
            
        Examples
        --------
        >>> scorer = Scorer()
        >>> # Get peptide property scores from sequence
        >>> scores = scorer.score(scores_to_include=["molecular_weight", "gravy"], peptide_sequence="ACDEFGH")
        >>> # Get Rosetta scores from PDB file
        >>> scores = scorer.score(scores_to_include=["rosetta_score", "packstat"], pdb_file="complex.pdb")
        """

        
        for score in scores_to_include:
            if score not in self.available_scores:
                raise ValueError(f"WARNING: {score} is not a valid score")

        scores = {}

        # When only peptide_sequence is provided, we can only calculate peptide properties
        if peptide_sequence and not pdb_file and not colab_dir:
            # Check if any non-peptide-property scores are requested
            peptide_property_scores = [
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
                if score not in peptide_property_scores:
                    raise ValueError(f"WARNING: {score} requires a PDB file or colab_dir")
            
            peptide_properties = PeptideProperties(peptide_sequence=peptide_sequence)
        elif colab_dir and not pdb_file:
            pdb_pattern = re.compile(
                r".*_relaxed_rank_001_.*\.pdb"
            )  # Regex for the top scoring docking result (relaxed)
            pdb_file = os.path.join(
                colab_dir,
                [f for f in os.listdir(colab_dir) if pdb_pattern.search(f)][0],
            )
            peptide_sequence = extract_sequence_from_pdb(pdb_file, chain_id="B")
            peptide_properties = PeptideProperties(pdb_file=pdb_file)
            rosetta_scorer = RosettaScorer(pdb_file)
        elif pdb_file:
            peptide_sequence = extract_sequence_from_pdb(pdb_file, chain_id="B")
            peptide_properties = PeptideProperties(pdb_file=pdb_file)
            rosetta_scorer = RosettaScorer(pdb_file)
        else:
            raise ValueError("Either pdb_file, colab_dir, or peptide_sequence must be provided")

        # Process scores based on what can be calculated
        if pdb_file:  # Rosetta scores need a PDB file
            if "all_rosetta_scores" in scores_to_include:
                rosetta_scores = rosetta_scorer.get_all_metrics()
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
            if "distance_score" in scores_to_include:
                scores["distance_score"] = distance_score_from_pdb(pdb_file)

        # ipTM score needs colab_dir
        if "iptm" in scores_to_include:
            if not colab_dir:
                print("WARNING: ipTM score needs a docking result directory.")
            else:
                scores["iptm"] = get_ipTM_from_dir(colab_dir)

        # Binding site scores need binding_site_residue_indices and a PDB file
        if "in_binding_site" in scores_to_include:
            if not binding_site_residue_indices:
                raise ValueError(
                    "WARNING: binding_site_residue_indices is required for in_binding_site score"
                )
            if colab_dir:
                # If colab_dir, we will get the fraction that are in the binding site ([0 to 1]) and a true false
                top_1_in_binding_site, fraction_in_binding_site = (
                    n_peptides_in_binding_site_colab_dir(
                        colab_dir,
                        binding_site_residue_indices=binding_site_residue_indices,
                    )
                )
                scores["in_binding_site"] = bool(top_1_in_binding_site)  # boolean
                scores["fraction_in_binding_site"] = fraction_in_binding_site  # float

            elif pdb_file:
                # if a single pdb, we will have a true/false if it is in binding site
                scores["in_binding_site"] = bool(is_peptide_in_binding_site_pdb_file(
                    pdb_file, binding_site_residue_indices=binding_site_residue_indices
                ))  # boolean
                
        # Peptide property scores can be calculated with either peptide_sequence or pdb_file
        if "peptide_properties" in scores_to_include:
            peptide_properties_dict = peptide_properties.get_all_properties()
            scores.update(peptide_properties_dict)
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

        return {peptide_sequence: scores}
    
    def print_scores(self):
        for score in self.available_scores:
            print(score)


if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    colab_dir_path = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs_erik/docked_peptides/1ydi_VGWEQLLTTIARTINEVENQILTR"
    scorer = Scorer()
    scores = scorer.score(scores_to_include=["rosetta_score"], pdb_file=pdb_file_path)
    print(f"Rosetta score for {pdb_file_path}: {scores}")

    scores = scorer.score(scores_to_include=["iptm", "rosetta_score", "uHrel"], colab_dir=colab_dir_path)
    print(f"Scores for {colab_dir_path}: {scores}")