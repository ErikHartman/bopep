from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import json
import glob
from typing import Optional
from bopep.scoring.pep_prot_distance import distance_score_from_structure
from bopep.scoring.rosetta_scorer import RosettaScorer
from bopep.scoring.is_peptide_in_binding_site import (
    is_peptide_in_binding_site_pdb_file,
    smooth_peptide_binding_site_score,
    get_receptor_contacts
)
from bopep.scoring.model_overlap import align_and_compute_rmsd, compute_intra_model_rmsd
from bopep.scoring.peptide_properties import PeptideProperties

from bopep.scoring.confidence_scores import get_peptide_plddt, get_weighted_peptide_plddt, get_peptide_pae, get_peptide_pde
from bopep.scoring.ipsae import get_ipsae_scores_from_structure_and_pae
from bopep.structure.parser import extract_sequence_from_structure
import os

class Scorer:

    def __init__(self):
        # Core scores that can be calculated for any method
        self.core_docking_scores = [
            "iptm", 
        ]
        
        # Structural scores that require PDB files
        self.structural_scores = [
            "rosetta_score", 
            "interface_sasa", 
            "interface_dG", 
            "interface_delta_hbond_unsat", 
            "packstat", 
            "distance_score",
            "in_binding_site", 
            "in_binding_site_score", 
            "n_contacts",

            "peptide_plddt", 
            "peptide_pae",
            "interface_peptide_plddt",
            "receptor_contacts",
            "ipsae_max",
            "ipsae_min"
        ] 
        self.peptide_property_scores = [
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
        
        self.method_specific_scores = {
            "alphafold": ["template_rmsd"],
            "boltz": [
                "peptide_pde",
                "confidence_score", 
                "complex_plddt", 
                "complex_iplddt",
                "complex_pde", 
                "complex_ipde", 
                "ligand_iptm", 
                "protein_iptm", 
                "chain_0_ptm", 
                "chain_1_ptm",
                "template_rmsd"
            ]
        }

        self.special_scores = [
            "inter_model_rmsd", 
            "template_rmsd",
            "intra_model_rmsd",
            "intra_alphafold_mean_rmsd",
            "intra_boltz_mean_rmsd", 
            "intra_all_mean_rmsd"
        ]
        
        
        all_method_specific = []
        for method, scores in self.method_specific_scores.items():
            all_method_specific.extend(scores)
            all_method_specific.extend([f"{method}_{score}" for score in scores])
            all_method_specific.extend([f"{method}_{score}" for score in self.core_docking_scores])
            all_method_specific.extend([f"{method}_{score}" for score in self.structural_scores])
            
        # Store all possible scores for the property
        self._all_possible_scores = (
            self.core_docking_scores + 
            self.structural_scores +
            self.peptide_property_scores + 
            all_method_specific + self.special_scores
        )
        
        self.supported_methods = ["alphafold", "boltz"]

    def get_available_scores(self, 
                           processed_dir=None, 
                           structure_file=None, 
                           binding_site_residue_indices=None,
                           template_structure=None):
        """
        Get scores available for the given context.
        
        Parameters
        ----------
        processed_dir : str, optional
            Path to processed directory to check for available models
        structure_file : str, optional
            Path to structure file (.pdb/.cif) if using single file input
        binding_site_residue_indices : list, optional
            Binding site residue indices - required for binding site scores
        template_structure : str, optional
            Template PDB file - required for template RMSD scores
            
        Returns
        -------
        list
            List of scores available for the given context
        """
        # Determine what models are available
        has_alphafold = False
        has_boltz = False
        
        if processed_dir:
            has_alphafold = os.path.exists(os.path.join(processed_dir, "alphafold_metrics.json"))
            has_boltz = os.path.exists(os.path.join(processed_dir, "boltz_metrics.json"))
        elif structure_file:
            # Single structure file - assume no method-specific data
            has_alphafold = False
            has_boltz = False
        
        both_models_available = has_alphafold and has_boltz
        
        # Start with all possible scores
        available_scores = []
        
        # Always available: peptide property scores
        available_scores.extend(self.peptide_property_scores)
        
        # Model-dependent scores
        if has_alphafold:
            # AlphaFold-specific scores
            available_scores.extend([f"alphafold_{score}" for score in self.core_docking_scores])
            available_scores.extend([f"alphafold_{score}" for score in self.structural_scores])
            
            # Generic scores only if no conflict with Boltz
            if not both_models_available:
                available_scores.extend(self.core_docking_scores)
                available_scores.extend(self.structural_scores)
        
        if has_boltz:
            # Boltz-specific scores
            boltz_specific = self.method_specific_scores["boltz"]
            available_scores.extend(boltz_specific)
            available_scores.extend([f"boltz_{score}" for score in boltz_specific])
            available_scores.extend([f"boltz_{score}" for score in self.core_docking_scores])
            available_scores.extend([f"boltz_{score}" for score in self.structural_scores])
            
            # Generic scores only if no conflict with AlphaFold
            if not both_models_available:
                available_scores.extend(self.core_docking_scores)
                available_scores.extend(self.structural_scores)
        
        # Special scores
        if both_models_available:
            available_scores.append("inter_model_rmsd")
            # Intra-model RMSD scores are available when we have processed directory with multiple models
            if processed_dir:
                available_scores.append("intra_model_rmsd")  # Returns all three
                available_scores.append("intra_alphafold_mean_rmsd")
                available_scores.append("intra_boltz_mean_rmsd")
                available_scores.append("intra_all_mean_rmsd")
        elif processed_dir:
            # Individual method intra-RMSD scores when only one method available
            if has_alphafold:
                available_scores.append("intra_alphafold_mean_rmsd")
            if has_boltz:
                available_scores.append("intra_boltz_mean_rmsd")
            # Generic intra_model_rmsd when only one method
            available_scores.append("intra_model_rmsd")
        
        # Template RMSD scores - only if template provided
        if template_structure is not None:
            if has_alphafold:
                available_scores.append("alphafold_template_rmsd")
            if has_boltz:
                available_scores.append("boltz_template_rmsd")
            if not both_models_available:
                available_scores.append("template_rmsd")
        
        # Binding site scores - only if binding site indices provided
        if binding_site_residue_indices is not None:
            binding_site_scores = ["in_binding_site", "in_binding_site_score"]
            
            if has_alphafold:
                available_scores.extend([f"alphafold_{score}" for score in binding_site_scores])
            if has_boltz:
                available_scores.extend([f"boltz_{score}" for score in binding_site_scores])
            if not both_models_available:
                available_scores.extend(binding_site_scores)
        
        # Single structure file scores (when no processed_dir)
        if structure_file and not processed_dir:
            generic_structural = ["rosetta_score", "interface_sasa", "interface_dG", 
                                "interface_delta_hbond_unsat", "packstat", "distance_score"]
            available_scores.extend(generic_structural)
            
            if binding_site_residue_indices is not None:
                available_scores.extend(["in_binding_site", "in_binding_site_score"])
            if template_structure is not None:
                available_scores.append("template_rmsd")
        
        return sorted(list(set(available_scores)))

    @property 
    def available_scores(self):
        """
        All possible scores (for backward compatibility).
        For context-aware scores, use get_available_scores() instead.
        """
        return self._all_possible_scores

    def score(
        self,
        scores_to_include: list,
        structure_file: str = None,
        processed_dir: str = None,
        binding_site_residue_indices: list = None,
        peptide_sequence: str = None,
        required_n_contact_residues: Optional[int] = 5,
        binding_site_distance_threshold: Optional[int] = 5.0,
        template_structure: Optional[str] = None,
    ) -> dict:
        """
        Calculate and return selected scores for a peptide.
        
        All score names must be explicit - no auto-resolution.
        Use method-prefixed names like "alphafold_iptm", "boltz_confidence_score" etc.
        """
        # Validate requested scores
        for score in scores_to_include:
            if score not in self.available_scores:
                raise ValueError(f"ERROR: '{score}' is not a valid score. Available scores: {self.available_scores}")

        # Determine what models are available
        has_alphafold = False
        has_boltz = False
        if processed_dir:
            has_alphafold = os.path.exists(os.path.join(processed_dir, "alphafold_metrics.json"))
            has_boltz = os.path.exists(os.path.join(processed_dir, "boltz_metrics.json"))
        
        both_models_available = has_alphafold and has_boltz
        
        # Note: Generic scores now automatically include both method-specific scores when both methods available
        
        # Validate parameter dependencies
        binding_site_scores = [
            "in_binding_site", "in_binding_site_score",
            "alphafold_in_binding_site", "alphafold_in_binding_site_score", 
            "boltz_in_binding_site", "boltz_in_binding_site_score"
        ]
        
        template_rmsd_scores = [
            "template_rmsd", "alphafold_template_rmsd", "boltz_template_rmsd"
        ]
        
        intra_rmsd_scores = [
            "intra_model_rmsd", "intra_alphafold_mean_rmsd", 
            "intra_boltz_mean_rmsd", "intra_all_mean_rmsd"
        ]
        
        for score in scores_to_include:
            if score in binding_site_scores and binding_site_residue_indices is None:
                raise ValueError(
                    f"Score '{score}' requires binding_site_residue_indices parameter to be provided."
                )
            
            if score in template_rmsd_scores and template_structure is None:
                raise ValueError(
                    f"Score '{score}' requires template_structure parameter to be provided."
                )
            
            if score in intra_rmsd_scores and processed_dir is None:
                raise ValueError(
                    f"Score '{score}' requires processed_dir parameter to be provided."
                )
            
            if score == "inter_model_rmsd" and not both_models_available:
                raise ValueError(
                    f"Score 'inter_model_rmsd' requires both AlphaFold and Boltz models to be available."
                )

        scores = {}
        
        # Determine peptide sequence
        if not peptide_sequence:
            if processed_dir:
                # Extract from first available metrics file
                for method in ["alphafold", "boltz"]:
                    method_file = os.path.join(processed_dir, f"{method}_metrics.json")
                    if os.path.exists(method_file):
                        with open(method_file, 'r') as f:
                            data = json.load(f)
                            peptide_sequence = data.get("peptide_sequence")
                            if peptide_sequence:
                                break
            elif structure_file:
                peptide_sequence = extract_sequence_from_structure(structure_file, chain_id="B")
            else:
                raise ValueError("Could not determine peptide sequence from provided inputs")
        
        if not peptide_sequence:
            raise ValueError("Peptide sequence is required")
        
        # Load all data upfront
        alphafold_data = {}
        boltz_data = {}
        alphafold_model_file = None
        boltz_model_file = None
        
        if has_alphafold:
            with open(os.path.join(processed_dir, "alphafold_metrics.json"), 'r') as f:
                alphafold_data = json.load(f)
            # Find AlphaFold model file
            for ext in ["pdb", "cif"]:
                pattern = os.path.join(processed_dir, f"alphafold_model_1.{ext}")
                files = glob.glob(pattern)
                if files:
                    alphafold_model_file = files[0]
                    break
        
        if has_boltz:
            with open(os.path.join(processed_dir, "boltz_metrics.json"), 'r') as f:
                boltz_data = json.load(f)
            # Find Boltz model file
            for ext in ["pdb", "cif"]:
                pattern = os.path.join(processed_dir, f"boltz_model_1.{ext}")
                files = glob.glob(pattern)
                if files:
                    boltz_model_file = files[0]
                    break
        
        # Setup structure file for structural scoring
        target_structure_file = structure_file
        if not target_structure_file:
            if alphafold_model_file:
                target_structure_file = alphafold_model_file
            elif boltz_model_file:
                target_structure_file = boltz_model_file
        
        # Initialize peptide properties
        peptide_properties = PeptideProperties(peptide_sequence=peptide_sequence)
        
        # Check method availability upfront
        alphafold_scores_requested = any(score.startswith("alphafold_") for score in scores_to_include)
        boltz_scores_requested = any(score.startswith("boltz_") for score in scores_to_include)
        
        if alphafold_scores_requested and not has_alphafold:
            raise ValueError("AlphaFold scores requested but no AlphaFold output available")
        
        if boltz_scores_requested and not has_boltz:
            raise ValueError("Boltz scores requested but no Boltz output available")
        
        if "alphafold_iptm" in scores_to_include:
            scores["alphafold_iptm"] = alphafold_data.get("iptm")
        
        if "alphafold_rosetta_score" in scores_to_include:
            rosetta_scorer = RosettaScorer(alphafold_model_file)
            scores["alphafold_rosetta_score"] = rosetta_scorer.get_rosetta_score()
        
        if "alphafold_interface_sasa" in scores_to_include:
            rosetta_scorer = RosettaScorer(alphafold_model_file)
            scores["alphafold_interface_sasa"] = rosetta_scorer.get_interface_sasa()
        
        if "alphafold_interface_dG" in scores_to_include:
            rosetta_scorer = RosettaScorer(alphafold_model_file)
            scores["alphafold_interface_dG"] = rosetta_scorer.get_interface_dG()
        
        if "alphafold_interface_delta_hbond_unsat" in scores_to_include:
            rosetta_scorer = RosettaScorer(alphafold_model_file)
            scores["alphafold_interface_delta_hbond_unsat"] = rosetta_scorer.get_interface_delta_hbond_unsat()
        
        if "alphafold_packstat" in scores_to_include:
            rosetta_scorer = RosettaScorer(alphafold_model_file)
            scores["alphafold_packstat"] = rosetta_scorer.get_packstat()
        
        if "alphafold_distance_score" in scores_to_include:
            scores["alphafold_distance_score"] = distance_score_from_structure(alphafold_model_file)
        
        if "alphafold_receptor_contacts" in scores_to_include:
            scores["alphafold_receptor_contacts"] = get_receptor_contacts(
                alphafold_model_file, "A", "B", binding_site_distance_threshold
            )
        
        if "alphafold_in_binding_site" in scores_to_include:
            n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                alphafold_model_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
            scores["alphafold_in_binding_site"] = in_binding_site
            scores["alphafold_n_contacts"] = n_contacts
        
        if "alphafold_in_binding_site_score" in scores_to_include:
            if binding_site_residue_indices is not None:
                scores["alphafold_in_binding_site_score"] = smooth_peptide_binding_site_score(
                    alphafold_model_file, binding_site_residue_indices, threshold=5.0, alpha=1)
            else:
                scores["alphafold_in_binding_site_score"] = None
        
        if "alphafold_template_rmsd" in scores_to_include:
            if template_structure is not None:
                scores["alphafold_template_rmsd"] = align_and_compute_rmsd(template_structure, alphafold_model_file, peptide_sequence)
            else:
                scores["alphafold_template_rmsd"] = None
        
        # AlphaFold confidence scores
        if "alphafold_peptide_plddt" in scores_to_include:
            peptide_plddt = get_peptide_plddt(alphafold_data.get("plddt", []), alphafold_model_file)
            scores["alphafold_peptide_plddt"] = peptide_plddt
        
        if "alphafold_interface_peptide_plddt" in scores_to_include:
            interface_peptide_plddt = get_weighted_peptide_plddt(alphafold_data.get("plddt", []), alphafold_model_file)
            scores["alphafold_interface_peptide_plddt"] = interface_peptide_plddt

        if "alphafold_peptide_pae" in scores_to_include:
            peptide_pae = get_peptide_pae(alphafold_data.get("pae", []), alphafold_model_file)
            scores["alphafold_peptide_pae"] = peptide_pae

        # AlphaFold IPSAE scores
        if "alphafold_ipsae_max" in scores_to_include or "alphafold_ipsae_min" in scores_to_include:
            pae_data = alphafold_data.get("pae", [])
            if pae_data:
                ipsae_scores = get_ipsae_scores_from_structure_and_pae(alphafold_model_file, pae_data)
                if "alphafold_ipsae_max" in scores_to_include:
                    scores["alphafold_ipsae_max"] = ipsae_scores.get("ipsae_max")
                if "alphafold_ipsae_min" in scores_to_include:
                    scores["alphafold_ipsae_min"] = ipsae_scores.get("ipsae_min")
            else:
                if "alphafold_ipsae_max" in scores_to_include:
                    scores["alphafold_ipsae_max"] = None
                if "alphafold_ipsae_min" in scores_to_include:
                    scores["alphafold_ipsae_min"] = None

        # Boltz docking scores
        if "boltz_iptm" in scores_to_include:
            scores["boltz_iptm"] = boltz_data.get("iptm")
        
        if "boltz_confidence_score" in scores_to_include:
            scores["boltz_confidence_score"] = boltz_data.get("confidence_score")
        
        if "boltz_complex_plddt" in scores_to_include:
            scores["boltz_complex_plddt"] = boltz_data.get("complex_plddt")
        
        if "boltz_complex_iplddt" in scores_to_include:
            scores["boltz_complex_iplddt"] = boltz_data.get("complex_iplddt")
        
        if "boltz_complex_pde" in scores_to_include:
            scores["boltz_complex_pde"] = boltz_data.get("complex_pde")
        
        if "boltz_complex_ipde" in scores_to_include:
            scores["boltz_complex_ipde"] = boltz_data.get("complex_ipde")
        
        if "boltz_ligand_iptm" in scores_to_include:
            scores["boltz_ligand_iptm"] = boltz_data.get("ligand_iptm")
        
        if "boltz_protein_iptm" in scores_to_include:
            scores["boltz_protein_iptm"] = boltz_data.get("protein_iptm")
        
        if "boltz_chain_0_ptm" in scores_to_include:
            chains_ptm = boltz_data.get("chains_ptm", {})
            scores["boltz_chain_0_ptm"] = chains_ptm.get("0")
        
        if "boltz_chain_1_ptm" in scores_to_include:
            chains_ptm = boltz_data.get("chains_ptm", {})
            scores["boltz_chain_1_ptm"] = chains_ptm.get("1")
        
        # Boltz structural scores
        if "boltz_rosetta_score" in scores_to_include:
            rosetta_scorer = RosettaScorer(boltz_model_file)
            scores["boltz_rosetta_score"] = rosetta_scorer.get_rosetta_score()
        
        if "boltz_interface_sasa" in scores_to_include:
            rosetta_scorer = RosettaScorer(boltz_model_file)
            scores["boltz_interface_sasa"] = rosetta_scorer.get_interface_sasa()
        
        if "boltz_interface_dG" in scores_to_include:
            rosetta_scorer = RosettaScorer(boltz_model_file)
            scores["boltz_interface_dG"] = rosetta_scorer.get_interface_dG()
        
        if "boltz_interface_delta_hbond_unsat" in scores_to_include:
            rosetta_scorer = RosettaScorer(boltz_model_file)
            scores["boltz_interface_delta_hbond_unsat"] = rosetta_scorer.get_interface_delta_hbond_unsat()
        
        if "boltz_packstat" in scores_to_include:
            rosetta_scorer = RosettaScorer(boltz_model_file)
            scores["boltz_packstat"] = rosetta_scorer.get_packstat()
        
        if "boltz_distance_score" in scores_to_include:
            scores["boltz_distance_score"] = distance_score_from_structure(boltz_model_file)
        
        if "boltz_receptor_contacts" in scores_to_include:
            scores["boltz_receptor_contacts"] = get_receptor_contacts(
                boltz_model_file, "A", "B", binding_site_distance_threshold
            )
        
        if "boltz_in_binding_site" in scores_to_include:
            n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                boltz_model_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
            scores["boltz_in_binding_site"] = in_binding_site
            scores["boltz_n_contacts"] = n_contacts
        
        if "boltz_in_binding_site_score" in scores_to_include:
            if binding_site_residue_indices is not None:
                scores["boltz_in_binding_site_score"] = smooth_peptide_binding_site_score(
                    boltz_model_file, binding_site_residue_indices, threshold=5.0, alpha=1)
            else:
                scores["boltz_in_binding_site_score"] = None
        
        if "boltz_template_rmsd" in scores_to_include:
            if template_structure is not None:
                scores["boltz_template_rmsd"] = align_and_compute_rmsd(template_structure, boltz_model_file, peptide_sequence)
            else:
                scores["boltz_template_rmsd"] = None
        
        # Boltz confidence scores
        if "boltz_peptide_plddt" in scores_to_include:
            peptide_plddt = get_peptide_plddt(boltz_data.get("plddt", []), boltz_model_file)
            scores["boltz_peptide_plddt"] = peptide_plddt

        if "boltz_interface_peptide_plddt" in scores_to_include:
            interface_peptide_plddt = get_weighted_peptide_plddt(boltz_data.get("plddt", []), boltz_model_file)
            scores["boltz_interface_peptide_plddt"] = interface_peptide_plddt
        
        if "boltz_peptide_pae" in scores_to_include:
            peptide_pae = get_peptide_pae(boltz_data.get("pae_matrix", []), boltz_model_file)
            scores["boltz_peptide_pae"] = peptide_pae

        if "boltz_peptide_pde" in scores_to_include:
            peptide_pde = get_peptide_pde(boltz_data.get("pde_matrix", []), boltz_model_file)
            scores["boltz_peptide_pde"] = peptide_pde

        # Boltz IPSAE scores  
        if "boltz_ipsae_max" in scores_to_include or "boltz_ipsae_min" in scores_to_include:
            pae_data = boltz_data.get("pae_matrix", [])
            if pae_data:
                ipsae_scores = get_ipsae_scores_from_structure_and_pae(boltz_model_file, pae_data)
                if "boltz_ipsae_max" in scores_to_include:
                    scores["boltz_ipsae_max"] = ipsae_scores.get("ipsae_max")
                if "boltz_ipsae_min" in scores_to_include:
                    scores["boltz_ipsae_min"] = ipsae_scores.get("ipsae_min")
            else:
                if "boltz_ipsae_max" in scores_to_include:
                    scores["boltz_ipsae_max"] = None
                if "boltz_ipsae_min" in scores_to_include:
                    scores["boltz_ipsae_min"] = None

        # Generic structural scores (now include both method-specific when both available)
        if "rosetta_score" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                rosetta_scorer = RosettaScorer(alphafold_model_file)
                scores["alphafold_rosetta_score"] = rosetta_scorer.get_rosetta_score()
                added = True
            if has_boltz and boltz_model_file:
                rosetta_scorer = RosettaScorer(boltz_model_file)
                scores["boltz_rosetta_score"] = rosetta_scorer.get_rosetta_score()
                added = True
            if not added:
                rosetta_scorer = RosettaScorer(target_structure_file)
                scores["rosetta_score"] = rosetta_scorer.get_rosetta_score()
            elif has_alphafold ^ has_boltz:
                scores["rosetta_score"] = scores.get("alphafold_rosetta_score") or scores.get("boltz_rosetta_score")
        
        if "interface_sasa" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                rosetta_scorer = RosettaScorer(alphafold_model_file)
                scores["alphafold_interface_sasa"] = rosetta_scorer.get_interface_sasa()
                added = True
            if has_boltz and boltz_model_file:
                rosetta_scorer = RosettaScorer(boltz_model_file)
                scores["boltz_interface_sasa"] = rosetta_scorer.get_interface_sasa()
                added = True
            if not added:
                rosetta_scorer = RosettaScorer(target_structure_file)
                scores["interface_sasa"] = rosetta_scorer.get_interface_sasa()
            elif has_alphafold ^ has_boltz:
                scores["interface_sasa"] = scores.get("alphafold_interface_sasa") or scores.get("boltz_interface_sasa")
        
        if "interface_dG" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                rosetta_scorer = RosettaScorer(alphafold_model_file)
                scores["alphafold_interface_dG"] = rosetta_scorer.get_interface_dG()
                added = True
            if has_boltz and boltz_model_file:
                rosetta_scorer = RosettaScorer(boltz_model_file)
                scores["boltz_interface_dG"] = rosetta_scorer.get_interface_dG()
                added = True
            if not added:
                rosetta_scorer = RosettaScorer(target_structure_file)
                scores["interface_dG"] = rosetta_scorer.get_interface_dG()
            elif has_alphafold ^ has_boltz:
                scores["interface_dG"] = scores.get("alphafold_interface_dG") or scores.get("boltz_interface_dG")
        
        if "interface_delta_hbond_unsat" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                rosetta_scorer = RosettaScorer(alphafold_model_file)
                scores["alphafold_interface_delta_hbond_unsat"] = rosetta_scorer.get_interface_delta_hbond_unsat()
                added = True
            if has_boltz and boltz_model_file:
                rosetta_scorer = RosettaScorer(boltz_model_file)
                scores["boltz_interface_delta_hbond_unsat"] = rosetta_scorer.get_interface_delta_hbond_unsat()
                added = True
            if not added:
                rosetta_scorer = RosettaScorer(target_structure_file)
                scores["interface_delta_hbond_unsat"] = rosetta_scorer.get_interface_delta_hbond_unsat()
            elif has_alphafold ^ has_boltz:
                scores["interface_delta_hbond_unsat"] = scores.get("alphafold_interface_delta_hbond_unsat") or scores.get("boltz_interface_delta_hbond_unsat")
        
        if "packstat" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                rosetta_scorer = RosettaScorer(alphafold_model_file)
                scores["alphafold_packstat"] = rosetta_scorer.get_packstat()
                added = True
            if has_boltz and boltz_model_file:
                rosetta_scorer = RosettaScorer(boltz_model_file)
                scores["boltz_packstat"] = rosetta_scorer.get_packstat()
                added = True
            if not added:
                rosetta_scorer = RosettaScorer(target_structure_file)
                scores["packstat"] = rosetta_scorer.get_packstat()
            elif has_alphafold ^ has_boltz:
                scores["packstat"] = scores.get("alphafold_packstat") or scores.get("boltz_packstat")
        
        if "distance_score" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                scores["alphafold_distance_score"] = distance_score_from_structure(alphafold_model_file)
                added = True
            if has_boltz and boltz_model_file:
                scores["boltz_distance_score"] = distance_score_from_structure(boltz_model_file)
                added = True
            if not added:
                scores["distance_score"] = distance_score_from_structure(target_structure_file)
            elif has_alphafold ^ has_boltz:
                scores["distance_score"] = scores.get("alphafold_distance_score") or scores.get("boltz_distance_score")
        
        if "in_binding_site" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                    alphafold_model_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
                scores["alphafold_in_binding_site"] = in_binding_site
                scores["alphafold_n_contacts"] = n_contacts
                added = True
            if has_boltz and boltz_model_file:
                n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                    boltz_model_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
                scores["boltz_in_binding_site"] = in_binding_site
                scores["boltz_n_contacts"] = n_contacts
                added = True
            if not added:
                n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                    target_structure_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
                scores["in_binding_site"] = in_binding_site
                scores["n_contacts"] = n_contacts
            elif has_alphafold ^ has_boltz:
                scores["in_binding_site"] = scores.get("alphafold_in_binding_site") or scores.get("boltz_in_binding_site")
                scores["n_contacts"] = scores.get("alphafold_n_contacts") or scores.get("boltz_n_contacts")
        
        if "in_binding_site_score" in scores_to_include:
            added = False
            if binding_site_residue_indices is not None:
                if has_alphafold and alphafold_model_file:
                    scores["alphafold_in_binding_site_score"] = smooth_peptide_binding_site_score(
                        alphafold_model_file, binding_site_residue_indices, threshold=5.0, alpha=1)
                    added = True
                if has_boltz and boltz_model_file:
                    scores["boltz_in_binding_site_score"] = smooth_peptide_binding_site_score(
                        boltz_model_file, binding_site_residue_indices, threshold=5.0, alpha=1)
                    added = True
                if not added:
                    scores["in_binding_site_score"] = smooth_peptide_binding_site_score(
                        target_structure_file, binding_site_residue_indices, threshold=5.0, alpha=1)
                elif has_alphafold ^ has_boltz:
                    scores["in_binding_site_score"] = scores.get("alphafold_in_binding_site_score") or scores.get("boltz_in_binding_site_score")
            else:
                scores["in_binding_site_score"] = None
        
        if "template_rmsd" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file and template_structure is not None:
                scores["alphafold_template_rmsd"] = align_and_compute_rmsd(template_structure, alphafold_model_file, peptide_sequence)
                added = True
            if has_boltz and boltz_model_file and template_structure is not None:
                scores["boltz_template_rmsd"] = align_and_compute_rmsd(template_structure, boltz_model_file, peptide_sequence)
                added = True
            if not added:
                if template_structure is None:
                    raise ValueError("template_rmsd requires template_structure parameter to be provided")
                else:
                    raise ValueError("template_rmsd requires docking output with model file")
            if has_alphafold ^ has_boltz:  # only one method available -> keep generic alias
                scores["template_rmsd"] = scores.get("alphafold_template_rmsd") or scores.get("boltz_template_rmsd")

        if "receptor_contacts" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                scores["alphafold_receptor_contacts"] = get_receptor_contacts(
                    alphafold_model_file, "A", "B", binding_site_distance_threshold
                )
                added = True
            if has_boltz and boltz_model_file:
                scores["boltz_receptor_contacts"] = get_receptor_contacts(
                    boltz_model_file, "A", "B", binding_site_distance_threshold
                )
                added = True
            if not added:
                scores["receptor_contacts"] = get_receptor_contacts(
                    target_structure_file, "A", "B", binding_site_distance_threshold
                )
            elif has_alphafold ^ has_boltz:
                scores["receptor_contacts"] = scores.get("alphafold_receptor_contacts") or scores.get("boltz_receptor_contacts")

        # Generic confidence scores (previously exclusive; now include both method-specific if both present)
        if "peptide_plddt" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                scores["alphafold_peptide_plddt"] = get_peptide_plddt(alphafold_data.get("plddt", []), alphafold_model_file)
                added = True
            if has_boltz and boltz_model_file:
                scores["boltz_peptide_plddt"] = get_peptide_plddt(boltz_data.get("plddt", []), boltz_model_file)
                added = True
            if not added:
                raise ValueError("peptide_plddt requires docking output with model file")
            if has_alphafold ^ has_boltz:  # only one method available -> keep generic alias
                scores["peptide_plddt"] = scores.get("alphafold_peptide_plddt") or scores.get("boltz_peptide_plddt")
        
        if "interface_peptide_plddt" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                scores["alphafold_interface_peptide_plddt"] = get_weighted_peptide_plddt(alphafold_data.get("plddt", []), alphafold_model_file)
                added = True
            if has_boltz and boltz_model_file:
                scores["boltz_interface_peptide_plddt"] = get_weighted_peptide_plddt(boltz_data.get("plddt", []), boltz_model_file)
                added = True
            if not added:
                raise ValueError("interface_peptide_plddt requires docking output with model file")
            if has_alphafold ^ has_boltz:
                scores["interface_peptide_plddt"] = scores.get("alphafold_interface_peptide_plddt") or scores.get("boltz_interface_peptide_plddt")
        
        if "peptide_pae" in scores_to_include:
            # Normalize PAE key names
            if "pae_matrix" not in alphafold_data and "pae" in alphafold_data:
                alphafold_data["pae_matrix"] = alphafold_data["pae"]
            if "pae_matrix" not in boltz_data and "pae" in boltz_data:
                boltz_data["pae_matrix"] = boltz_data["pae"]
            added = False
            if has_alphafold and alphafold_model_file:
                scores["alphafold_peptide_pae"] = get_peptide_pae(alphafold_data.get("pae_matrix", []), alphafold_model_file)
                added = True
            if has_boltz and boltz_model_file:
                scores["boltz_peptide_pae"] = get_peptide_pae(boltz_data.get("pae_matrix", []), boltz_model_file)
                added = True
            if not added:
                raise ValueError("peptide_pae requires docking output with model file")
            if has_alphafold ^ has_boltz: # XOR (^) means this is only true when one method is available :)
                scores["peptide_pae"] = scores.get("alphafold_peptide_pae") or scores.get("boltz_peptide_pae")
        
        if "peptide_pde" in scores_to_include:
            # Only Boltz currently provides PDE
            if has_boltz and boltz_model_file:
                scores["boltz_peptide_pde"] = get_peptide_pde(boltz_data.get("pde_matrix", []), boltz_model_file)
                scores["peptide_pde"] = scores["boltz_peptide_pde"]  # always expose generic alias since single source
            else:
                raise ValueError("peptide_pde requires Boltz output with model file")
        
        # Generic IPSAE scores
        if "ipsae_max" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                pae_data = alphafold_data.get("pae", [])
                if pae_data:
                    ipsae_scores = get_ipsae_scores_from_structure_and_pae(alphafold_model_file, pae_data)
                    scores["alphafold_ipsae_max"] = ipsae_scores.get("ipsae_max")
                    added = True
            if has_boltz and boltz_model_file:
                pae_data = boltz_data.get("pae_matrix", [])
                if pae_data:
                    ipsae_scores = get_ipsae_scores_from_structure_and_pae(boltz_model_file, pae_data)
                    scores["boltz_ipsae_max"] = ipsae_scores.get("ipsae_max")
                    added = True
            if not added:
                raise ValueError("ipsae_max requires docking output with model file and PAE data")
            if has_alphafold ^ has_boltz:
                scores["ipsae_max"] = scores.get("alphafold_ipsae_max") or scores.get("boltz_ipsae_max")
        
        if "ipsae_min" in scores_to_include:
            added = False
            if has_alphafold and alphafold_model_file:
                pae_data = alphafold_data.get("pae", [])
                if pae_data:
                    ipsae_scores = get_ipsae_scores_from_structure_and_pae(alphafold_model_file, pae_data)
                    scores["alphafold_ipsae_min"] = ipsae_scores.get("ipsae_min")
                    added = True
            if has_boltz and boltz_model_file:
                pae_data = boltz_data.get("pae_matrix", [])
                if pae_data:
                    ipsae_scores = get_ipsae_scores_from_structure_and_pae(boltz_model_file, pae_data)
                    scores["boltz_ipsae_min"] = ipsae_scores.get("ipsae_min")
                    added = True
            if not added:
                raise ValueError("ipsae_min requires docking output with model file and PAE data")
            if has_alphafold ^ has_boltz:
                scores["ipsae_min"] = scores.get("alphafold_ipsae_min") or scores.get("boltz_ipsae_min")
        
        if "intra_model_rmsd" in scores_to_include:
            if not processed_dir:
                raise ValueError("intra_model_rmsd requires processed directory with model files")
            scores.update(compute_intra_model_rmsd(processed_dir, peptide_sequence))
        
        # Individual intra-model RMSD scores
        if any(score in scores_to_include for score in ["intra_alphafold_mean_rmsd", "intra_boltz_mean_rmsd", "intra_all_mean_rmsd"]):
            if not processed_dir:
                raise ValueError("Intra-model RMSD scores require processed directory with model files")
            intra_rmsd_results = compute_intra_model_rmsd(processed_dir, peptide_sequence)
            
            if "intra_alphafold_mean_rmsd" in scores_to_include:
                scores["intra_alphafold_mean_rmsd"] = intra_rmsd_results.get("intra_alphafold_mean_rmsd")
            if "intra_boltz_mean_rmsd" in scores_to_include:
                scores["intra_boltz_mean_rmsd"] = intra_rmsd_results.get("intra_boltz_mean_rmsd")
            if "intra_all_mean_rmsd" in scores_to_include:
                scores["intra_all_mean_rmsd"] = intra_rmsd_results.get("intra_all_mean_rmsd")
        
        # Generic docking scores (iptm) now collect both if present; generic alias only if one
        if "iptm" in scores_to_include:
            added = False
            if has_alphafold:
                scores["alphafold_iptm"] = alphafold_data.get("iptm")
                added = True
            if has_boltz:
                scores["boltz_iptm"] = boltz_data.get("iptm")
                added = True
            if not added:
                raise ValueError("iptm requires docking output")
            if has_alphafold ^ has_boltz:
                scores["iptm"] = scores.get("alphafold_iptm") or scores.get("boltz_iptm")
        
        if "inter_model_rmsd" in scores_to_include:
            if has_boltz and boltz_model_file and has_alphafold and alphafold_model_file:
                scores["inter_model_rmsd"] = align_and_compute_rmsd(boltz_model_file, alphafold_model_file, peptide_sequence)
            else:
                raise ValueError("inter_model_rmsd requires both Boltz and AlphaFold model files")
        
        if "peptide_properties" in scores_to_include:
            scores.update(peptide_properties.get_all_properties())
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
            scores["hydrophobic_aa_percent"] = peptide_properties.get_hydrophobic_aa_percent()
        if "polar_aa_percent" in scores_to_include:
            scores["polar_aa_percent"] = peptide_properties.get_polar_aa_percent()
        if "positively_charged_aa_percent" in scores_to_include:
            scores["positively_charged_aa_percent"] = peptide_properties.get_positively_charged_aa_percent()
        if "negatively_charged_aa_percent" in scores_to_include:
            scores["negatively_charged_aa_percent"] = peptide_properties.get_negatively_charged_aa_percent()
        if "delta_net_charge_frac" in scores_to_include:
            scores["delta_net_charge_frac"] = peptide_properties.get_delta_net_charge_frac()
        if "uHrel" in scores_to_include:
            scores["uHrel"] = peptide_properties.get_uHrel()
        
        return {peptide_sequence: scores}


    def score_batch(
        self,
        scores_to_include: list,
        inputs: list,
        input_type: str = "structure_file",
        binding_site_residue_indices: list = None,
        binding_site_distance_threshold: float = None,
        required_n_contact_residues: Optional[int] = None,
        template_structures: dict = None,
        n_jobs: int = None,
    ) -> dict:
        """
        Score multiple peptides in parallel.

        Parameters
        ----------
        scores_to_include : list
            List of score names to include (same as in score method)
        inputs : list
            List of inputs based on input_type (structure_files, processed_dir, or peptide_sequences)
        input_type : str
            Type of input: "structure_file", "processed_dir", or "peptide_sequence"
        binding_site_residue_indices : list, optional
            List of residue indices defining the binding site
        template_structures : dict, optional
            Dictionary mapping peptide sequences to template PDB file paths for RMSD calculation.
            Only used if "template_rmsd" is in scores_to_include.
        n_jobs : int, optional
            Number of parallel jobs to run. Default is None (use all available cores)

        Returns
        -------
        dict
            Dictionary with results for all inputs, keyed by peptide sequence
        """
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)

        n_jobs = min(n_jobs, len(inputs))
        all_scores = {}

        # Use ProcessPoolExecutor for parallelization
        if n_jobs > 1:
            print(f"Processing {len(inputs)} inputs using {n_jobs} cores...")
            # Create argument tuples for each input
            args_list = [
                (
                    self,
                    scores_to_include,
                    input_val,
                    input_type,
                    binding_site_residue_indices,
                    required_n_contact_residues,
                    binding_site_distance_threshold,
                    template_structures,
                )
                for input_val in inputs
            ]

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Use the static method for parallel processing
                futures = [
                    executor.submit(Scorer._process_single_input, *args)
                    for args in args_list
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_scores.update(result)
                    except Exception as e:
                        print(f"Error processing input: {e}")
        else:
            # Sequential processing
            for input_value in inputs:
                try:
                    result = self._process_single_input(
                        self,
                        scores_to_include,
                        input_value,
                        input_type,
                        binding_site_residue_indices,
                        required_n_contact_residues,
                        binding_site_distance_threshold,
                        template_structures,
                    )
                    all_scores.update(result)
                except Exception as e:
                    print(f"Error processing input {input_value}: {e}")
                    traceback.print_exc()
        print(f"Scored {len(all_scores)} inputs.")
        return all_scores

    @staticmethod
    def _process_single_input(
        scorer, scores_to_include, input_value, input_type, binding_site_residue_indices, required_n_contact_residues, binding_site_distance_threshold, template_structures
    ):
        """
        Process a single input for scoring.

        This is a static method to allow pickling for multiprocessing.
        """
        # Extract template_structure if template_rmsd scoring is requested
        template_structure = None
        if "template_rmsd" in scores_to_include and template_structures:
            try:
                if input_type == "processed_dir":
                    # Extract peptide sequence from metrics.json
                    metrics_file = os.path.join(input_value, "metrics.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        peptide_sequence = metrics.get("peptide_sequence")
                    else:
                        print(f"WARNING: No metrics.json found in {input_value}")
                        peptide_sequence = None
                elif input_type == "structure_file":
                    peptide_sequence = extract_sequence_from_structure(input_value, chain_id="B")
                else:  
                    raise ValueError(
                        f"WARNING: Unsupported input_type: {input_type}. Must be 'structure_file' or 'processed_dir' when using template_rmsd scoring."
                    )
                
                # Lookup template for this peptide
                if peptide_sequence:
                    template_structure = template_structures.get(peptide_sequence)
                    if template_structure and not os.path.exists(template_structure):
                        print(f"WARNING: Template PDB file not found: {template_structure}")
                        template_structure = None
                        
            except Exception as e:
                print(f"WARNING: Error extracting peptide sequence for template lookup: {e}")

        if input_type == "structure_file":
            return scorer.score(
                scores_to_include,
                structure_file=input_value,
                binding_site_residue_indices=binding_site_residue_indices,
                required_n_contact_residues=required_n_contact_residues,
                binding_site_distance_threshold=binding_site_distance_threshold,
                template_structure=template_structure,
            )
        elif input_type == "processed_dir":
            return scorer.score(
                scores_to_include,
                processed_dir=input_value,
                binding_site_residue_indices=binding_site_residue_indices,
                required_n_contact_residues=required_n_contact_residues,
                binding_site_distance_threshold=binding_site_distance_threshold,
                template_structure=template_structure,
            )
        elif input_type == "peptide_sequence":
            return scorer.score(
                scores_to_include, 
                peptide_sequence=input_value,
            )


if __name__ == "__main__":
    structure_file_path = "/srv/data1/er8813ha/bopep/docked/cd14_processed/processed/4glf_NENARQQLERQNK/boltz_model_1.pdb"
    scorer = Scorer()

    # Test dynamic available scores
    print("All possible scores:", len(scorer.available_scores))
    
    # Test with binding site parameters
    binding_site_indices = list(range(1, 70))
    available_with_bs = scorer.get_available_scores(
        structure_file=structure_file_path,
        binding_site_residue_indices=binding_site_indices
    )
    print("Available scores with binding site:", len(available_with_bs))
    
    # Test without binding site parameters
    available_no_bs = scorer.get_available_scores(structure_file=structure_file_path)
    print("Available scores without binding site:", len(available_no_bs))

    # Single score example
    scores = scorer.score(
        scores_to_include=["molecular_weight"], 
        structure_file=structure_file_path
    )
    print(f"Molecular weight score: {scores}")
