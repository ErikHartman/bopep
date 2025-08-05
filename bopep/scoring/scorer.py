from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import json
import glob
from typing import Optional, List
from bopep.scoring.pep_prot_distance import distance_score_from_pdb
from bopep.scoring.rosetta_scorer import RosettaScorer
from bopep.scoring.is_peptide_in_binding_site import (
    is_peptide_in_binding_site_pdb_file,
    smooth_peptide_binding_site_score,
)
from bopep.scoring.model_overlap import align_and_compute_rmsd
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.scoring.parser import MetricsParser
from bopep.scoring.confidence_scores import calculate_peptide_confidence_scores
from bopep.docking.utils import extract_sequence_from_pdb
import os
import re


class Scorer:

    def __init__(self):
        # Core scores that can be calculated for any method
        self.core_docking_scores = [
            "iptm", "ptm", "plddt", "pae"
        ]
        
        # Structural scores that require PDB files
        self.structural_scores = [
            "rosetta_score", "interface_sasa", "interface_dG", 
            "interface_delta_hbond_unsat", "packstat", "distance_score",
            "in_binding_site", "in_binding_site_score", "template_rmsd",
            "peptide_plddt", "weighted_plddt_overall", "weighted_plddt_residues",
            "peptide_pae", "peptide_pde"
        ]        # Peptide property scores (method-independent)
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
        
        # Method-specific scores (only available for specific methods)
        self.method_specific_scores = {
            "alphafold": [
                "model_count", "best_model", "original_rank", "relaxed"
            ],
            "boltz": [
                "confidence_score", "best_model_id", "complex_plddt", "complex_iplddt",
                "complex_pde", "complex_ipde", "ligand_iptm", "protein_iptm", 
                "has_clash", "fraction_plausible", "chain_0_ptm", "chain_1_ptm"
            ]
        }
        
        # Special scores
        self.special_scores = [
            "all_rosetta_scores",
        ]
        
        # Build comprehensive available scores list
        all_method_specific = []
        for method, scores in self.method_specific_scores.items():
            all_method_specific.extend(scores)
            all_method_specific.extend([f"{method}_{score}" for score in scores])
            all_method_specific.extend([f"{method}_{score}" for score in self.core_docking_scores])
            all_method_specific.extend([f"{method}_{score}" for score in self.structural_scores])
        
        self.available_scores = (
            self.core_docking_scores + 
            self.structural_scores +
            self.peptide_property_scores + 
            all_method_specific +
            self.special_scores
        )
        
        # Supported docking methods (extensible for future methods like ESMFold)
        self.supported_methods = ["alphafold", "boltz"]

    def _get_available_methods(self, processed_dir: str) -> List[str]:
        """
        Get list of methods that have model files in the processed directory.
        """
        available_methods = []
        
        for method in self.supported_methods:
            # Check for method-specific model files
            model_pattern = os.path.join(processed_dir, f"{method}_model_*.pdb")
            if glob.glob(model_pattern):
                available_methods.append(method)
        
        return available_methods
    
    def _get_method_model_file(self, processed_dir: str, method: str, model_num: int = 1) -> Optional[str]:
        """
        Get the path to a specific method's model file.
        
        Args:
            processed_dir: Processed directory path
            method: Method name (alphafold, boltz, etc.)
            model_num: Model number (default 1 for best model)
            
        Returns:
            Path to model file or None if not found
        """
        # Look for PDB files first, then CIF
        for ext in ["pdb", "cif"]:
            pattern = os.path.join(processed_dir, f"{method}_model_{model_num}.{ext}")
            files = glob.glob(pattern)
            if files:
                return files[0]
        
        return None

    def score(
        self,
        scores_to_include: list,
        pdb_file: str = None,
        processed_dir: str = None,
        binding_site_residue_indices: list = None,
        peptide_sequence: str = None,
        required_n_contact_residues: Optional[int] = 5,
        binding_site_distance_threshold: Optional[int] = 5.0,
        template_pdb: Optional[str] = None,
    ) -> dict:
        """
        Calculate and return selected scores for a peptide with smart method detection.

        The scorer automatically detects available methods and applies intelligent naming:
        - Single method: Generic names (iptm, distance_score)  
        - Multiple methods: Method-prefixed names (alphafold_iptm, boltz_iptm)
        - Method-specific scores: Must specify exact method (confidence_score only for Boltz)
        """
        
        # Validate requested scores
        for score in scores_to_include:
            if score not in self.available_scores:
                raise ValueError(f"ERROR: '{score}' is not a valid score. Available scores: {self.available_scores}")

        scores = {}
        peptide_properties = None  # Initialize to avoid UnboundLocalError
        available_methods = []
        parsed_metrics = {}
        
        # Determine peptide sequence and available data sources
        if peptide_sequence and not pdb_file and not processed_dir:
            # Only peptide properties available
            peptide_properties = PeptideProperties(peptide_sequence=peptide_sequence)
            available_methods = []
            
        elif processed_dir:
            # Use parser to get all method-specific metrics
            parser = MetricsParser()
            parsed_metrics = parser.parse_processed_dir(processed_dir)
            available_methods = parser.get_available_methods(processed_dir)
            
            # Extract peptide sequence from parser data or first available metrics file
            if not peptide_sequence:
                for method in available_methods:
                    method_file = os.path.join(processed_dir, f"{method}_metrics.json")
                    if os.path.exists(method_file):
                        with open(method_file, 'r') as f:
                            data = json.load(f)
                            peptide_sequence = data.get("peptide_sequence")
                            if peptide_sequence:
                                break
            
            # Setup for structural scoring (find best model file)
            if not pdb_file:
                for method in available_methods:
                    model_file = self._get_method_model_file(processed_dir, method, 1)
                    if model_file:
                        pdb_file = model_file
                        break
                        
            # Initialize peptide properties
            if pdb_file:
                peptide_properties = PeptideProperties(pdb_file=pdb_file)
            elif peptide_sequence:
                peptide_properties = PeptideProperties(peptide_sequence=peptide_sequence)
                
        elif pdb_file:
            # Single PDB file provided
            peptide_sequence = extract_sequence_from_pdb(pdb_file, chain_id="B")
            peptide_properties = PeptideProperties(pdb_file=pdb_file)
            available_methods = []
            
        else:
            raise ValueError("Either pdb_file, processed_dir, or peptide_sequence must be provided")
        
        # Ensure we have peptide_sequence
        if not peptide_sequence:
            raise ValueError("Could not determine peptide sequence from provided inputs")

        # Smart score resolution
        resolved_scores = self._resolve_score_requests(scores_to_include, available_methods, parsed_metrics)
        
        # Calculate peptide property scores (if peptide_properties available)
        if peptide_properties:
            scores.update(self._calculate_peptide_property_scores(resolved_scores, peptide_properties))
        
        # Calculate structural scores (if PDB available)
        if pdb_file:
            scores.update(self._calculate_structural_scores(
                resolved_scores, pdb_file, processed_dir, available_methods, 
                binding_site_residue_indices, required_n_contact_residues, 
                binding_site_distance_threshold, template_pdb, peptide_sequence
            ))
        
        # Add parsed docking metrics (if processed_dir available)
        if processed_dir and parsed_metrics:
            scores.update(self._extract_docking_scores(resolved_scores, parsed_metrics, available_methods))
        
        # Calculate confidence scores using raw confidence data (if available)
        if processed_dir and pdb_file and parsed_metrics:
            scores.update(self._calculate_confidence_scores(
                resolved_scores, parsed_metrics, pdb_file, available_methods, peptide_sequence
            ))
        
        return {peptide_sequence: scores}

    def score_batch(
        self,
        scores_to_include: list,
        inputs: list,
        input_type: str = "pdb_file",
        binding_site_residue_indices: list = None,
        binding_site_distance_threshold: float = None,
        required_n_contact_residues: Optional[int] = None,
        template_pdbs: dict = None,
        n_jobs: int = None,
    ) -> dict:
        """
        Score multiple peptides in parallel.

        Parameters
        ----------
        scores_to_include : list
            List of score names to include (same as in score method)
        inputs : list
            List of inputs based on input_type (pdb_files, colab_dirs, or peptide_sequences)
        input_type : str
            Type of input: "pdb_file", "processed_dir", or "peptide_sequence"
        binding_site_residue_indices : list, optional
            List of residue indices defining the binding site
        template_pdbs : dict, optional
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
                    template_pdbs,
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
                        template_pdbs,
                    )
                    all_scores.update(result)
                except Exception as e:
                    print(f"Error processing input {input_value}: {e}")
                    traceback.print_exc()
        print(f"Scored {len(all_scores)} inputs.")
        return all_scores

    @staticmethod
    def _process_single_input(
        scorer, scores_to_include, input_value, input_type, binding_site_residue_indices, required_n_contact_residues, binding_site_distance_threshold, template_pdbs
    ):
        """
        Process a single input for scoring.

        This is a static method to allow pickling for multiprocessing.
        """
        # Extract template_pdb if template_rmsd scoring is requested
        template_pdb = None
        if "template_rmsd" in scores_to_include and template_pdbs:
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
                elif input_type == "pdb_file":
                    peptide_sequence = extract_sequence_from_pdb(input_value, chain_id="B")
                else:  
                    raise ValueError(
                        f"WARNING: Unsupported input_type: {input_type}. Must be 'pdb_file' or 'processed_dir' when using template_rmsd scoring."
                    )
                
                # Lookup template for this peptide
                if peptide_sequence:
                    template_pdb = template_pdbs.get(peptide_sequence)
                    if template_pdb and not os.path.exists(template_pdb):
                        print(f"WARNING: Template PDB file not found: {template_pdb}")
                        template_pdb = None
                        
            except Exception as e:
                print(f"WARNING: Error extracting peptide sequence for template lookup: {e}")

        if input_type == "pdb_file":
            return scorer.score(
                scores_to_include,
                pdb_file=input_value,
                binding_site_residue_indices=binding_site_residue_indices,
                required_n_contact_residues=required_n_contact_residues,
                binding_site_distance_threshold=binding_site_distance_threshold,
                template_pdb=template_pdb,
            )
        elif input_type == "processed_dir":
            return scorer.score(
                scores_to_include,
                processed_dir=input_value,
                binding_site_residue_indices=binding_site_residue_indices,
                required_n_contact_residues=required_n_contact_residues,
                binding_site_distance_threshold=binding_site_distance_threshold,
                template_pdb=template_pdb,
            )
        elif input_type == "peptide_sequence":
            return scorer.score(
                scores_to_include, 
                peptide_sequence=input_value,
            )

    def get_available_scores(self):
        return self.available_scores

    def _resolve_score_requests(self, requested_scores: list, available_methods: list, parsed_metrics: dict) -> dict:
        """
        Resolve score requests using smart method detection.
        
        Rules:
        1. Single method available: Generic names work (iptm → alphafold_iptm if only AlphaFold)
        2. Multiple methods available: Must use prefixed names (alphafold_iptm, boltz_confidence_score)
        3. Method-specific scores: Always require exact prefix
        
        Returns dict with resolved score names and their sources.
        """
        resolved = {}
        
        for score in requested_scores:
            # Check if it's already a method-specific score
            if any(score.startswith(f"{method}_") for method in ["alphafold", "boltz"]):
                # Validate the method is available
                method = score.split("_")[0]
                if method not in available_methods and available_methods:  # Only check if we have methods
                    raise ValueError(f"Score '{score}' requires {method} method, but only {available_methods} are available")
                resolved[score] = score
                continue
            
            # Check if it's a core docking score that can be auto-resolved
            if score in self.core_docking_scores:
                if len(available_methods) == 0:
                    # No docking methods available
                    raise ValueError(f"Score '{score}' requires docking output (processed_dir), but none available")
                elif len(available_methods) == 1:
                    # Single method - use generic name, resolve to method-specific internally
                    method = available_methods[0]
                    resolved[score] = f"{method}_{score}"
                else:
                    # Multiple methods - require user to specify which one
                    method_scores = [f"{method}_{score}" for method in available_methods if f"{method}_{score}" in parsed_metrics]
                    if method_scores:
                        raise ValueError(f"Multiple methods available for '{score}'. Please specify: {method_scores}")
                    else:
                        raise ValueError(f"Score '{score}' not available in any method: {available_methods}")
            
            # Check if it's a structural score that can be auto-resolved
            elif score in self.structural_scores:
                if len(available_methods) == 0:
                    # No methods available - can still calculate if pdb_file is provided directly
                    resolved[score] = score
                elif len(available_methods) == 1:
                    # Single method - use generic name for structural scores
                    resolved[score] = score
                else:
                    # Multiple methods - require user to specify which one for structural scores
                    method_scores = [f"{method}_{score}" for method in available_methods]
                    raise ValueError(f"Multiple methods available for structural score '{score}'. Please specify: {method_scores}")
            
            else:
                # Non-docking, non-structural score - use as-is
                resolved[score] = score
                
        return resolved

    def _calculate_peptide_property_scores(self, resolved_scores: dict, peptide_properties) -> dict:
        """Calculate peptide property scores."""
        scores = {}
        
        property_score_map = {
            "peptide_properties": "get_all_properties",
            "molecular_weight": "get_molecular_weight", 
            "aromaticity": "get_aromaticity",
            "instability_index": "get_instability_index",
            "isoelectric_point": "get_isoelectric_point",
            "gravy": "get_gravy",
            "helix_fraction": "get_helix_fraction",
            "turn_fraction": "get_turn_fraction", 
            "sheet_fraction": "get_sheet_fraction",
            "hydrophobic_aa_percent": "get_hydrophobic_aa_percent",
            "polar_aa_percent": "get_polar_aa_percent",
            "positively_charged_aa_percent": "get_positively_charged_aa_percent",
            "negatively_charged_aa_percent": "get_negatively_charged_aa_percent",
            "delta_net_charge_frac": "get_delta_net_charge_frac",
            "uHrel": "get_uHrel"
        }
        
        for original_score, resolved_score in resolved_scores.items():
            if resolved_score in property_score_map:
                method_name = property_score_map[resolved_score]
                result = getattr(peptide_properties, method_name)()
                if resolved_score == "peptide_properties":
                    scores.update(result)  # get_all_properties returns dict
                else:
                    scores[original_score] = result
                    
        return scores

    def _calculate_structural_scores(self, resolved_scores: dict, pdb_file: str, processed_dir: str, available_methods: list,
                                  binding_site_residue_indices: list, required_n_contact_residues: int,
                                  binding_site_distance_threshold: float, template_pdb: str, peptide_sequence: str) -> dict:
        """Calculate structural scores that require PDB files."""
        scores = {}
        
        # Determine which scores need to be calculated and for which methods
        method_specific_structural_scores = {}  # {method: [scores_to_calculate]}
        non_method_scores = {}  # {score_name: resolved_score}
        
        for original_score, resolved_score in resolved_scores.items():
            # Check if this is a method-specific structural score
            if any(resolved_score.startswith(f"{method}_") for method in ["alphafold", "boltz"]):
                method = resolved_score.split("_")[0]
                base_score = "_".join(resolved_score.split("_")[1:])
                if base_score in self.structural_scores:
                    if method not in method_specific_structural_scores:
                        method_specific_structural_scores[method] = []
                    method_specific_structural_scores[method].append((original_score, base_score))
            # Check if this is a structural score (non-method-specific)
            elif resolved_score in self.structural_scores:
                non_method_scores[original_score] = resolved_score
        
        # Calculate method-specific structural scores
        for method in method_specific_structural_scores:
            if method not in available_methods:
                continue
                
            model_file = self._get_method_model_file(processed_dir, method, 1) if processed_dir else None
            if not model_file:
                continue
                
            # Calculate scores for this method's model
            method_scores = self._calculate_structural_scores_for_pdb(
                method_specific_structural_scores[method], model_file,
                binding_site_residue_indices, required_n_contact_residues,
                binding_site_distance_threshold, template_pdb, peptide_sequence
            )
            scores.update(method_scores)
        
        # Calculate non-method-specific structural scores
        if non_method_scores:
            # For non-method scores, find the appropriate PDB file
            target_pdb_file = None
            
            if processed_dir:
                # Use processed directory - get first available method's model file
                for method in available_methods:
                    model_file = self._get_method_model_file(processed_dir, method, 1)
                    if model_file:
                        target_pdb_file = model_file
                        break
            else:
                # No processed directory - use the provided pdb_file
                target_pdb_file = pdb_file
            
            if target_pdb_file:
                # Convert to format expected by _calculate_structural_scores_for_pdb
                score_list = [(original_score, resolved_score) for original_score, resolved_score in non_method_scores.items()]
                method_scores = self._calculate_structural_scores_for_pdb(
                    score_list, target_pdb_file,
                    binding_site_residue_indices, required_n_contact_residues,
                    binding_site_distance_threshold, template_pdb, peptide_sequence
                )
                scores.update(method_scores)
        
        return scores
    
    def _calculate_structural_scores_for_pdb(self, score_list: list, pdb_file: str,
                                          binding_site_residue_indices: list, required_n_contact_residues: int,
                                          binding_site_distance_threshold: float, template_pdb: str, peptide_sequence: str) -> dict:
        """Calculate structural scores for a specific PDB file."""
        scores = {}
        
        # Initialize Rosetta scorer if needed
        rosetta_scorer = None
        rosetta_scores_needed = any(score_name in ["all_rosetta_scores", "rosetta_score", "interface_sasa", 
                                                  "interface_dG", "interface_delta_hbond_unsat", "packstat"] 
                                   for _, score_name in score_list)
        
        if rosetta_scores_needed:
            rosetta_scorer = RosettaScorer(pdb_file)
        
        # Calculate individual scores
        for original_score, score_name in score_list:
            if score_name == "all_rosetta_scores" and rosetta_scorer:
                scores.update(rosetta_scorer.get_all_metrics())
            elif score_name == "rosetta_score" and rosetta_scorer:
                scores[original_score] = rosetta_scorer.get_rosetta_score()
            elif score_name == "interface_sasa" and rosetta_scorer:
                scores[original_score] = rosetta_scorer.get_interface_sasa()
            elif score_name == "interface_dG" and rosetta_scorer:
                scores[original_score] = rosetta_scorer.get_interface_dG()
            elif score_name == "interface_delta_hbond_unsat" and rosetta_scorer:
                scores[original_score] = rosetta_scorer.get_interface_delta_hbond_unsat()
            elif score_name == "packstat" and rosetta_scorer:
                scores[original_score] = rosetta_scorer.get_packstat()
            elif score_name == "distance_score":
                scores[original_score] = distance_score_from_pdb(pdb_file)
            elif score_name == "in_binding_site":
                if not binding_site_residue_indices:
                    raise ValueError("binding_site_residue_indices required for in_binding_site score")
                n_contacts, in_binding_site = is_peptide_in_binding_site_pdb_file(
                    pdb_file, binding_site_residue_indices, binding_site_distance_threshold, required_n_contact_residues)
                scores[original_score] = in_binding_site
                scores["n_contacts"] = n_contacts
            elif score_name == "in_binding_site_score":
                if not binding_site_residue_indices:
                    raise ValueError("binding_site_residue_indices required for in_binding_site_score")
                scores[original_score] = smooth_peptide_binding_site_score(
                    pdb_file, binding_site_residue_indices, threshold=5.0, alpha=1)
            elif score_name == "template_rmsd":
                if not template_pdb:
                    raise ValueError("template_pdb required for template_rmsd score")
                scores[original_score] = align_and_compute_rmsd(template_pdb, pdb_file, peptide_sequence)
        
        return scores

    def _calculate_confidence_scores(self, resolved_scores: dict, parsed_metrics: dict, pdb_file: str, 
                                   available_methods: list, peptide_sequence: str) -> dict:
        """Calculate confidence-based scores using raw confidence data."""
        scores = {}
        
        # Only calculate confidence scores that were requested
        confidence_score_names = ["peptide_plddt", "weighted_plddt_overall", "weighted_plddt_residues", "peptide_pae", "peptide_pde"]
        requested_confidence_scores = {}
        
        for original_score, resolved_score in resolved_scores.items():
            # Check for method-specific confidence scores (e.g., "alphafold_peptide_plddt")
            for method in available_methods:
                for conf_score in confidence_score_names:
                    if resolved_score == f"{method}_{conf_score}":
                        if method not in requested_confidence_scores:
                            requested_confidence_scores[method] = {}
                        requested_confidence_scores[method][original_score] = conf_score
                        break
            
            # Check for generic confidence scores (when single method available)
            if resolved_score in confidence_score_names and len(available_methods) == 1:
                method = available_methods[0]
                if method not in requested_confidence_scores:
                    requested_confidence_scores[method] = {}
                requested_confidence_scores[method][original_score] = resolved_score
        
        # Calculate confidence scores for each method
        for method, method_scores in requested_confidence_scores.items():
            try:
                # Calculate all confidence scores for this method
                confidence_scores = calculate_peptide_confidence_scores(
                    parsed_metrics, pdb_file, method
                )
                
                # Map calculated scores to requested score names
                for original_score, conf_score_name in method_scores.items():
                    if conf_score_name in confidence_scores:
                        scores[original_score] = confidence_scores[conf_score_name]
                        
            except Exception as e:
                print(f"Error calculating confidence scores for {method}: {e}")
        
        return scores

    def _extract_docking_scores(self, resolved_scores: dict, parsed_metrics: dict, available_methods: list) -> dict:
        """Extract docking scores from parsed metrics."""
        scores = {}
        
        for original_score, resolved_score in resolved_scores.items():
            if resolved_score in parsed_metrics:
                scores[original_score] = parsed_metrics[resolved_score]
                
        return scores


if __name__ == "__main__":
    pdb_file_path = "/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV/alphafold_model_1.pdb"
    processed_dir_path = "/home/er8813ha/bopep/examples/docking/both_docking_output/processed/4glf_NYLSELSEHV"  # Example processed directory
    scorer = Scorer()

    # Single score example
    scores = scorer.score(scores_to_include=["rosetta_score"], pdb_file=pdb_file_path)
    print(f"Rosetta score for {pdb_file_path}: {scores}")

    # Example with processed directory (commented out as path may not exist)
    # scores = scorer.score(
    #     scores_to_include=[
    #         "iptm",
    #         "rosetta_score",
    #         "uHrel",
    #         "peptide_plddt",
    #         "in_binding_site",
    #         "in_binding_site_score",
    #     ],
    #     processed_dir=processed_dir_path,
    #     binding_site_residue_indices=[
    #         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    #         121, 122, 123, 124, 125, 126, 127, 128,
    #     ],
    # )
    # print(f"Scores for {processed_dir_path}: {scores}")

    # Batch scoring example
    peptide_sequences = ["ACDEFGH", "KLMNPQRS", "TVWY"]
    batch_scores = scorer.score_batch(
        scores_to_include=["molecular_weight", "gravy", "helix_fraction"],
        inputs=peptide_sequences,
        input_type="peptide_sequence",
        n_jobs=3,
    )

    print(f"Batch scores for {len(peptide_sequences)} peptides: {batch_scores}")
