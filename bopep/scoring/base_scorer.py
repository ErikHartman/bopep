"""
Base scorer class with shared functionality for both complex and monomer scoring.

This module provides the foundation for scoring proteins and peptides,
including sequence-based properties, DSSP analysis, and confidence metrics.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
from typing import Optional, Dict, Any
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.scoring.dssp import DSSPAnalyzer
from bopep.scoring.model_overlap import align_and_compute_rmsd, compute_intra_model_rmsd
from bopep.structure.parser import extract_sequence_from_structure


class BaseScorer:
    """
    Base class for scoring proteins/peptides with shared scoring functionality.
    
    Provides:
    - Sequence-based properties (molecular weight, aromaticity, etc.)
    - DSSP secondary structure analysis
    - RMSD calculations
    - Parallel batch scoring infrastructure
    """
    
    def __init__(self):
        # Sequence-based scores that work without structure
        self.sequence_property_scores = [
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
        
        # DSSP-based scores (require structure)
        self.dssp_scores = [
            "dssp_helix_fraction",
            "dssp_strand_fraction",
            "dssp_loop_fraction",
        ]
        
        # RMSD scores (require template or multiple models)
        self.rmsd_scores = [
            "template_rmsd",
            "intra_model_rmsd",
            "inter_model_rmsd",
        ]
    
    def _score_sequence_properties(
        self, 
        scores_to_include: list, 
        peptide_properties: PeptideProperties
    ) -> Dict[str, Any]:
        """
        Score sequence-based properties.
        
        Parameters
        ----------
        scores_to_include : list
            List of score names to calculate
        peptide_properties : PeptideProperties
            PeptideProperties instance for the sequence
            
        Returns
        -------
        dict
            Dictionary of sequence property scores
        """
        scores = {}
        
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
            
        return scores
    
    def _score_dssp(
        self,
        scores_to_include: list,
        structure_file: str,
        chain_id: str = "B"
    ) -> Dict[str, Any]:
        """
        Score DSSP-based secondary structure.
        
        Parameters
        ----------
        scores_to_include : list
            List of score names to calculate
        structure_file : str
            Path to structure file (PDB/CIF)
        chain_id : str
            Chain ID to analyze (default: "B" for complexes, "A" for monomers)
            
        Returns
        -------
        dict
            Dictionary of DSSP scores
        """
        scores = {}
        
        if not structure_file:
            return scores
        
        dssp_analyzer = None
        if any(score in scores_to_include for score in self.dssp_scores):
            dssp_analyzer = DSSPAnalyzer(structure_file, chain_id=chain_id)
        
        if "dssp_helix_fraction" in scores_to_include:
            scores["dssp_helix_fraction"] = dssp_analyzer.get_dssp_helix_fraction()
        if "dssp_strand_fraction" in scores_to_include:
            scores["dssp_strand_fraction"] = dssp_analyzer.get_dssp_strand_fraction()
        if "dssp_loop_fraction" in scores_to_include:
            scores["dssp_loop_fraction"] = dssp_analyzer.get_dssp_loop_fraction()
            
        return scores
    
    def _score_template_rmsd(
        self,
        template_structure: str,
        model_structure: str,
        sequence: str
    ) -> Optional[float]:
        """
        Calculate RMSD between template and model.
        
        Parameters
        ----------
        template_structure : str
            Path to template PDB file
        model_structure : str
            Path to model PDB file
        sequence : str
            Peptide/protein sequence for alignment
            
        Returns
        -------
        float or None
            RMSD value or None if calculation fails
        """
        if not template_structure or not model_structure:
            return None
        
        return align_and_compute_rmsd(template_structure, model_structure, sequence)
    
    @property
    def available_scores(self):
        """
        Get all possible scores. Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement available_scores property")
    
    def score(self, *args, **kwargs):
        """
        Main scoring method. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement score() method")
    
    def score_batch(
        self,
        scores_to_include: list,
        inputs: list,
        input_type: str = "structure_file",
        n_jobs: int = None,
        **kwargs
    ) -> dict:
        """
        Score multiple inputs in parallel.
        
        This method provides a generic batch scoring framework.
        Subclasses should override _process_single_input for custom behavior.
        
        Parameters
        ----------
        scores_to_include : list
            List of score names to include
        inputs : list
            List of inputs based on input_type
        input_type : str
            Type of input (implementation-specific)
        n_jobs : int, optional
            Number of parallel jobs (default: all available cores - 1)
        **kwargs : dict
            Additional parameters passed to individual scoring calls
            
        Returns
        -------
        dict
            Dictionary with results for all inputs
        """
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
        n_jobs = min(n_jobs, len(inputs))
        all_scores = {}
        
        if n_jobs > 1:
            print(f"Processing {len(inputs)} inputs using {n_jobs} cores...")
            args_list = [
                (self, scores_to_include, input_val, input_type, kwargs)
                for input_val in inputs
            ]
            
            completed_count = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(self._process_single_input, *args)
                    for args in args_list
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_scores.update(result)
                        completed_count += 1
                        print(f"Scoring progress: {completed_count}/{len(inputs)}")
                    except Exception as e:
                        completed_count += 1
                        print(f"Error processing input ({completed_count}/{len(inputs)}): {e}")
        else:
            # Sequential processing
            for i, input_value in enumerate(inputs, 1):
                try:
                    result = self._process_single_input(
                        self, scores_to_include, input_value, input_type, kwargs
                    )
                    all_scores.update(result)
                    print(f"Scoring progress: {i}/{len(inputs)}")
                except Exception as e:
                    print(f"Error processing input {input_value} ({i}/{len(inputs)}): {e}")
                    traceback.print_exc()
        
        print(f"Scoring complete! Processed {len(all_scores)} inputs.")
        return all_scores
    
    @staticmethod
    def _process_single_input(scorer, scores_to_include, input_value, input_type, extra_kwargs):
        """
        Process a single input for scoring.
        
        This is a static method to allow pickling for multiprocessing.
        Override in subclasses for custom behavior.
        """
        raise NotImplementedError("Subclasses must implement _process_single_input()")
