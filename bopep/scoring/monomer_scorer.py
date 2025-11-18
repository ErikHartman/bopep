"""
Monomer scorer for unconditional protein generation.

This module provides scoring functionality for single-chain proteins
predicted by AlphaFold monomer, focusing on intrinsic properties rather
than binding interactions.
"""

import os
import json
import traceback
from typing import Optional, Dict, Any, List
from bopep.scoring.base_scorer import BaseScorer
from bopep.scoring.peptide_properties import PeptideProperties
from bopep.structure.parser import extract_sequence_from_structure


class MonomerScorer(BaseScorer):
    """
    Scorer for monomer proteins with intrinsic property metrics.
    
    Handles scoring of AlphaFold monomer predictions, including:
    - Confidence scores (pLDDT, pTM, PAE)
    - Sequence properties
    - Secondary structure (DSSP)
    - Structural quality metrics
    """
    
    def __init__(self):
        super().__init__()
        
        # AlphaFold monomer confidence scores
        self.confidence_scores = [
            "plddt",      # Mean pLDDT for full protein
            "ptm",        # Predicted TM-score
            "pae",        # Mean PAE
        ]
        
        
        # Store all possible scores
        self._all_possible_scores = (
            self.confidence_scores  +
            self.sequence_property_scores +
            self.dssp_scores +
            self.rmsd_scores
        )
    
    @property
    def available_scores(self):
        """Get all available scores for monomers."""
        return self._all_possible_scores
    
    def get_available_scores(
        self,
        processed_dir: Optional[str] = None,
        structure_file: Optional[str] = None,
        template_structure: Optional[str] = None,
    ) -> List[str]:
        """
        Get scores available for the given context.
        
        Parameters
        ----------
        processed_dir : str, optional
            Path to processed directory with AlphaFold output
        structure_file : str, optional
            Path to structure file (.pdb/.cif)
        template_structure : str, optional
            Template PDB file for RMSD calculation
            
        Returns
        -------
        list
            List of available score names
        """
        available = []
        
        # Always available: sequence properties
        available.extend(self.sequence_property_scores)
        
        # Confidence scores (require AlphaFold output)
        if processed_dir:
            alphafold_metrics_path = os.path.join(processed_dir, "alphafold_metrics.json")
            if os.path.exists(alphafold_metrics_path):
                available.extend(self.confidence_scores)
        
        # Structural scores (require structure file)
        if structure_file or processed_dir:
            available.extend(self.structural_scores)
            available.extend(self.dssp_scores)
        
        # Template RMSD (requires template)
        if template_structure is not None:
            available.append("template_rmsd")
        
        return sorted(list(set(available)))
    
    def score(
        self,
        scores_to_include: List[str],
        sequence: Optional[str] = None,
        structure_file: Optional[str] = None,
        processed_dir: Optional[str] = None,
        template_structure: Optional[str] = None,
        chain_id: str = "A",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate scores for a monomer protein.
        
        Parameters
        ----------
        scores_to_include : list
            List of score names to calculate
        sequence : str, optional
            Protein sequence (required if no structure_file or processed_dir)
        structure_file : str, optional
            Path to structure file (PDB/CIF)
        processed_dir : str, optional
            Path to processed AlphaFold output directory
        template_structure : str, optional
            Path to template PDB for RMSD calculation
        chain_id : str
            Chain ID to analyze (default: "A")
            
        Returns
        -------
        dict
            Dictionary mapping sequence to scores: {sequence: {score_name: value, ...}}
        """
        # Validate inputs
        if not sequence and not structure_file and not processed_dir:
            raise ValueError("Must provide either sequence, structure_file, or processed_dir")
        
        # Validate requested scores
        for score in scores_to_include:
            if score not in self.available_scores:
                raise ValueError(
                    f"ERROR: '{score}' is not a valid score. "
                    f"Available scores: {self.available_scores}"
                )
        
        scores = {}
        
        # Extract sequence if not provided
        if not sequence:
            if processed_dir:
                # Try to get sequence from metrics
                alphafold_metrics_path = os.path.join(processed_dir, "alphafold_metrics.json")
                if os.path.exists(alphafold_metrics_path):
                    with open(alphafold_metrics_path, "r") as f:
                        data = json.load(f)
                        sequence = data.get("sequence")
                if not sequence and structure_file is None:
                    # Find structure file in processed_dir
                    pdb_files = [f for f in os.listdir(processed_dir) if f.endswith('.pdb')]
                    if pdb_files:
                        structure_file = os.path.join(processed_dir, pdb_files[0])
            
            if not sequence and structure_file:
                sequence = extract_sequence_from_structure(structure_file, chain_id=chain_id)
        
        if not sequence:
            raise ValueError("Could not determine protein sequence")
        
        # Get structure file path
        if not structure_file and processed_dir:
            pdb_files = [f for f in os.listdir(processed_dir) if f.endswith('.pdb')]
            if pdb_files:
                structure_file = os.path.join(processed_dir, pdb_files[0])
        
        # Initialize peptide properties for sequence-based scoring
        peptide_properties = PeptideProperties(peptide_sequence=sequence)
        
        # Score sequence properties
        seq_scores = self._score_sequence_properties(scores_to_include, peptide_properties)
        scores.update(seq_scores)
        
        # Score DSSP if structure available
        if structure_file and any(s in scores_to_include for s in self.dssp_scores):
            dssp_scores = self._score_dssp(scores_to_include, structure_file, chain_id=chain_id)
            scores.update(dssp_scores)
        
        # Score AlphaFold confidence metrics
        if processed_dir:
            alphafold_metrics_path = os.path.join(processed_dir, "alphafold_metrics.json")
            if os.path.exists(alphafold_metrics_path):
                with open(alphafold_metrics_path, "r") as f:
                    af_data = json.load(f)
                
                if "plddt" in scores_to_include:
                    # Mean pLDDT for full protein
                    plddt_vector = af_data.get("plddt_vector", [])
                    if plddt_vector:
                        scores["plddt"] = sum(plddt_vector) / len(plddt_vector)
                    else:
                        scores["plddt"] = None
                
                if "ptm" in scores_to_include:
                    scores["ptm"] = af_data.get("ptm")
                
                if "pae" in scores_to_include:
                    # Mean PAE for full protein
                    pae_matrix = af_data.get("pae_matrix", [])
                    if pae_matrix:
                        flat_pae = [val for row in pae_matrix for val in row]
                        scores["pae"] = sum(flat_pae) / len(flat_pae) if flat_pae else None
                    else:
                        scores["pae"] = None
        
    
        
        # Template RMSD
        if "template_rmsd" in scores_to_include and template_structure and structure_file:
            scores["template_rmsd"] = self._score_template_rmsd(
                template_structure, structure_file, sequence
            )
        
        return {sequence: scores}
    
    @staticmethod
    def _process_single_input(scorer, scores_to_include, input_value, input_type, extra_kwargs):
        """
        Process a single input for scoring.
        
        This is a static method to allow pickling for multiprocessing.
        
        Parameters
        ----------
        scorer : MonomerScorer
            Scorer instance
        scores_to_include : list
            Scores to calculate
        input_value : str
            Input value (sequence, structure file, or processed dir)
        input_type : str
            Type: "sequence", "structure_file", or "processed_dir"
        extra_kwargs : dict
            Additional parameters
            
        Returns
        -------
        dict
            Scoring results
        """
        template_structure = extra_kwargs.get("template_structure")
        chain_id = extra_kwargs.get("chain_id", "A")
        
        # Extract template if needed
        needs_template = "template_rmsd" in scores_to_include
        if needs_template and "template_structures" in extra_kwargs:
            template_structures = extra_kwargs["template_structures"]
            
            # Determine sequence for template lookup
            sequence = None
            if input_type == "sequence":
                sequence = input_value
            elif input_type == "processed_dir":
                metrics_path = os.path.join(input_value, "alphafold_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        data = json.load(f)
                        sequence = data.get("sequence")
            elif input_type == "structure_file":
                sequence = extract_sequence_from_structure(input_value, chain_id=chain_id)
            
            if sequence:
                template_structure = template_structures.get(sequence)
        
        if input_type == "sequence":
            return scorer.score(
                scores_to_include,
                sequence=input_value,
                template_structure=template_structure,
                chain_id=chain_id,
            )
        elif input_type == "structure_file":
            return scorer.score(
                scores_to_include,
                structure_file=input_value,
                template_structure=template_structure,
                chain_id=chain_id,
            )
        elif input_type == "processed_dir":
            return scorer.score(
                scores_to_include,
                processed_dir=input_value,
                template_structure=template_structure,
                chain_id=chain_id,
            )
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")


if __name__ == "__main__":
    # Example usage
    scorer = MonomerScorer()
    
    print("Available scores:", scorer.available_scores)
    
    # Test with sequence only
    #test_sequence = "ACDEFGHIKLMNPQRSTVWY"
    scores = scorer.score(
        scores_to_include=["dssp_helix_fraction", "molecular_weight", "ptm"],
        processed_dir="/home/er8813ha/bopep/test_folding/processed/seq_10689525",
    )
    print(f"{scores}")
