"""
AlphaFold monomer folding for unconditional protein generation.

This module provides functionality to fold single-chain proteins using
ColabFold's AlphaFold implementation, without requiring a target structure.
"""

import os
import shutil
import subprocess
import json
import glob
import re
from typing import List, Optional
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AlphaFoldMonomer:
    """
    AlphaFold monomer folding for unconditional protein generation.
    
    Uses ColabFold to predict structures of single-chain proteins without
    requiring a target structure or binding partner.
    """
    
    def __init__(
        self,
        output_dir: str = "folding_output",
        num_models: int = 5,
        num_recycles: int = 3,
        recycle_early_stop_tolerance: float = 0.5,
        amber: bool = True,
        num_relax: int = 1,
        save_raw: bool = False,
        force: bool = False,
        msa_mode: str = "single_sequence",
        colabfold_batch_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AlphaFold monomer folder.
        
        Parameters
        ----------
        output_dir : str
            Directory for output files (default: "folding_output")
        num_models : int
            Number of models to generate (default: 5)
        num_recycles : int
            Number of recycling steps (default: 3, faster for monomers)
        recycle_early_stop_tolerance : float
            Early stopping tolerance (default: 0.5)
        amber : bool
            Whether to use AMBER relaxation (default: True)
        num_relax : int
            Number of top models to relax (default: 1)
        save_raw : bool
            Whether to keep raw ColabFold output (default: False)
        force : bool
            Whether to overwrite existing results (default: False)
        msa_mode : str
            MSA mode for ColabFold. Options:
            - "mmseqs2_uniref_env" (default): full MSA search
            - "single_sequence": use single sequence without MSA
            - "mmseqs2_uniref": MMseqs2 with UniRef only
        colabfold_batch_path : str, optional
            Path to colabfold_batch executable (auto-detected if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_models = num_models
        self.num_recycles = num_recycles
        self.recycle_early_stop_tolerance = recycle_early_stop_tolerance
        self.amber = amber
        self.num_relax = num_relax
        self.save_raw = save_raw
        self.force = force
        self.msa_mode = msa_mode
        
        # Auto-detect colabfold_batch if not provided
        if colabfold_batch_path is None:
            self.colabfold_batch_path = shutil.which("colabfold_batch")
            if self.colabfold_batch_path is None:
                raise ValueError(
                    "colabfold_batch not found in PATH. Please install ColabFold or "
                    "specify colabfold_batch_path explicitly."
                )
        else:
            self.colabfold_batch_path = colabfold_batch_path
        
        logging.info(f"Using colabfold_batch at: {self.colabfold_batch_path}")
    
    def fold(self, sequences: List[str]) -> List[str]:
        """
        Fold protein sequences using AlphaFold monomer.
        
        Parameters
        ----------
        sequences : list of str
            Protein sequences to fold
            
        Returns
        -------
        list of str
            List of paths to processed output directories for each sequence
        """
        if not sequences:
            raise ValueError("No sequences provided for folding")
        
        processed_dirs = []
        
        for sequence in sequences:
            logging.info(f"Folding sequence: {sequence}")
            processed_dir = self._fold_single_sequence(sequence)
            processed_dirs.append(processed_dir)
        
        return processed_dirs
    
    def _fold_single_sequence(self, sequence: str) -> str:
        """
        Fold a single protein sequence.
        
        Parameters
        ----------
        sequence : str
            Protein sequence to fold
            
        Returns
        -------
        str
            Path to processed output directory
        """
        # Create sequence-specific directories
        seq_id = f"{sequence}"
        
        raw_dir = self.output_dir / "raw" / seq_id
        processed_dir = self.output_dir / "processed" / seq_id
        
        # Check if already processed
        if processed_dir.exists() and not self.force:
            alphafold_metrics = processed_dir / "alphafold_metrics.json"
            if alphafold_metrics.exists():
                logging.info(f"Sequence already processed: {processed_dir}")
                return str(processed_dir)
        
        # Create raw output directory
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create FASTA file
        fasta_file = raw_dir / f"{seq_id}.fasta"
        with open(fasta_file, 'w') as f:
            f.write(f">{seq_id}\n{sequence}\n")
        
        # Run ColabFold
        logging.info(f"Running ColabFold for {seq_id}...")
        self._run_colabfold(fasta_file, raw_dir)
        
        # Process output
        logging.info(f"Processing ColabFold output...")
        processed_dir = self._process_raw_output(raw_dir, sequence, seq_id)
        
        # Clean up raw files if requested
        if not self.save_raw:
            logging.info(f"Cleaning up raw output...")
            shutil.rmtree(raw_dir)
        
        return str(processed_dir)
    
    def _run_colabfold(self, fasta_file: Path, output_dir: Path):
        """
        Execute ColabFold batch prediction.
        
        Parameters
        ----------
        fasta_file : Path
            Path to input FASTA file
        output_dir : Path
            Directory for ColabFold output
        """
        cmd = [
            self.colabfold_batch_path,
            str(fasta_file),
            str(output_dir),
            "--num-models", str(self.num_models),
            "--num-recycle", str(self.num_recycles),
            "--recycle-early-stop-tolerance", str(self.recycle_early_stop_tolerance),
            "--msa-mode", self.msa_mode,
        ]
        
        if self.amber:
            cmd.extend(["--amber", "--num-relax", str(self.num_relax)])
        
        logging.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logging.info("ColabFold completed successfully")
            if result.stdout:
                logging.debug(f"ColabFold stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"ColabFold failed with error: {e}")
            if e.stdout:
                logging.error(f"stdout: {e.stdout}")
            if e.stderr:
                logging.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"ColabFold execution failed: {e}")
    
    def _process_raw_output(
        self, 
        raw_dir: Path, 
        sequence: str, 
        seq_id: str
    ) -> Path:
        """
        Process ColabFold raw output into standardized format.
        
        Parameters
        ----------
        raw_dir : Path
            Directory containing raw ColabFold output
        sequence : str
            Input protein sequence
        seq_id : str
            Sequence identifier
            
        Returns
        -------
        Path
            Path to processed output directory
        """
        processed_dir = self.output_dir / "processed" / seq_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Find output files
        relaxed_pdbs = sorted(glob.glob(str(raw_dir / "*_relaxed_rank_*.pdb")))
        unrelaxed_pdbs = sorted(glob.glob(str(raw_dir / "*_unrelaxed_rank_*.pdb")))
        score_jsons = sorted(glob.glob(str(raw_dir / "*_scores_rank_*.json")))
        
        if not relaxed_pdbs and not unrelaxed_pdbs:
            raise RuntimeError(f"No PDB files found in {raw_dir}")
        
        logging.info(f"Found {len(relaxed_pdbs)} relaxed and {len(unrelaxed_pdbs)} unrelaxed models")
        
        # Process each model
        all_models_metrics = []
        
        # Get all available ranks
        available_ranks = set()
        for pdb_file in relaxed_pdbs + unrelaxed_pdbs:
            rank_match = re.search(r'rank_(\d+)', os.path.basename(pdb_file))
            if rank_match:
                available_ranks.add(int(rank_match.group(1)))
        
        for rank_num in sorted(available_ranks):
            rank_str = f"{rank_num:03d}"
            
            # Prefer relaxed for rank 1, unrelaxed for others
            selected_pdb = None
            if rank_num == 1 and relaxed_pdbs:
                candidates = [f for f in relaxed_pdbs if f"rank_{rank_str}" in f]
                if candidates:
                    selected_pdb = candidates[0]
            
            if selected_pdb is None:
                candidates = [f for f in unrelaxed_pdbs if f"rank_{rank_str}" in f]
                if candidates:
                    selected_pdb = candidates[0]
            
            if selected_pdb is None:
                logging.warning(f"No PDB found for rank {rank_num}")
                continue
            
            # Copy PDB to processed directory
            model_num = len(all_models_metrics) + 1
            dest_pdb = processed_dir / f"alphafold_model_{model_num}.pdb"
            shutil.copy(selected_pdb, dest_pdb)
            
            # Extract metrics from scores JSON
            score_file = None
            for json_file in score_jsons:
                if f"rank_{rank_str}" in json_file:
                    score_file = json_file
                    break
            
            if score_file:
                with open(score_file, 'r') as f:
                    scores = json.load(f)
                
                model_metrics = {
                    "model_num": model_num,
                    "rank": rank_num,
                    "pdb_file": str(dest_pdb),
                    "ptm": scores.get("ptm"),
                    "plddt": scores.get("plddt"),
                    "pae": scores.get("pae"),
                }
                all_models_metrics.append(model_metrics)
        
        if not all_models_metrics:
            raise RuntimeError("No models were successfully processed")
        
        # Save metrics for best model (rank 1)
        best_model = all_models_metrics[0]
        
        metrics = {
            "sequence": sequence,
            "ptm": best_model.get("ptm"),
            "plddt_vector": best_model.get("plddt"),
            "pae_matrix": best_model.get("pae"),
            "num_models": len(all_models_metrics),
        }
        
        metrics_file = processed_dir / "alphafold_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Processed {len(all_models_metrics)} models for {seq_id}")
        logging.info(f"Best model pTM: {best_model.get('ptm'):.3f}")
        
        return processed_dir


if __name__ == "__main__":
    # Example usage
    folder = AlphaFoldMonomer(
        output_dir="test_folding",
        num_models=2,
        num_recycles=2,
    )
    
    test_sequences = [
        "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",  # Short test
    ]
    
    results = folder.fold(test_sequences)
    print(f"Folding complete. Results: {results}")
