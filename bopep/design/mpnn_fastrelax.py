import os
import sys
import re
import glob
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import concurrent.futures

try:
    from pyrosetta import *
    from pyrosetta.rosetta import *
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.core.pose import Pose
    import pyrosetta.rosetta as rosetta
    PYROSETTA_AVAILABLE = True
except ImportError:
    logging.warning("PyRosetta not available. Some functionality will be limited.")
    PYROSETTA_AVAILABLE = False


class MPNNFastRelax:
    """
    A class for running ProteinMPNN + FastRelax pipeline on RFDiffusion designs.
    
    This class handles the complete pipeline from RFDiffusion PDB files through
    ProteinMPNN sequence design to PyRosetta FastRelax optimization.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        designs_dir: Optional[str] = None,
        protein_mpnn_path: Optional[str] = None,
        mpnn_chains: str = "A",
        mpnn_env: str = sys.executable,
    ):
        """
        Initialize the MPNNFastRelax class.
        
        Parameters
        ----------
        output_dir : str, optional
            Base output directory. If None, uses OUTPUT_DIR from environment.
        designs_dir : str, optional
            Directory containing RFDiffusion PDB files.
        protein_mpnn_path : str, optional
            Path to ProteinMPNN installation.
        mpnn_chains : str, default "A"
            Chains to design with ProteinMPNN.
        mpnn_env : str, default sys.executable
            Path to the Python environment for running ProteinMPNN. If None, uses the current Python executable.
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(levelname)s: %(message)s', 
            stream=sys.stderr
        )

        # Set up paths
        base_output_dir = Path(output_dir)
        
        self.output_dir = base_output_dir
        self.designs_dir = Path(designs_dir or base_output_dir / "designs")
        self.sequence_output_dir = base_output_dir / "mpnn_fastrelax_outputs"
        
        # ProteinMPNN configuration
        self.protein_mpnn_path = Path(protein_mpnn_path or os.getenv("PROTEIN_MPNN_PATH"))
        self.mpnn_chains = mpnn_chains
        self.mpnn_env = mpnn_env
        
        # Initialize PyRosetta and FastRelax
        self._initialize_pyrosetta()
        
        # Storage for interface dG scores
        self.interface_dg_scores = []
        
        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sequence_output_dir.mkdir(exist_ok=True, parents=True)
    
    def _initialize_pyrosetta(self):
        """Initialize PyRosetta and FastRelax mover."""
        if not PYROSETTA_AVAILABLE:
            logging.warning("PyRosetta not available. FastRelax functionality disabled.")
            self.fast_relax = None
            return
            
        try:
            from pyrosetta import init
            from pyrosetta.rosetta import protocols
            
            init("-beta_nov16 -in:file:silent_struct_type binary -use_terminal_residues true -mute all")

            # Fetch the custom FastRelax XML            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(script_dir, "rosetta/RosettaFastRelaxUtil.xml")
            
            if not os.path.exists(xml_path):
                logging.warning(f"Rosetta FastRelax XML file not found: {xml_path}")
                logging.warning("FastRelax functionality may be limited")
                self.fast_relax = None
            else:
                objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml_path)
                self.fast_relax = objs.get_mover('FastRelax')
        except Exception as e:
            logging.error(f"Failed to initialize PyRosetta: {e}")
            self.fast_relax = None
    
    def find_design_pdbs(self, designs_dir: Optional[str] = None) -> List[str]:
        """
        Find RFDiffusion design PDB files.
        
        Parameters
        ----------
        designs_dir : str, optional
            Directory to search. If None, uses self.designs_dir.
        
        Returns
        -------
        List[str]
            List of PDB file paths.
        """
        search_dir = Path(designs_dir) if designs_dir else self.designs_dir
        pattern = os.path.join(search_dir, "sample_*/design_*.pdb")
        pdbs = glob.glob(pattern)
        
        if not pdbs:
            raise FileNotFoundError(f"No PDBs found matching {pattern}")
        
        return sorted(pdbs)
    
    def extract_peptide_from_pdb(self, pdb_file: str) -> Optional[Dict[str, Any]]:
        """
        Extract peptide information from PDB file.
        
        Parameters
        ----------
        pdb_file : str
            Path to PDB file.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing peptide information or None if failed.
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("structure", pdb_file)
            chain = next((c for c in structure.get_chains() if c.id == "A"), None)
            if chain is None:
                chain = next(structure.get_chains(), None)
            
            if chain is None:
                logging.warning(f"No chain in {pdb_file}")
                return None
            
            seq = "".join(
                seq1(res.get_resname()) if PDB.is_aa(res) else "X" 
                for res in chain
            )
            
            sample_id_match = re.search(r"sample_(\d+)", pdb_file)
            sample_id = int(sample_id_match.group(1)) if sample_id_match else None
            
            return {
                "pdb_file": os.path.basename(pdb_file),
                "sample_id": sample_id,
                "chain_ids": chain.id,
                "sequence": seq,
                "length": len(seq),
                "full_path": pdb_file,
            }
        except Exception as exc:
            logging.warning(f"Error reading {pdb_file}: {exc}")
            return None
    
    def run_single_mpnn(self, pdb: str, temperature: float) -> bool:
        """
        Run ProteinMPNN on a single PDB file.
        
        Parameters
        ----------
        pdb : str
            Path to PDB file.
        temperature : float
            Sampling temperature for ProteinMPNN.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        sample_id_match = re.search(r"sample_(\d+)", pdb)
        if not sample_id_match:
            logging.error(f"Could not extract sample ID from {pdb}")
            return False
        
        sample_id = sample_id_match.group(1)
        out_dir = self.sequence_output_dir / "fastas" / f"sample_{sample_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        cmd = [
            self.mpnn_env,
            str(self.protein_mpnn_path / "protein_mpnn_run.py"),
            "--pdb_path", pdb,
            "--pdb_path_chains", self.mpnn_chains,
            "--out_folder", str(out_dir),
            "--num_seq_per_target", "1",
            "--sampling_temp", str(temperature),
            "--batch_size", "1",
        ]
        
        try:
            subprocess.run(
                cmd, 
                check=True, 
                cwd=self.protein_mpnn_path, 
                capture_output=True, 
                text=True, 
                env=env
            )
            logging.info(f"ProteinMPNN completed for {os.path.basename(os.path.dirname(pdb))}")
            return True
        except subprocess.CalledProcessError as err:
            logging.error(f"ProteinMPNN failed on {os.path.basename(pdb)}: {err}")
            if err.stderr:
                logging.error(f"Stderr: {err.stderr.strip()}")
            return False
        except Exception as e:
            logging.error(f"Error running ProteinMPNN on {os.path.basename(pdb)}: {e}")
            return False
    
    def run_proteinmpnn(self, pdb_files: List[str], temperature: float, threads: int = 1) -> Path:
        """
        Run ProteinMPNN on multiple PDB files.
        
        Parameters
        ----------
        pdb_files : List[str]
            List of PDB file paths.
        temperature : float
            Sampling temperature.
        threads : int, default 1
            Number of threads for parallel processing.
        
        Returns
        -------
        Path
            Output directory containing FASTA files.
        """
        out_root = self.sequence_output_dir / "fastas"
        out_root.mkdir(parents=True, exist_ok=True)
        successful_runs = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self.run_single_mpnn, pdb, temperature) 
                for pdb in pdb_files
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    successful_runs += 1
        
        logging.info(f"ProteinMPNN completed: {successful_runs}/{len(pdb_files)} structures")
        return out_root
    
    def seq_from_fasta(self, fa_path: Path) -> Optional[str]:
        """
        Extract sequence from FASTA file (second record).
        
        Parameters
        ----------
        fa_path : Path
            Path to FASTA file.
        
        Returns
        -------
        Optional[str]
            Sequence string or None if not found.
        """
        try:
            with fa_path.open() as fh:
                lines = [l.strip() for l in fh if l.strip()]
            
            seqs = []
            for i, line in enumerate(lines):
                if line.startswith(">") and i + 1 < len(lines):
                    seqs.append(lines[i + 1].split("/")[0])
            
            if len(seqs) >= 2:
                return seqs[1]
            else:
                logging.warning(f"Not enough sequences in {fa_path}. Expected at least 2, found {len(seqs)}.")
            return None
        except Exception as e:
            logging.error(f"Error reading FASTA {fa_path}: {e}")
            return None
    
    def extract_fasta_metrics(self, fa_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Extract metrics from FASTA header.
        
        Parameters
        ----------
        fa_path : Path
            Path to FASTA file.
        
        Returns
        -------
        Tuple[Optional[float], Optional[float], Optional[float]]
            score, global_score, seq_recovery
        """
        try:
            with fa_path.open() as fh:
                lines = [l.strip() for l in fh if l.strip()]
            
            header_lines = [l for l in lines if l.startswith(">")]
            if len(header_lines) < 2:
                return None, None, None
            
            header = header_lines[1]
            
            def extract_float(pattern):
                m = re.search(pattern, header)
                return float(m.group(1)) if m else None
            
            score = extract_float(r"score=([\d\.\-eE]+)")
            global_score = extract_float(r"global_score=([\d\.\-eE]+)")
            seq_recovery = extract_float(r"seq_recovery=([\d\.\-eE]+)")
            
            return score, global_score, seq_recovery
        except Exception as e:
            logging.error(f"Error extracting metrics from {fa_path}: {e}")
            return None, None, None
    
    def interface_dG(self, pose) -> float:
        """
        Compute interface dG for a pose.
        
        Parameters
        ----------
        pose : Pose
            PyRosetta pose object.
        
        Returns
        -------
        float
            Interface dG value.
        """
        if not PYROSETTA_AVAILABLE:
            logging.error("PyRosetta not available for interface_dG calculation")
            return 0.0
            
        try:
            from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
            ia = InterfaceAnalyzerMover()
            ia.set_interface('A_B')
            ia.apply(pose)
            return ia.get_interface_dG()
        except Exception as e:
            logging.error(f"Error calculating interface dG: {e}")
            return 0.0
    
    @staticmethod
    def thread_and_relax_job(args) -> List[Any]:
        """
        Worker function for threading and FastRelax.
        
        Parameters
        ----------
        args : tuple
            Arguments for the worker function.
        
        Returns
        -------
        List[Any]
            Results from threading and relaxation.
        """
        (pdb, seq, threaded_pdb_path, relaxed_pdb_path, cycle, sample_id) = args
        
        if not PYROSETTA_AVAILABLE:
            logging.error("PyRosetta not available for thread_and_relax_job")
            return [None, os.path.basename(relaxed_pdb_path), cycle, sample_id]
        
        try:
            # Import PyRosetta functions in worker process
            from pyrosetta import init, pose_from_pdb
            from pyrosetta.rosetta import protocols
            from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
            import pyrosetta.rosetta as rosetta
            
            # PyRosetta init in worker process
            init("-beta_nov16 -in:file:silent_struct_type binary -use_terminal_residues true -mute all basic.io.database core.scoring")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            xml = os.path.join(script_dir, "rosetta/RosettaFastRelaxUtil.xml")
            
            if os.path.exists(xml):
                objs = protocols.rosetta_scripts.XmlObjects.create_from_file(xml)
                FastRelax = objs.get_mover('FastRelax')
            else:
                logging.warning(f"XML file not found: {xml}")
                return [None, os.path.basename(relaxed_pdb_path), cycle, sample_id]
            
            # Thread sequence
            pose = pose_from_pdb(pdb)
            rsd_set = pose.residue_type_set_for_pose(rosetta.core.chemical.FULL_ATOM_t)
            
            if pose.chain_sequence(1) and len(seq) != pose.chain_sequence(1).__len__():
                raise ValueError(f"Sequence length ({len(seq)}) does not match chain A length ({pose.chain_sequence(1).__len__()}).")
            
            aa_1_3 = {
                aa: rosetta.core.chemical.name_from_aa(rosetta.core.chemical.aa_from_oneletter_code(aa)) 
                for aa in seq
            }
            
            chain_start = pose.conformation().chain_begin(1)
            for i, mut_to in enumerate(seq):
                resi = chain_start + i
                name3 = aa_1_3[mut_to]
                new_res = rosetta.core.conformation.ResidueFactory.create_residue(rsd_set.name_map(name3))
                pose.replace_residue(resi, new_res, True)
            
            pose.dump_pdb(str(threaded_pdb_path))
            
            # FastRelax
            movemap = rosetta.core.kinematics.MoveMap()
            for i in range(1, pose.total_residue() + 1):
                if pose.pdb_info().chain(i) == "A":
                    movemap.set_bb(i, True)
                    movemap.set_chi(i, True)
                else:
                    movemap.set_bb(i, False)
                    movemap.set_chi(i, False)
            
            FastRelax.set_movemap(movemap)
            FastRelax.apply(pose)
            pose.dump_pdb(str(relaxed_pdb_path))
            
            # dG calculation
            ia = InterfaceAnalyzerMover()
            ia.set_interface('A_B')
            ia.apply(pose)
            dG = ia.get_interface_dG()
            
            return [dG, os.path.basename(relaxed_pdb_path), cycle, sample_id]
        except Exception as e:
            logging.error(f"Thread+Relax failed for {pdb}: {e}")
            return [None, os.path.basename(relaxed_pdb_path), cycle, sample_id]
    
    def run_mpnn_fastrelax_pipeline(
        self, 
        pdb_files: List[str], 
        temperature: float, 
        cycles: int = 1, 
        threads: int = 1
    ) -> bool:
        """
        Run the complete MPNN + FastRelax pipeline.
        
        Parameters
        ----------
        pdb_files : List[str]
            List of input PDB files.
        temperature : float
            ProteinMPNN sampling temperature.
        cycles : int, default 1
            Number of MPNN + FastRelax cycles.
        threads : int, default 1
            Number of parallel threads.
        
        Returns
        -------
        bool
            True if successful.
        """
        out_root = self.sequence_output_dir
        pdbs_dir = out_root / "pdbs"
        pdbs_dir.mkdir(parents=True, exist_ok=True)
        current_pdbs = pdb_files
        
        for cycle in range(cycles):
            logging.info(f"Cycle {cycle+1}/{cycles}: Running ProteinMPNN on input PDBs")
            mpnn_root = self.run_proteinmpnn(current_pdbs, temperature, threads)
            
            logging.info(f"Cycle {cycle+1}/{cycles}: Threading + FastRelax in parallel")
            jobs = []
            
            for pdb in current_pdbs:
                sample_id_match = re.search(r"sample_(\d+)", pdb)
                if not sample_id_match:
                    continue
                    
                sample_id = sample_id_match.group(1)
                sample_dir_pdbs = pdbs_dir / f"sample_{sample_id}"
                sample_dir_pdbs.mkdir(parents=True, exist_ok=True)
                
                # Find FASTA file
                if cycle == 0:
                    seq_fa = list((mpnn_root / f"sample_{sample_id}" / "seqs").glob("design_0*.fa"))
                else:
                    seq_fa = list((mpnn_root / f"sample_{sample_id}" / "seqs").glob(f"cycle{cycle}_relaxed.fa"))
                
                if not seq_fa:
                    logging.warning(f"No FASTA file found for {pdb} (cycle {cycle})")
                    continue
                
                seq = self.seq_from_fasta(seq_fa[0])
                if not seq:
                    logging.warning(f"No sequence in {seq_fa[0]}")
                    continue
                
                threaded_pdb_path = sample_dir_pdbs / f"cycle{cycle+1}_threaded.pdb"
                relaxed_pdb_path = sample_dir_pdbs / f"cycle{cycle+1}_relaxed.pdb"
                jobs.append((pdb, seq, threaded_pdb_path, relaxed_pdb_path, cycle+1, sample_id))
            
            next_pdbs = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
                results = list(executor.map(self.thread_and_relax_job, jobs))
            
            for res in results:
                if res[0] is not None:
                    self.interface_dg_scores.append(res)
                    # Use current cycle's relaxed PDB for next cycle
                    curr_cycle = int(res[2])
                    next_pdbs.append(str(pdbs_dir / f"sample_{res[3]}" / f"cycle{curr_cycle}_relaxed.pdb"))
                else:
                    logging.warning(f"Thread+Relax failed for {res[1]}")
            
            # Write dG scores to CSV after each cycle
            if self.interface_dg_scores:
                dg_df = pd.DataFrame(
                    self.interface_dg_scores, 
                    columns=["interface_dG", "relaxed_pdb", "cycle", "sample_id"]
                )
                csv_path = self.sequence_output_dir / "interface_dG_scores.csv"
                
                if csv_path.exists():
                    # Append only new rows
                    old_df = pd.read_csv(csv_path)
                    combined_df = pd.concat([old_df, dg_df]).drop_duplicates().reset_index(drop=True)
                    combined_df.to_csv(csv_path, index=False)
                else:
                    dg_df.to_csv(csv_path, index=False)
                
                logging.info(f"interface_dG_scores.csv updated after cycle {cycle+1}")
            
            current_pdbs = next_pdbs
        
        return True
    
    def create_output_csv(self, fastas_root: Path, output_path: str) -> None:
        """
        Create comprehensive output CSV with all results.
        
        Parameters
        ----------
        fastas_root : Path
            Root directory containing FASTA files.
        output_path : str
            Path for output CSV file.
        """
        # Load dG scores
        dg_csv_path = self.sequence_output_dir / "interface_dG_scores.csv"
        if dg_csv_path.exists():
            dg_df = pd.read_csv(dg_csv_path)
            dg_df["cycle"] = dg_df["cycle"].astype(int)
        else:
            dg_df = pd.DataFrame()
        
        # Load sample data
        sample_csv_path = self.output_dir / "samples" / "peptide_samples.csv"
        if sample_csv_path.exists():
            sample_df = pd.read_csv(sample_csv_path)
        else:
            sample_df = pd.DataFrame()
        
        records = []
        for sample_dir in sorted(fastas_root.glob("sample_*")):
            seq_fas = list((sample_dir / "seqs").glob("*"))
            for fa in seq_fas:
                seq = self.seq_from_fasta(fa)
                sample_id_match = re.search(r"sample_(\d+)", str(sample_dir))
                
                # Determine cycle number
                if fa.name == "design_0.fa":
                    cycle = 1
                    fasta_cycle_name = "design_0.fa"
                else:
                    m_design = re.search(r"design_(\d+).fa", fa.name)
                    m_cycle = re.search(r"cycle(\d+)", fa.name)
                    if m_design:
                        cycle = int(m_design.group(1)) + 1
                    elif m_cycle:
                        cycle = int(m_cycle.group(1)) + 1
                    else:
                        cycle = ""
                    fasta_cycle_name = fa.name
                
                score, global_score, seq_recovery = self.extract_fasta_metrics(fa)
                
                # Find corresponding PDB path
                pdb_path = None
                if (sample_id_match and sample_id_match.group(1).isdigit() and 
                    str(cycle).isdigit()):
                    pdb_cycle_idx = int(cycle)
                    pdb_path = str(
                        self.sequence_output_dir / "pdbs" / 
                        f"sample_{sample_id_match.group(1)}" / 
                        f"cycle{pdb_cycle_idx}_relaxed.pdb"
                    )
                
                records.append({
                    "sample_id": sample_id_match.group(1) if sample_id_match else "",
                    "cycle": cycle,
                    "sequence": seq,
                    "fasta_file": fasta_cycle_name,
                    "mpnn_score": score,
                    "mpnn_global_score": global_score,
                    "mpnn_seq_recovery": seq_recovery,
                    "full_path": pdb_path,
                })
        
        # Create DataFrame and filter
        df = pd.DataFrame(records)
        df = df[
            df["sample_id"].apply(lambda x: str(x).isdigit()) & 
            df["cycle"].apply(lambda x: str(x).isdigit())
        ]
        
        if not df.empty:
            df["sample_id"] = df["sample_id"].astype(int)
            df["cycle"] = df["cycle"].astype(int)
            
            # Merge with dG scores
            if not dg_df.empty:
                dg_df["sample_id"] = dg_df["sample_id"].astype(int)
                df = df.merge(dg_df, on=["sample_id", "cycle"], how="left")
            
            # Merge with sample data
            if not sample_df.empty:
                df = df.merge(sample_df, on="sample_id", how="left")
        df = df[["sample_id", "cycle", "sequence", "interface_dG",  "mpnn_score", "mpnn_global_score", "mpnn_seq_recovery", "full_path",  "relaxed_pdb", "fasta_file"]]
        df.to_csv(output_path, index=False)
        logging.info(f"Sequences saved to {output_path}")
    
    def run(
        self,
        designs_dir: Optional[str] = None,
        temperature: float = 0.1,
        relax_cycles: int = 1,
        threads: int = 4,
        limited_run: int = 0,
        output_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete MPNN + FastRelax pipeline.
        
        Parameters
        ----------
        designs_dir : str, optional
            Directory containing design PDB files.
        temperature : float, default 0.1
            ProteinMPNN sampling temperature.
        relax_cycles : int, default 1
            Number of MPNN + FastRelax cycles.
        threads : int, default 4
            Number of parallel threads.
        limited_run : int, default 0
            Limit number of PDBs (0 = no limit).
        output_csv : str, optional
            Output CSV path.
        
        Returns
        -------
        Dict[str, Any]
            Results summary.
        """
        try:
            # Find PDB files
            pdb_files = self.find_design_pdbs(designs_dir)
            
            # Extract sequences and sample info
            seq_dicts = [self.extract_peptide_from_pdb(pdb) for pdb in pdb_files]
            seq_df = pd.DataFrame([d for d in seq_dicts if d])
            seq_df = seq_df.sort_values("sample_id")
            
            if limited_run > 0:
                logging.info(f'Limiting to {limited_run} PDB files for testing')
                seq_df = seq_df.iloc[:limited_run]
            
            logging.info(f"Found {len(seq_df)} PDB files (sorted by sample_id)")
            
            if seq_df.empty:
                raise ValueError("No PDB files found")
            
            # Run pipeline
            logging.info(f"Running MPNN + FastRelax pipeline (T={temperature})")
            success = self.run_mpnn_fastrelax_pipeline(
                seq_df["full_path"].tolist(), 
                temperature, 
                relax_cycles, 
                threads
            )
            
            # Save results
            if self.interface_dg_scores:
                dg_df = pd.DataFrame(
                    self.interface_dg_scores, 
                    columns=["interface_dG", "relaxed_pdb", "cycle", "sample_id"]
                )
                dg_df.to_csv(self.sequence_output_dir / "interface_dG_scores.csv", index=False)
                logging.info(f"interface_dG_scores.csv saved to {self.sequence_output_dir}")
            
            # Create output CSV
            output_path = output_csv or str(self.output_dir / "borf_output.csv")
            self.create_output_csv(self.sequence_output_dir / "fastas", output_path)
            
            results = {
                "success": success,
                "processed_pdbs": len(seq_df),
                "output_csv": output_path,
                "interface_dg_scores": len(self.interface_dg_scores),
                "sequence_output_dir": str(self.sequence_output_dir)
            }
            
            logging.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise


def main():
    """Command-line interface for MPNNFastRelax."""
    parser = argparse.ArgumentParser(
        description="Run ProteinMPNN + FastRelax pipeline on RFDiffusion designs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--designs-dir", help="Folder with RFDiffusion PDBs")
    parser.add_argument("--output", help="CSV output path")
    parser.add_argument("--output-dir", help="Base output directory")
    parser.add_argument("--threads", type=int, default=4, help="Threads for parallel processing")
    parser.add_argument("--temperature", type=float, default=0.1, help="ProteinMPNN sampling temperature")
    parser.add_argument("--relax-cycles", type=int, default=1, help="Number of cycles for MPNN + FastRelax")
    parser.add_argument("--limited-run", type=int, default=0, help="Limit number of PDBs to process")
    parser.add_argument("--mpnn-env", help="Python environment/executable for ProteinMPNN")
    parser.add_argument("--test", action="store_true", help="Add '_test' suffix to output paths")
    
    args = parser.parse_args()
    
    try:
        # Initialize MPNNFastRelax
        mpnn_fastrelax = MPNNFastRelax(
            output_dir=args.output_dir,
            designs_dir=args.designs_dir,
            mpnn_env=args.mpnn_env
        )
        
        # Run pipeline
        results = mpnn_fastrelax.run(
            designs_dir=args.designs_dir,
            temperature=args.temperature,
            relax_cycles=args.relax_cycles,
            threads=args.threads,
            limited_run=args.limited_run,
            output_csv=args.output
        )
        
        logging.info(f"Results: {results}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()