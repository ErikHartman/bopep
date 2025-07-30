import os
import sys
import time
import argparse
import concurrent.futures
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd

class RFDiffusion:
    """
    A class for running RFdiffusion protein design tasks with GPU parallelization.
    
    This class handles the execution of RFdiffusion inference on multiple GPUs,
    processes peptide samples, and manages output directories and logging.
    """
    
    def __init__(
        self,
        rfdiffusion_path: str,
        output_dir: Optional[str] = None,
        pdb_path: Optional[str] = None,
        models_path: Optional[str] = None,
        python_env_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the RFDiffusion class.

        Parameters
        ----------
        rfdiffusion_path : str
            Path to RFdiffusion installation directory. (Mandatory)
        output_dir : str, optional
            Directory for output files. If None, uses OUTPUT_DIR from environment or current directory.
        pdb_path : str, optional
            Path to the input PDB file. If None, uses default PLO1_PATH.
        models_path : str, optional
            Path to RFdiffusion models directory.
        python_env_path : str, optional
            Path to Python environment for RFdiffusion. If None, uses the expected Python executable in RFdiffusion.
        checkpoint_path : str, optional
            Path to the checkpoint file to use.
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(levelname)s: %(message)s', 
            stream=sys.stderr
        )

        # Validate mandatory argument
        if not rfdiffusion_path:
            raise ValueError("rfdiffusion_path is a mandatory argument and must be provided.")

        # Set up paths
        self.output_dir = Path(output_dir)
        self.designs_dir = self.output_dir / "designs"
        self.logs_dir = self.output_dir / "logs"
        self.samples_csv = self.output_dir / "samples" / "peptide_samples.csv"

        # RFdiffusion configuration paths
        self.pdb_path = Path(pdb_path) if pdb_path else None
        self.rfdiffusion_path = rfdiffusion_path
        self.models_path = models_path or f"{self.rfdiffusion_path}/models"
        self.python_env_path = python_env_path or f"{self.rfdiffusion_path}/env/rf_env/bin/python"
        self.checkpoint_path = checkpoint_path or f"{self.models_path}/Complex_beta_ckpt.pt"

        # Create necessary directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.designs_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
    
    def get_available_gpus(self) -> List[int]:
        """
        Detect available GPUs for processing.
        
        Returns
        -------
        List[int]
            List of available GPU indices.
        """
        try:
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_devices:
                logging.info(f"Using GPUs from CUDA_VISIBLE_DEVICES: {cuda_devices}")
                devices = [int(idx.strip()) for idx in cuda_devices.split(',') if idx.strip()]
                if devices:
                    return devices
            
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], 
                capture_output=True, text=True, check=True
            )
            gpu_indices = [int(idx.strip()) for idx in result.stdout.strip().split('\n') if idx.strip()]
            return gpu_indices
        except (subprocess.SubprocessError, FileNotFoundError):
            logging.warning("Could not detect GPUs. Assuming one GPU (index 0).")
            return [0]
    
    def load_samples(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load peptide samples from CSV file.
        
        Parameters
        ----------
        csv_path : str, optional
            Path to the samples CSV file. If None, uses self.samples_csv.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample information.
        """
        if csv_path is None:
            csv_path = self.samples_csv
        else:
            csv_path = Path(csv_path)
            
        if not csv_path.exists():
            logging.error(f"Samples file not found at {csv_path}")
            logging.error("Please ensure the samples file exists.")
            raise FileNotFoundError(f"Samples file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        return df.reset_index(drop=True)
    
    def run_rfdiffusion_single(self, args: Tuple[Any, int, bool]) -> Tuple[bool, str, int]:
        """
        Run RFdiffusion for a single sample.
        
        Parameters
        ----------
        args : tuple
            Tuple containing (sample, gpu_id, dry_run).
        
        Returns
        -------
        Tuple[bool, str, int]
            Success status, sample ID, and GPU ID.
        """
        sample, gpu_id, dry_run = args
        sample_id = sample['sample_id']
        length = sample['length']
        hotspots = sample['hotspots']
        
        log_file = self.logs_dir / f"sample_{sample_id}.log"
        design_dir = self.designs_dir / f"sample_{sample_id}"
        design_dir.mkdir(exist_ok=True, parents=True)
        
        params = [
            f"inference.output_prefix={design_dir}/design",
            f"inference.model_directory_path={self.models_path}",
            f"inference.input_pdb={self.pdb_path}",
            "inference.num_designs=1",
            f"contigmap.contigs=[A359-471/0 {length}]",
            f"ppi.hotspot_res=[{hotspots}]",
            f"inference.ckpt_override_path={self.checkpoint_path}",
        ]
        
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [self.python_env_path, f"{self.rfdiffusion_path}/scripts/run_inference.py"]
        cmd.extend(params)
        
        logging.info(f"Running sample {sample_id} on GPU {gpu_id}")
        
        if dry_run:
            print(f"[DRY RUN] GPU {gpu_id}: {' '.join(cmd)}")
            return True, sample_id, gpu_id
        
        with open(log_file, 'w') as log:
            try:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=my_env)
                logging.info(f"Completed sample {sample_id}")
                return True, sample_id, gpu_id
            except subprocess.CalledProcessError as e:
                logging.error(f"Error running sample {sample_id}: {e}")
                if log_file.exists():
                    with open(log_file, 'r') as errlog:
                        error_lines = errlog.readlines()
                        last_lines = error_lines[-10:] if len(error_lines) > 10 else error_lines
                        logging.error(f"Error log: {log_file}")
                        for line in last_lines:
                            logging.error(f"  > {line.strip()}")
                return False, sample_id, gpu_id
            except Exception as ex:
                logging.error(f"Unexpected error for sample {sample_id}: {ex}")
                return False, sample_id, gpu_id
    
    @staticmethod
    def worker(tasks: List[Tuple[Any, int, bool, dict]]) -> List[Tuple[bool, str, int]]:
        """
        Worker function for processing multiple tasks on a single GPU.
        Each task tuple must now include a config dict for RFDiffusion instantiation.
        """
        results = []
        if tasks:
            # Extract config from the first task
            _, _, _, config = tasks[0]
            diffusion_instance = RFDiffusion(
                rfdiffusion_path=config['rfdiffusion_path'],
                output_dir=config.get('output_dir'),
                pdb_path=config.get('pdb_path'),
                models_path=config.get('models_path'),
                python_env_path=config.get('python_env_path'),
                checkpoint_path=config.get('checkpoint_path')
            )
            for args in tasks:
                sample, gpu_id, dry_run, _ = args
                results.append(diffusion_instance.run_rfdiffusion_single((sample, gpu_id, dry_run)))
        return results
    
    def process_samples(
        self, 
        samples_df: Optional[pd.DataFrame] = None,
        gpus: Optional[List[int]] = None,
        dry_run: bool = False,
        skip_existing: bool = True
    ) -> Tuple[int, int]:
        """
        Process multiple samples with GPU parallelization.
        
        Parameters
        ----------
        samples_df : pd.DataFrame, optional
            DataFrame containing samples to process. If None, loads from self.samples_csv.
        gpus : List[int], optional
            List of GPU indices to use. If None, auto-detects available GPUs.
        dry_run : bool, default False
            If True, print commands without executing them.
        skip_existing : bool, default True
            If True, skip samples that have already been processed.
        
        Returns
        -------
        Tuple[int, int]
            Number of successful and failed runs.
        """
        if samples_df is None:
            samples_df = self.load_samples()
        
        if gpus is None:
            gpus = self.get_available_gpus()
        
        if not gpus:
            logging.error("No GPUs detected. Cannot run RFdiffusion.")
            raise RuntimeError("No GPUs available")
        
        logging.info(f"Processing {len(samples_df)} samples with {len(gpus)} GPUs: {gpus}")
        
        # Filter out samples that have already been processed
        if skip_existing:
            filtered_samples = []
            for _, row in samples_df.iterrows():
                design_dir = self.designs_dir / f"sample_{row['sample_id']}"
                design_file = design_dir / "design_0.pdb"
                if not design_file.exists():
                    filtered_samples.append(row)
                else:
                    logging.info(f"Skipping sample {row['sample_id']} (already processed)")
            
            if not filtered_samples:
                logging.info("All samples have already been processed.")
                return 0, 0
            
            samples_df = pd.DataFrame(filtered_samples)
        
        # Group tasks by GPU
        gpu_tasks = {gpu: [] for gpu in gpus}
        # Prepare config dict for worker
        config = {
            'rfdiffusion_path': self.rfdiffusion_path,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'pdb_path': str(self.pdb_path) if self.pdb_path else None,
            'models_path': self.models_path,
            'python_env_path': self.python_env_path,
            'checkpoint_path': self.checkpoint_path
        }
        for i, (_, row) in enumerate(samples_df.iterrows()):
            gpu_id = gpus[i % len(gpus)]
            gpu_tasks[gpu_id].append((row, gpu_id, dry_run, config))
        
        successful_runs = 0
        failed_runs = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            futures = []
            for gpu_id in gpus:
                if gpu_tasks[gpu_id]:  # Only submit if there are tasks
                    futures.append(executor.submit(RFDiffusion.worker, gpu_tasks[gpu_id]))
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    for success, sample_id, gpu_id in results:
                        if success:
                            successful_runs += 1
                        else:
                            failed_runs += 1
                except Exception as e:
                    logging.error(f"Error in worker process: {e}")
                    failed_runs += 1
        return successful_runs, failed_runs
    
    def run(
        self,
        samples_csv: Optional[str] = None,
        dry_run: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to run the complete RFdiffusion pipeline.
        
        Parameters
        ----------
        samples_csv : str, optional
            Path to samples CSV file. If None, uses self.samples_csv.
        dry_run : bool, default False
            If True, print commands without executing them.
        skip_existing : bool, default True
            If True, skip samples that have already been processed.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing run statistics and results.
        """
        gpus = self.get_available_gpus()
        if not gpus:
            logging.error("No GPUs detected. Cannot run RFdiffusion.")
            raise RuntimeError("No GPUs available")
        
        logging.info(f"Using {len(gpus)} GPUs: {gpus}")
        
        try:
            samples_df = self.load_samples(samples_csv)
        except FileNotFoundError as e:
            logging.error(str(e))
            raise
        
        if samples_df.empty:
            logging.error("No samples found.")
            raise ValueError("No samples to process")
        
        if dry_run:
            logging.info("Running in dry-run mode.")
        
        start_time = time.time()
        successful, failed = self.process_samples(
            samples_df=samples_df,
            gpus=gpus,
            dry_run=dry_run,
            skip_existing=skip_existing
        )
        elapsed_time = time.time() - start_time
        
        results = {
            'successful_runs': successful,
            'failed_runs': failed,
            'elapsed_time': elapsed_time,
            'gpus_used': gpus,
            'total_samples': len(samples_df)
        }
        
        logging.info(f"Run completed in {elapsed_time:.2f}s - Success: {successful}, Failed: {failed}")
        
        return results


def main():
    """Command-line interface for RFDiffusion."""
    parser = argparse.ArgumentParser(description="Run RFdiffusion on GPU")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing them")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results")
    parser.add_argument("--samples-csv", type=str,
                       help="Path to samples CSV file")
    parser.add_argument("--pdb-path", type=str,
                       help="Path to input PDB file")
    
    args = parser.parse_args()
    
    try:
        # Initialize RFDiffusion with command line arguments
        rf_diffusion = RFDiffusion(
            output_dir=args.output_dir,
            pdb_path=args.pdb_path
        )
        
        # Run the pipeline
        results = rf_diffusion.run(
            samples_csv=args.samples_csv,
            dry_run=args.dry_run
        )
        
        print(f"Results: {results}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()