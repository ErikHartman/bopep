#!/usr/bin/env python3
"""
Synthesis module for running the complete protein design pipeline.

This module provides the borf class which runs the following workflow:
1. RFDiffusion for structure generation
2. ProteinMPNN + FastRelax for sequence design and optimization
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from bopep.design.diffusion import RFDiffusion
from bopep.design.mpnn_fastrelax import MPNNFastRelax


class Borf:
    """
    A comprehensive orchestrator for protein-peptide design synthesis.
    
    This class coordinates the complete pipeline from initial peptide specifications
    through RFDiffusion structure generation to ProteinMPNN sequence design and
    PyRosetta FastRelax optimization.
    
    Examples
    --------
    >>> # Basic usage
    >>> borf = Borf(output_dir="my_synthesis")
    >>> results = borf.run(samples_csv="peptide_samples.csv")

    >>> # Custom configuration
    >>> borf = Borf(
    ...     output_dir="custom_output",
    ...     rfdiffusion_path="/path/to/RFdiffusion",
    ...     protein_mpnn_path="/path/to/ProteinMPNN",
    ...     pdb_path="/path/to/target.pdb"
    ... )
    >>> results = borf.run_complete_pipeline()
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        rfdiffusion_path: Optional[str] = None,
        protein_mpnn_path: Optional[str] = None,
        pdb_path: str = None,
        models_path: Optional[str] = None,
        rfd_env_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mpnn_chains: str = "A",
        mpnn_env: Optional[str] = None,
    ):
        """
        Initialize the Borf orchestrator.
        
        Parameters
        ----------
        output_dir : str, optional
            Base output directory. If None, uses OUTPUT_DIR from environment.
        rfdiffusion_path : str, optional
            Path to RFDiffusion installation. If None, uses RFDIFFUSION_PATH from environment.
        protein_mpnn_path : str, optional
            Path to ProteinMPNN installation. If None, uses PROTEIN_MPNN_PATH from environment.
        pdb_path : str
            Path to target protein PDB file. (Mandatory)
        models_path : str, optional
            Path to RFDiffusion models directory.
        rfd_env_path : str, optional
            Path to Python environment for RFDiffusion.
        checkpoint_path : str, optional
            Path to RFDiffusion checkpoint file.
        mpnn_chains : str, default "A"
            Chains to design with ProteinMPNN.
        mpnn_env : str, optional
            Python environment/executable for ProteinMPNN. If None, uses sys.executable.
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stderr
        )
        
        # Load environment variables
        
        # Set up base paths
        self.output_dir = Path(output_dir)
        
        # Store configuration
        if pdb_path is None:
            raise ValueError("pdb_path is a mandatory argument and must be provided.")
        self.config = {
            'output_dir': str(self.output_dir),
            'rfdiffusion_path': rfdiffusion_path,
            'protein_mpnn_path': protein_mpnn_path,
            'pdb_path': pdb_path,
            'models_path': models_path,
            'rfd_env_path': rfd_env_path,
            'checkpoint_path': checkpoint_path,
            'mpnn_chains': mpnn_chains,
            'mpnn_env': mpnn_env or sys.executable,
        }
        
        # Initialize components (lazy loading)
        self._rfdiffusion = None
        self._mpnn_fastrelax = None
        
        # Create base output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        logging.info(f"Borf initialized with output directory: {self.output_dir}")

    @property
    def rfdiffusion(self) -> RFDiffusion:
        """Lazy initialization of RFDiffusion component."""
        if self._rfdiffusion is None:
            if not self.config['rfdiffusion_path']:
                raise ValueError("RFDiffusion path not specified. Set rfdiffusion_path or RFDIFFUSION_PATH environment variable.")
            
            self._rfdiffusion = RFDiffusion(
                rfdiffusion_path=self.config['rfdiffusion_path'],
                output_dir=str(self.output_dir),
                pdb_path=self.config['pdb_path'],
                models_path=self.config['models_path'],
                python_env_path=self.config['rfd_env_path'],
                checkpoint_path=self.config['checkpoint_path']
            )
        return self._rfdiffusion
    
    @property
    def mpnn_fastrelax(self) -> MPNNFastRelax:
        """Lazy initialization of MPNNFastRelax component."""
        if self._mpnn_fastrelax is None:
            self._mpnn_fastrelax = MPNNFastRelax(
                output_dir=str(self.output_dir),
                designs_dir=str(self.output_dir / "designs"),
                protein_mpnn_path=self.config['protein_mpnn_path'],
                mpnn_chains=self.config['mpnn_chains'],
                mpnn_env=self.config['mpnn_env']
            )
        return self._mpnn_fastrelax
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate the configuration and check for required dependencies.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary indicating which components are properly configured.
        """
        validation = {
            'rfdiffusion_configured': bool(self.config['rfdiffusion_path']),
            'protein_mpnn_configured': bool(self.config['protein_mpnn_path']),
            'pdb_file_exists': bool(self.config['pdb_path'] and Path(self.config['pdb_path']).exists()),
            'output_dir_writable': self.output_dir.exists() and os.access(self.output_dir, os.W_OK)
        }
        
        # Check RFDiffusion installation
        if validation['rfdiffusion_configured']:
            rf_path = Path(self.config['rfdiffusion_path'])
            validation['rfdiffusion_exists'] = rf_path.exists()
            validation['rfdiffusion_script_exists'] = (rf_path / "scripts" / "run_inference.py").exists()
        else:
            validation['rfdiffusion_exists'] = False
            validation['rfdiffusion_script_exists'] = False
        
        # Check ProteinMPNN installation
        if validation['protein_mpnn_configured']:
            mpnn_path = Path(self.config['protein_mpnn_path'])
            validation['protein_mpnn_exists'] = mpnn_path.exists()
            validation['protein_mpnn_script_exists'] = (mpnn_path / "protein_mpnn_run.py").exists()
        else:
            validation['protein_mpnn_exists'] = False
            validation['protein_mpnn_script_exists'] = False
        
        return validation
    
    def print_configuration(self):
        """Print the current configuration and validation status."""
        print("=== Borf Configuration ===")
        for key, value in self.config.items():
            if value is None:
                value = "Not set"
            print(f"{key:20}: {value}")
        
        print("\n=== Configuration Validation ===")
        validation = self.validate_configuration()
        for key, status in validation.items():
            status_str = "✓ OK" if status else "✗ FAILED"
            print(f"{key:25}: {status_str}")
    
    def create_sample_data(
        self, 
        output_path: Optional[str] = None,
        sample_data: Optional[List[Dict]] = None
    ) -> str:
        """
        Create sample peptide data for the pipeline.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the samples CSV. If None, uses default location.
        sample_data : List[Dict], optional
            Custom sample data. If None, creates default examples.
        
        Returns
        -------
        str
            Path to the created samples CSV file.
        """
        if output_path is None:
            samples_dir = self.output_dir / "samples"
            samples_dir.mkdir(exist_ok=True, parents=True)
            output_path = samples_dir / "peptide_samples.csv"
        
        if sample_data is None:
            # Create default sample data
            sample_data = [
                {
                    'sample_id': 1,
                    'length': 10,
                    'hotspots': 'A400,A403,A407'
                },
                {
                    'sample_id': 2,
                    'length': 12,
                    'hotspots': 'A402,A405,A408'
                },
                {
                    'sample_id': 3,
                    'length': 8,
                    'hotspots': 'A401,A404,A406'
                },
                {
                    'sample_id': 4,
                    'length': 15,
                    'hotspots': 'A399,A402,A405,A408'
                },
                {
                    'sample_id': 5,
                    'length': 11,
                    'hotspots': 'A400,A403,A407,A410'
                }
            ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Created sample data with {len(sample_data)} samples at: {output_path}")
        return str(output_path)
    
    def run_rfdiffusion_only(
        self,
        samples_csv: Optional[str] = None,
        dry_run: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Run only the RFDiffusion step of the pipeline.
        
        Parameters
        ----------
        samples_csv : str, optional
            Path to samples CSV file. If None, creates default samples.
        dry_run : bool, default False
            If True, print commands without executing them.
        skip_existing : bool, default True
            If True, skip samples that have already been processed.
        
        Returns
        -------
        Dict[str, Any]
            Results from RFDiffusion run.
        """
        if samples_csv is None:
            samples_csv = self.create_sample_data()
        
        logging.info("Running RFDiffusion step...")
        results = self.rfdiffusion.run(
            samples_csv=samples_csv,
            dry_run=dry_run,
            skip_existing=skip_existing
        )
        
        logging.info(f"RFDiffusion completed: {results['successful_runs']} successful, {results['failed_runs']} failed")
        return results
    
    def run_mpnn_fastrelax_only(
        self,
        designs_dir: Optional[str] = None,
        temperature: float = 0.1,
        relax_cycles: int = 1,
        threads: int = 4,
        limited_run: int = 0,
        output_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run only the ProteinMPNN + FastRelax step of the pipeline.
        
        Parameters
        ----------
        designs_dir : str, optional
            Directory containing RFDiffusion design PDB files.
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
            Results from MPNN + FastRelax run.
        """
        logging.info("Running ProteinMPNN + FastRelax step...")
        results = self.mpnn_fastrelax.run(
            designs_dir=designs_dir,
            temperature=temperature,
            relax_cycles=relax_cycles,
            threads=threads,
            limited_run=limited_run,
            output_csv=output_csv
        )
        
        logging.info(f"MPNN + FastRelax completed: {results['processed_pdbs']} PDBs processed")
        return results
    
    def run_complete_pipeline(
        self,
        samples_csv: Optional[str] = None,
        # RFDiffusion parameters
        rf_dry_run: bool = False,
        rf_skip_existing: bool = True,
        # MPNN + FastRelax parameters
        temperature: float = 0.1,
        relax_cycles: int = 1,
        threads: int = 4,
        limited_run: int = 0,
        output_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete synthesis pipeline: RFDiffusion → ProteinMPNN + FastRelax.
        
        Parameters
        ----------
        samples_csv : str, optional
            Path to samples CSV file. If None, creates default samples.
        rf_dry_run : bool, default False
            If True, run RFDiffusion in dry-run mode.
        rf_skip_existing : bool, default True
            If True, skip existing RFDiffusion outputs.
        temperature : float, default 0.1
            ProteinMPNN sampling temperature.
        relax_cycles : int, default 1
            Number of MPNN + FastRelax cycles.
        threads : int, default 4
            Number of parallel threads for MPNN + FastRelax.
        limited_run : int, default 0
            Limit number of PDBs for testing (0 = no limit).
        output_csv : str, optional
            Final output CSV path.
        
        Returns
        -------
        Dict[str, Any]
            Combined results from both pipeline steps.
        """
        logging.info("=== Starting Complete Synthesis Pipeline ===")
        start_time = time.time()
        
        # Validate configuration
        validation = self.validate_configuration()
        if not all([validation['rfdiffusion_configured'], validation['protein_mpnn_configured']]):
            raise ValueError("Pipeline not properly configured. Use print_configuration() to check status.")
        
        if samples_csv is None:
            samples_csv = self.create_sample_data()
        
        # Step 1: RFDiffusion
        logging.info("Step 1: Running RFDiffusion...")
        rf_results = self.run_rfdiffusion_only(
            samples_csv=samples_csv,
            dry_run=rf_dry_run,
            skip_existing=rf_skip_existing
        )
        # Step 2: ProteinMPNN + FastRelax
        logging.info("Step 2: Running ProteinMPNN + FastRelax...")
        mpnn_results = self.run_mpnn_fastrelax_only(
            designs_dir=str(self.output_dir / "designs"),
            temperature=temperature,
            relax_cycles=relax_cycles,
            threads=threads,
            limited_run=limited_run,
            output_csv=output_csv
        )
        
        # Combine results
        elapsed_time = time.time() - start_time
        combined_results = {
            'pipeline_success': True,
            'elapsed_time': elapsed_time,
            'rfdiffusion_results': rf_results,
            'mpnn_fastrelax_results': mpnn_results,
            'final_output_csv': mpnn_results.get('output_csv'),
            'total_designs_generated': rf_results.get('successful_runs', 0),
            'total_sequences_optimized': mpnn_results.get('processed_pdbs', 0),
            'interface_dg_scores': mpnn_results.get('interface_dg_scores', 0)
        }
        
        logging.info(f"=== Pipeline Completed in {elapsed_time:.2f}s ===")
        logging.info(f"Designs generated: {combined_results['total_designs_generated']}")
        logging.info(f"Sequences optimized: {combined_results['total_sequences_optimized']}")
        logging.info(f"Final output: {combined_results['final_output_csv']}")
        
        return combined_results
    
    def run(
        self,
        samples_csv: Optional[str] = None,
        pipeline_steps: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main method to run the synthesis pipeline with flexible step selection.
        
        Parameters
        ----------
        samples_csv : str, optional
            Path to samples CSV file.
        pipeline_steps : List[str], optional
            Steps to run. Options: ['rfdiffusion', 'mpnn_fastrelax', 'complete'].
            If None, runs complete pipeline.
        **kwargs
            Additional parameters passed to specific pipeline methods.
        
        Returns
        -------
        Dict[str, Any]
            Results from the requested pipeline steps.
        """
        if pipeline_steps is None:
            pipeline_steps = ['complete']
        
        if 'complete' in pipeline_steps:
            return self.run_complete_pipeline(samples_csv=samples_csv, **kwargs)
        
        results = {}
        
        if 'rfdiffusion' in pipeline_steps:
            results['rfdiffusion'] = self.run_rfdiffusion_only(samples_csv=samples_csv, **kwargs)
        
        if 'mpnn_fastrelax' in pipeline_steps:
            results['mpnn_fastrelax'] = self.run_mpnn_fastrelax_only(**kwargs)
        
        return results
    
    def get_available_steps(self) -> List[str]:
        """Get list of available pipeline steps."""
        return ['rfdiffusion', 'mpnn_fastrelax', 'complete']
    
    def cleanup_intermediate_files(self, keep_logs: bool = True):
        """
        Clean up intermediate files to save space.
        
        Parameters
        ----------
        keep_logs : bool, default True
            If True, keep log files for debugging.
        """
        logging.info("Cleaning up intermediate files...")
        
        # Add cleanup logic here if needed
        # For now, just log the action
        logging.info("Cleanup completed")


def main():
    """Command-line interface for Borf."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run complete protein-peptide synthesis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--samples-csv", help="Path to samples CSV file")
    parser.add_argument("--config", action="store_true", help="Print configuration and exit")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    
    # Pipeline control
    parser.add_argument("--steps", nargs="+", choices=["rfdiffusion", "mpnn_fastrelax", "complete"],
                       default=["complete"], help="Pipeline steps to run")
    
    # RFDiffusion options
    parser.add_argument("--rfdiffusion-path", help="Path to RFDiffusion installation")
    parser.add_argument("--pdb-path", help="Path to target protein PDB file")
    parser.add_argument("--rf-dry-run", action="store_true", help="Run RFDiffusion in dry-run mode")
    
    # MPNN + FastRelax options
    parser.add_argument("--protein-mpnn-path", help="Path to ProteinMPNN installation")
    parser.add_argument("--temperature", type=float, default=0.1, help="ProteinMPNN sampling temperature")
    parser.add_argument("--relax-cycles", type=int, default=1, help="Number of MPNN + FastRelax cycles")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads")
    parser.add_argument("--limited-run", type=int, default=0, help="Limit number of PDBs to process")
    
    args = parser.parse_args()
    
    try:
        # Initialize Borf
        borf = Borf(
            output_dir=args.output_dir,
            rfdiffusion_path=args.rfdiffusion_path,
            protein_mpnn_path=args.protein_mpnn_path,
            pdb_path=args.pdb_path,
        )
        
        # Print configuration if requested
        if args.config:
            borf.print_configuration()
            return
        
        # Run pipeline
        results = borf.run(
            samples_csv=args.samples_csv,
            pipeline_steps=args.steps,
            rf_dry_run=args.rf_dry_run,
            temperature=args.temperature,
            relax_cycles=args.relax_cycles,
            threads=args.threads,
            limited_run=args.limited_run
        )
        
        print(f"\n=== Results ===")
        print(f"Pipeline completed successfully!")
        if 'pipeline_success' in results:
            print(f"Total time: {results['elapsed_time']:.2f}s")
            print(f"Final output: {results.get('final_output_csv', 'N/A')}")
        
    except Exception as e:
        logging.error(f"Error in synthesis pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
