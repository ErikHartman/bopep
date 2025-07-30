#!/usr/bin/env python3
"""
Example script demonstrating how to use the RFDiffusion class from the bopep package.
"""

import sys
import os
from pathlib import Path

# Add the bopep package to the path if needed
sys.path.append(str(Path(__file__).parent.parent))

from bopep import RFDiffusion
import pandas as pd


def create_sample_data(output_dir: str = "example_output") -> str:
    """
    Create sample data for demonstration.
    
    Parameters
    ----------
    output_dir : str
        Directory to save sample data
        
    Returns
    -------
    str
        Path to the created samples CSV file
    """
    # Create example directory structure
    output_path = Path(output_dir)
    samples_dir = output_path / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample peptide data
    sample_data = {
        'sample_id': [1, 2, 3, 4, 5],
        'length': [10, 12, 8, 15, 11],
        'hotspots': [
            'A400,A403,A407',
            'A402,A405,A408',
            'A401,A404,A406',
            'A399,A402,A405,A408',
            'A400,A403,A407,A410'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    csv_path = samples_dir / "peptide_samples.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Created sample data at: {csv_path}")
    return str(csv_path)


def example_basic_usage():
    """Demonstrate basic usage of RFDiffusion class."""
    print("=== Basic RFDiffusion Usage Example ===")
    
    # Create sample data
    samples_csv = create_sample_data("example_output")
    
    # Initialize RFDiffusion with custom settings
    rf_diffusion = RFDiffusion(
        output_dir="example_output",
        # You would set these to your actual paths:
        # pdb_path="/path/to/your/protein.pdb",
        # rfdiffusion_path="/path/to/RFdiffusion",
        # models_path="/path/to/RFdiffusion/models"
    )
    
    # Run in dry-run mode to see what would be executed
    print("\nRunning in dry-run mode...")
    results = rf_diffusion.run(
        samples_csv=samples_csv,
        dry_run=True,
        skip_existing=True
    )
    
    print(f"Dry-run results: {results}")


def example_custom_configuration():
    """Demonstrate RFDiffusion with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create sample data
    samples_csv = create_sample_data("custom_output")
    
    # Initialize with custom paths (adjust these to your system)
    rf_diffusion = RFDiffusion(
        output_dir="custom_output",
        pdb_path="/custom/path/to/protein.pdb",  # Your PDB file
        rfdiffusion_path="/custom/path/to/RFdiffusion",  # Your RFdiffusion installation
        models_path="/custom/path/to/models",  # Your models directory
        python_env_path="/custom/path/to/python",  # Your Python environment
        checkpoint_path="/custom/path/to/checkpoint.pt"  # Your checkpoint file
    )
    
    # Load samples and check available GPUs
    samples_df = rf_diffusion.load_samples(samples_csv)
    gpus = rf_diffusion.get_available_gpus()
    
    print(f"Loaded {len(samples_df)} samples")
    print(f"Available GPUs: {gpus}")
    
    # Process samples (in dry-run mode for safety)
    successful, failed = rf_diffusion.process_samples(
        samples_df=samples_df,
        gpus=gpus,
        dry_run=True,
        skip_existing=True
    )
    
    print(f"Would process: {successful} successful, {failed} failed")


def example_manual_processing():
    """Demonstrate manual step-by-step processing."""
    print("\n=== Manual Processing Example ===")
    
    # Create sample data
    samples_csv = create_sample_data("manual_output")
    
    # Initialize RFDiffusion
    rf_diffusion = RFDiffusion(output_dir="manual_output")
    
    # Step 1: Load samples
    samples_df = rf_diffusion.load_samples(samples_csv)
    print(f"Loaded samples:\n{samples_df}")
    
    # Step 2: Check available GPUs
    gpus = rf_diffusion.get_available_gpus()
    print(f"Available GPUs: {gpus}")
    
    # Step 3: Process subset of samples
    subset_df = samples_df.head(2)  # Process only first 2 samples
    successful, failed = rf_diffusion.process_samples(
        samples_df=subset_df,
        gpus=gpus[:1] if gpus else [0],  # Use only first GPU
        dry_run=True,
        skip_existing=False
    )
    
    print(f"Processed subset: {successful} successful, {failed} failed")



# === Synthesiser Orchestrator Example ===
"""
Example script demonstrating how to use the Synthesiser orchestrator from the bopep package.
"""

from bopep import Synthesiser

def run_full_pipeline_with_synthesiser():
    """
    Run all steps of the process using the Synthesiser orchestrator.
    """
    print("\n=== Full Pipeline with Synthesiser ===")
    
    # Initialize the Synthesiser orchestrator
    synthesiser = Synthesiser(
        output_dir="pipeline_output",
        # You would set these to your actual paths:
        # rfdiffusion_path="/path/to/RFdiffusion",
        # protein_mpnn_path="/path/to/ProteinMPNN", 
        # pdb_path="/path/to/your/protein.pdb"
    )
    
    # Check configuration
    print("Configuration status:")
    synthesiser.print_configuration()
    
    # Create sample data
    samples_csv = synthesiser.create_sample_data("pipeline_output")
    
    # Option 1: Run complete pipeline in one call
    print("\nRunning complete pipeline...")
    try:
        results = synthesiser.run_complete_pipeline(
            samples_csv=samples_csv,
            rf_dry_run=True,  # Safe dry-run mode for demo
            temperature=0.1,
            relax_cycles=1,
            threads=4,
            limited_run=5  # Process only 5 designs for testing
        )
        print(f"Pipeline results: {results}")
    except Exception as e:
        print(f"Pipeline error (expected without proper setup): {e}")
    
    # Option 2: Run individual steps
    print("\nRunning individual steps...")
    try:
        # Step 1: RFDiffusion only
        rf_results = synthesiser.run_rfdiffusion_only(
            samples_csv=samples_csv,
            dry_run=True
        )
        print(f"RFDiffusion results: {rf_results}")
        
        # Step 2: MPNN + FastRelax (would run if designs existed)
        print("MPNN + FastRelax would run here if designs existed")
        
    except Exception as e:
        print(f"Individual steps error (expected): {e}")
    
    print("\n=== Synthesiser pipeline demonstration completed! ===")

if __name__ == "__main__":
    print("RFDiffusion & MPNNFastRelax Class Examples")
    print("=" * 50)
    try:
        # Run original RFDiffusion examples
        example_basic_usage()
        example_custom_configuration()
        example_manual_processing()
        # Run integrated full pipeline example
        run_full_pipeline_with_synthesiser()
        # Run new Synthesiser orchestrator example
        run_full_pipeline_with_synthesiser()
        print("\n=== All examples completed successfully! ===")
        print("\nTo use RFDiffusion and MPNNFastRelax individually:")
        print("from bopep import RFDiffusion, MPNNFastRelax")
        print("rf_diffusion = RFDiffusion(output_dir='my_output')")
        print("results = rf_diffusion.run(samples_csv='my_samples.csv')")
        print("mpnn_fastrelax = MPNNFastRelax(output_dir='my_output', designs_dir='my_designs')")
        print("results = mpnn_fastrelax.run()")
        print("\nTo use the Synthesiser orchestrator:")
        print("from bopep import Synthesiser")
        print("synthesiser = Synthesiser(output_dir='my_output')")
        print("results = synthesiser.run_complete_pipeline()")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: This is expected if dependencies are not installed or paths are not configured.")
