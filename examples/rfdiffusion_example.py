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


if __name__ == "__main__":
    print("RFDiffusion Class Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_configuration()
        example_manual_processing()
        
        print("\n=== Examples completed successfully! ===")
        print("\nTo use RFDiffusion in your own code:")
        print("from bopep import RFDiffusion")
        print("rf_diffusion = RFDiffusion(output_dir='my_output')")
        print("results = rf_diffusion.run(samples_csv='my_samples.csv')")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: This is expected if RFdiffusion is not installed or paths are not configured.")
