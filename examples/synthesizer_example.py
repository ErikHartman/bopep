#!/usr/bin/env python3
"""
Updated example script demonstrating how to use the Synthesizer orchestrator from the bopep package.
"""

import sys
import os
from pathlib import Path

# Add the bopep package to the path if needed
sys.path.append(str(Path(__file__).parent.parent))

from bopep import Synthesizer
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
    """Demonstrate basic usage of Synthesizer class."""
    print("=== Basic Synthesizer Usage Example ===")
    
    # Initialize Synthesizer with default settings
    synthesizer = Synthesizer(
        output_dir="example_output",
        # You would set these to your actual paths:
        # rfdiffusion_path="/path/to/RFdiffusion",
        # protein_mpnn_path="/path/to/ProteinMPNN",
        # pdb_path="/path/to/your/protein.pdb"
    )
    
    # Check configuration
    print("\nConfiguration status:")
    synthesizer.print_configuration()
    
    # Create sample data
    samples_csv = synthesizer.create_sample_data("example_output")
    
    print(f"\nCreated sample data: {samples_csv}")
    print("To run the complete pipeline, use:")
    print("results = synthesizer.run_complete_pipeline()")


def example_step_by_step():
    """Demonstrate step-by-step pipeline execution."""
    print("\n=== Step-by-Step Pipeline Example ===")
    
    # Initialize Synthesizer
    synthesizer = Synthesizer(output_dir="step_by_step_output")
    
    # Create sample data
    samples_csv = synthesizer.create_sample_data("step_by_step_output")
    
    print("\nStep 1: RFDiffusion only (dry-run)")
    try:
        rf_results = synthesizer.run_rfdiffusion_only(
            samples_csv=samples_csv,
            dry_run=True,  # Safe dry-run mode
            skip_existing=True
        )
        print(f"RFDiffusion dry-run results: {rf_results}")
    except Exception as e:
        print(f"RFDiffusion step error (expected without proper setup): {e}")
    
    print("\nStep 2: MPNN + FastRelax (would run if designs existed)")
    print("mpnn_results = synthesizer.run_mpnn_fastrelax_only()")
    
    print("\nStep 3: Complete pipeline")
    print("complete_results = synthesizer.run_complete_pipeline()")


def example_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Initialize with custom paths (adjust these to your system)
    synthesizer = Synthesizer(
        output_dir="custom_output",
        rfdiffusion_path="/path/to/RFdiffusion",  # Your RFdiffusion installation  
        protein_mpnn_path="/path/to/ProteinMPNN",  # Your ProteinMPNN installation
        pdb_path="/path/to/your/protein.pdb",  # Your target protein
        mpnn_env="conda run -n proteinmpnn python",  # Custom environment
        test_mode=True  # Enable test mode
    )
    
    print("Custom configuration:")
    synthesizer.print_configuration()
    
    # Get available pipeline steps
    steps = synthesizer.get_available_steps()
    print(f"\nAvailable pipeline steps: {steps}")
    
    # Example of running specific steps
    print("\nTo run specific steps:")
    print("results = synthesizer.run(pipeline_steps=['rfdiffusion'])")
    print("results = synthesizer.run(pipeline_steps=['mpnn_fastrelax'])")
    print("results = synthesizer.run(pipeline_steps=['complete'])")


def example_complete_pipeline():
    """Show how to run the complete pipeline with various options."""
    print("\n=== Complete Pipeline Examples ===")
    
    synthesizer = Synthesizer(output_dir="complete_pipeline_output")
    
    print("Example 1: Basic complete pipeline")
    print("results = synthesizer.run_complete_pipeline()")
    
    print("\nExample 2: Complete pipeline with custom parameters")
    print("results = synthesizer.run_complete_pipeline(")
    print("    samples_csv='my_samples.csv',")
    print("    rf_dry_run=False,")
    print("    temperature=0.2,")
    print("    relax_cycles=2,")
    print("    threads=8,")
    print("    limited_run=10  # Process only 10 designs for testing")
    print(")")
    
    print("\nExample 3: Using the main run() method")
    print("results = synthesizer.run(")
    print("    samples_csv='my_samples.csv',")
    print("    pipeline_steps=['complete'],")
    print("    temperature=0.1,")
    print("    threads=4")
    print(")")
    
    print("\nExpected results structure:")
    expected_results = {
        "pipeline_success": True,
        "elapsed_time": 1234.56,
        "rfdiffusion_results": {"successful_runs": 5, "failed_runs": 0},
        "mpnn_fastrelax_results": {"processed_pdbs": 5, "interface_dg_scores": 15},
        "final_output_csv": "/path/to/output.csv",
        "total_designs_generated": 5,
        "total_sequences_optimized": 5
    }
    print(f"  {expected_results}")


if __name__ == "__main__":
    print("Synthesizer Orchestrator Examples")
    print("=" * 50)
    
    print("The Synthesizer class provides a unified interface for:")
    print("1. RFDiffusion structure generation")
    print("2. ProteinMPNN sequence design") 
    print("3. PyRosetta FastRelax optimization")
    print("4. Complete pipeline orchestration")
    print("")
    
    # Run examples
    example_basic_usage()
    example_step_by_step()
    example_custom_configuration()
    example_complete_pipeline()
    
    print("\n=== Examples completed! ===")
    print("\nTo use Synthesizer in your own code:")
    print("from bopep import Synthesizer")
    print("synthesizer = Synthesizer(output_dir='my_output')")
    print("results = synthesizer.run_complete_pipeline()")
    print("")
    print("Requirements:")
    print("- RFDiffusion installation")
    print("- ProteinMPNN installation") 
    print("- PyRosetta installation")
    print("- Target protein PDB file")
    print("- Required Python packages: pandas, biopython, dotenv")
    print("")
    print("Environment variables (optional):")
    print("- RFDIFFUSION_PATH: Path to RFDiffusion installation")
    print("- PROTEIN_MPNN_PATH: Path to ProteinMPNN installation")
    print("- OUTPUT_DIR: Default output directory")
    print("- PLO1_PATH: Default target protein PDB file")
