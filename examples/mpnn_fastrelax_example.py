#!/usr/bin/env python3
"""
Example script demonstrating how to use the MPNNFastRelax class from the bopep package.
"""

import sys
import os
from pathlib import Path

# Add the bopep package to the path if needed
sys.path.append(str(Path(__file__).parent.parent))

from bopep import MPNNFastRelax
import pandas as pd


def create_sample_design_pdbs(output_dir: str = "example_output") -> str:
    """
    Create sample design directory structure for demonstration.
    Note: This creates empty PDB files for demo purposes.
    
    Parameters
    ----------
    output_dir : str
        Directory to save sample data
        
    Returns
    -------
    str
        Path to the created designs directory
    """
    # Create example directory structure
    output_path = Path(output_dir)
    designs_dir = output_path / "designs"
    
    # Create sample directories and empty PDB files
    for sample_id in [1, 2, 3, 4, 5]:
        sample_dir = designs_dir / f"sample_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create empty PDB file (in real use, these would be RFDiffusion outputs)
        pdb_file = sample_dir / "design_0.pdb"
        with open(pdb_file, 'w') as f:
            f.write("# This is a placeholder PDB file for demonstration\n")
            f.write("# In real usage, this would be an RFDiffusion output\n")
    
    print(f"Created sample design structure at: {designs_dir}")
    return str(designs_dir)


def example_basic_usage():
    """Demonstrate basic usage of MPNNFastRelax class."""
    print("=== Basic MPNNFastRelax Usage Example ===")
    
    try:
        # Initialize MPNNFastRelax with default settings
        mpnn_fastrelax = MPNNFastRelax(
            output_dir="example_output",
            test_mode=True  # Use test mode for safer demonstration
        )
        
        print(f"Initialized MPNNFastRelax")
        print(f"Output directory: {mpnn_fastrelax.output_dir}")
        print(f"Sequence output directory: {mpnn_fastrelax.sequence_output_dir}")
        print(f"PyRosetta available: {mpnn_fastrelax.fast_relax is not None}")
        
        # Note: We won't run the full pipeline as it requires specific dependencies
        print("\nTo run the full pipeline, you would call:")
        print("results = mpnn_fastrelax.run()")
        
    except Exception as e:
        print(f"Note: This error is expected without proper dependencies: {e}")


def example_custom_configuration():
    """Demonstrate MPNNFastRelax with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    try:
        # Initialize with custom paths (adjust these to your system)
        mpnn_fastrelax = MPNNFastRelax(
            output_dir="custom_output",
            designs_dir="/path/to/your/designs",  # Your designs directory
            protein_mpnn_path="/path/to/ProteinMPNN",  # Your ProteinMPNN installation
            mpnn_chains="A",  # Design only chain A
            test_mode=True
        )
        
        print(f"Custom configuration:")
        print(f"  Designs directory: {mpnn_fastrelax.designs_dir}")
        print(f"  ProteinMPNN path: {mpnn_fastrelax.protein_mpnn_path}")
        print(f"  MPNN chains: {mpnn_fastrelax.mpnn_chains}")
        
        # Example of finding design PDBs (would fail with demo data)
        try:
            # This would normally find your RFDiffusion PDB files
            pdbs = mpnn_fastrelax.find_design_pdbs()
            print(f"Found {len(pdbs)} design PDB files")
        except FileNotFoundError:
            print("No design PDB files found (expected for demo)")
        
    except Exception as e:
        print(f"Configuration example: {e}")


def example_pipeline_components():
    """Demonstrate individual pipeline components."""
    print("\n=== Pipeline Components Example ===")
    
    try:
        mpnn_fastrelax = MPNNFastRelax(
            output_dir="components_output",
            test_mode=True
        )
        
        # Example of extracting peptide information from PDB
        # (Would work with real PDB files)
        demo_pdb = "example_design.pdb"
        print(f"Example: Extract peptide info from {demo_pdb}")
        print("peptide_info = mpnn_fastrelax.extract_peptide_from_pdb(pdb_file)")
        
        # Example of running ProteinMPNN
        print("\nExample: Run ProteinMPNN on PDB files")
        print("fastas_dir = mpnn_fastrelax.run_proteinmpnn(pdb_files, temperature=0.1)")
        
        # Example of creating output CSV
        print("\nExample: Create comprehensive output CSV")
        print("mpnn_fastrelax.create_output_csv(fastas_root, output_path)")
        
        print("\nFull pipeline parameters:")
        pipeline_params = {
            "temperature": 0.1,
            "relax_cycles": 1,
            "threads": 4,
            "limited_run": 0
        }
        print(f"  {pipeline_params}")
        
    except Exception as e:
        print(f"Components example: {e}")


def example_run_parameters():
    """Show different ways to run the pipeline."""
    print("\n=== Run Parameters Example ===")
    
    try:
        mpnn_fastrelax = MPNNFastRelax(output_dir="run_example", test_mode=True)
        
        print("Example 1: Basic run")
        print("results = mpnn_fastrelax.run()")
        
        print("\nExample 2: Custom parameters")
        print("results = mpnn_fastrelax.run(")
        print("    designs_dir='/path/to/designs',")
        print("    temperature=0.2,")
        print("    relax_cycles=2,")
        print("    threads=8,")
        print("    limited_run=10,  # Process only 10 PDBs")
        print("    output_csv='/path/to/output.csv'")
        print(")")
        
        print("\nExample 3: Test run with limited processing")
        print("results = mpnn_fastrelax.run(limited_run=5)")
        
        print("\nExpected results structure:")
        expected_results = {
            "success": True,
            "processed_pdbs": 10,
            "output_csv": "/path/to/output.csv",
            "interface_dg_scores": 25,
            "sequence_output_dir": "/path/to/output"
        }
        print(f"  {expected_results}")
        
    except Exception as e:
        print(f"Run parameters example: {e}")


if __name__ == "__main__":
    print("MPNNFastRelax Class Examples")
    print("=" * 50)
    
    print("This class provides a complete pipeline for:")
    print("1. Running ProteinMPNN on RFDiffusion designs")
    print("2. Threading sequences onto structures") 
    print("3. Running PyRosetta FastRelax optimization")
    print("4. Calculating interface binding energies")
    print("5. Generating comprehensive output CSV files")
    print("")
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_pipeline_components()
    example_run_parameters()
    
    print("\n=== Examples completed! ===")
    print("\nTo use MPNNFastRelax in your own code:")
    print("from bopep import MPNNFastRelax")
    print("mpnn_fastrelax = MPNNFastRelax(output_dir='my_output')")
    print("results = mpnn_fastrelax.run(designs_dir='my_designs')")
    print("")
    print("Requirements:")
    print("- RFDiffusion design PDB files")
    print("- ProteinMPNN installation")
    print("- PyRosetta installation")
    print("- Appropriate XML configuration file")
    print("- Required Python packages: pandas, biopython, dotenv")
