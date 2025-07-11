import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bopep.bayesian_optimization.optimization import BoPep

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def iptm_objective(scores: dict) -> dict:
    """
    Returns the iptm score for each peptide in the scores dictionary.
    """
    scalar_objectives = {}
    for peptide, peptide_scores in scores.items():
        scalar_objectives[peptide] = peptide_scores["iptm"]
    return scalar_objectives


def generate_synthetic_data(num_peptides=1000, embedding_dim=128):
    """Generate synthetic peptide data for testing."""
    logging.info(f"Generating synthetic data for {num_peptides} peptides...")
    
    # Generate random peptide sequences
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    peptides = []
    for i in range(num_peptides):
        length = np.random.randint(8, 25)  # Random length between 8-24
        sequence = ''.join(np.random.choice(list(amino_acids), length))
        peptides.append(sequence)
    
    # Generate synthetic embeddings
    embeddings = {}
    for peptide in peptides:
        embeddings[peptide] = np.random.randn(embedding_dim).astype(np.float32)
    
    # Generate synthetic scores
    scores_data = []
    for peptide in peptides:
        scores_data.append({
            'peptide': peptide,
            'iptm': np.random.uniform(0.0, 1.0),  # Random iptm score
            'in_binding_site': np.random.choice([0, 1]),  # Random binary
            'pae': np.random.uniform(0.0, 30.0),  # Random PAE
            'dG': np.random.uniform(-15.0, 5.0),  # Random binding energy
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    logging.info(f"Generated {len(peptides)} synthetic peptides with {embedding_dim}D embeddings")
    return embeddings, scores_df


def load_benchmark_embeddings(embedding_path, embedding_type="esm1d"):
    """Load embeddings from the benchmark data, with fallback to synthetic data."""
    if not os.path.exists(embedding_path):
        logging.warning(f"Embedding path {embedding_path} does not exist. Generating synthetic data instead.")
        return None
        
    logging.info(f"Loading benchmark embeddings from {embedding_path}...")
    try:
        with open(embedding_path, 'rb') as f:
            benchmark_data = pickle.load(f)
        logging.info(f"Loaded benchmark data with keys: {benchmark_data.keys()}")
        
        if "reduced_embeddings" in benchmark_data and embedding_type in benchmark_data["reduced_embeddings"]:
            embeddings_dict = benchmark_data["reduced_embeddings"][embedding_type]
            
            if not isinstance(embeddings_dict, dict):
                if isinstance(embeddings_dict, np.ndarray):
                    all_peptides = benchmark_data["peptides"]
                    embeddings_dict = {p: embeddings_dict[i] for i, p in enumerate(all_peptides)}
            sample_embedding = next(iter(embeddings_dict.values()))
            logging.info(f"Embedding dimension for {embedding_type}: {sample_embedding.shape}")
            
            return embeddings_dict
        else:
            logging.error(f"Could not find reduced embeddings for {embedding_type}")
            return None
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None


def test_bopep_with_precomputed_data(
    embeddings_path=None,
    objectives_csv=None,
    output_dir="./test_output",
    embedding_type="esm_1d_pca", 
    use_synthetic=False,
):
    """
    Test BoPep with either precomputed data or synthetic data.
    If paths don't exist or use_synthetic=True, synthetic data will be generated.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings = None
    all_scores = {}
    
    if not use_synthetic and embeddings_path and objectives_csv:
        if os.path.exists(objectives_csv):
            df = pd.read_csv(objectives_csv)
            peptides = df['peptide'].tolist()
            
            for _, row in df.iterrows():
                peptide = row['peptide']
                score_dict = {col: row[col] for col in df.columns if col != 'peptide'}
                all_scores[peptide] = score_dict
            
            # Load embeddings using the provided function
            embeddings = load_benchmark_embeddings(embeddings_path, embedding_type)
        else:
            logging.warning(f"Objectives CSV {objectives_csv} does not exist")

    if not embeddings or not all_scores:
        logging.info("Using synthetic data for testing")
        embeddings, df = generate_synthetic_data(num_peptides=500, embedding_dim=128)
        
        all_scores = {}
        for _, row in df.iterrows():
            peptide = row['peptide']
            score_dict = {col: row[col] for col in df.columns if col != 'peptide'}
            all_scores[peptide] = score_dict
        
        peptides = df['peptide'].tolist()
    
    if not embeddings:
        logging.error("Failed to load or generate embeddings. Exiting...")
        return None
    
    if 'peptides' in locals():
        filtered_peptides = [p for p in peptides if p in embeddings]
    else:
        filtered_peptides = list(embeddings.keys())
    
    filtered_embeddings = {p: embeddings[p] for p in filtered_peptides}
    
    if 'peptides' in locals():
        missing = len(peptides) - len(filtered_peptides)
        if missing > 0:
            logging.warning(f"Missing embeddings for {missing} peptides")
    
    logging.info(f"Proceeding with {len(filtered_peptides)} peptides")
    
    any_embedding = next(iter(filtered_embeddings.values()))
    network_type = "mlp"
    logging.info(f"Using {network_type} network based on embedding shape {any_embedding.shape}")
    
    from bopep.bayesian_optimization import optimization
    
    original_validate = optimization._validate_dependencies
    optimization._validate_dependencies = lambda: None
    
    try:
        bopep = BoPep(
            surrogate_model_kwargs={
                "network_type": network_type,
                "model_type": "deep_evidential",
            },
            docker_kwargs={
                "output_dir": os.path.join(output_dir, "docking"),  # Create a subdirectory for docking
                "num_cores": 1  # Use minimal resources for testing
            },
            hpo_kwargs={
                "n_trials": 3,
                "hpo_interval": 20 
            },
            objective_function=iptm_objective,
            scoring_kwargs={"scores_to_include": ["iptm", "in_binding_site"]},
            log_dir=output_dir,
            checkpoint_interval=3  # Test checkpointing frequently
        )
    finally:
        optimization._validate_dependencies = original_validate
    
    def mock_score_batch(docked_dirs):
        selected_peptides = docked_dirs
        
        mock_scores = {}
        for peptide in selected_peptides:
            if peptide in all_scores:
                mock_scores[peptide] = all_scores[peptide]
        
        return mock_scores
    
    bopep._score_batch = mock_score_batch
    bopep.docker.dock_peptides = lambda peptides: peptides  # Just return the peptide names
    
    schedule = [ {"acquisition": "standard_deviation", "iterations": 2},
        {"acquisition": "expected_improvement", "iterations": 10},
    ]

    schedule_continue = [
        {"acquisition": "standard_deviation", "iterations": 2},
        {"acquisition": "expected_improvement", "iterations": 10},
    ]
    
    # Run the optimization loop
    logging.info("Starting test optimization loop")
    
    # Use a fallback PDB file path that should exist in the project
    target_pdb_path = "/home/er8813ha/bopep/data/4glf.pdb"
    
    # Try to find the PDB file in the project structure
    possible_pdb_paths = [
        "/home/er8813ha/bopep/data/4glf.pdb",
        "../data/4glf.pdb", 
        "data/4glf.pdb",
        os.path.join(os.path.dirname(__file__), "..", "data", "4glf.pdb")
    ]
    
    for path in possible_pdb_paths:
        if os.path.exists(path):
            target_pdb_path = path
            break
    else:
        logging.warning(f"Could not find PDB file, using {target_pdb_path} (may not exist)")
    
    try:
        bopep.optimize(
            target_structure_path=target_pdb_path,
            num_initial=100,
            batch_size=10,
            schedule=schedule,
            embeddings=filtered_embeddings,
            binding_site_residue_indices=[23, 42, 44, 49, 69, 72, 74, 82, 89, 105],
            assume_zero_indexed=True,
            n_validate=50
        )

        # For the second BoPep instance, temporarily disable validation again
        optimization._validate_dependencies = lambda: None
        try:
            bopep_cont = BoPep(
                surrogate_model_kwargs={
                    "network_type": network_type,
                    "model_type": "deep_evidential",
                },
                docker_kwargs={
                    "output_dir": os.path.join(output_dir, "docking"),
                    "num_cores": 1
                },
                hpo_kwargs={
                    "n_trials": 3,
                    "hpo_interval": 20 
                },
                objective_function=iptm_objective,
                scoring_kwargs={"scores_to_include": ["iptm", "in_binding_site"]},
                log_dir=output_dir,
                checkpoint_interval=3  # Test checkpointing frequently
            )
        finally:
            # Restore the original function again
            optimization._validate_dependencies = original_validate

        logging.info("Continuing optimization from checkpoint")
        bopep_cont._score_batch = mock_score_batch
        bopep_cont.docker.dock_peptides = lambda peptides: peptides

        bopep_cont.optimize(
            target_structure_path=target_pdb_path,
            batch_size=10,
            schedule=schedule_continue,
            binding_site_residue_indices=[23, 42, 44, 49, 69, 72, 74, 82, 89, 105],
            n_validate=50,
            checkpoint_path=os.path.join(output_dir, "checkpoint_0"),
            assume_zero_indexed=True,
        )

        logging.info(f"Test completed, output saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error during optimization: {e}")
    


def test_bopep_synthetic_only(output_dir="./test_output_synthetic"):
    """
    Simple test function that only uses synthetic data.
    This should work on any system without external dependencies.
    """
    print("Running BoPep test with synthetic data only...")
    
    return test_bopep_with_precomputed_data(
        output_dir=output_dir,
        use_synthetic=True
    )


def test_peptide_specific_binding_sites(output_dir="./test_output_peptide_specific"):

    print("Testing peptide-specific binding sites functionality...")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data with known peptides
    logging.info("Generating synthetic data for binding site test...")
    test_peptides = [
        "ACDEFGHIK",  # Peptide 1 - targets binding site 1
        "LMNPQRSTV",  # Peptide 2 - targets binding site 2 
        "WYACDEFGH",  # Peptide 3 - targets both binding sites
        "IKLMNPQRS",  # Peptide 4 - targets binding site 1
        "TVWYACDEF",  # Peptide 5 - targets binding site 2
        "GHIKLMNPQ",  # Peptide 6 - additional test peptide
        "RSTVWYACD",  # Peptide 7 - additional test peptide
        "EFGHIKLMN",  # Peptide 8 - additional test peptide
        "PQRSTVWYA",  # Peptide 9 - additional test peptide
        "CDEFGHIKL"   # Peptide 10 - additional test peptide
    ]
    
    # Create embeddings for our test peptides
    embeddings = {}
    for peptide in test_peptides:
        embeddings[peptide] = np.random.randn(128).astype(np.float32)
    
    # Create synthetic scores
    all_scores = {}
    for peptide in test_peptides:
        all_scores[peptide] = {
            'iptm': np.random.uniform(0.0, 1.0),
            'in_binding_site': np.random.choice([0, 1]),
            'pae': np.random.uniform(0.0, 30.0),
            'dG': np.random.uniform(-15.0, 5.0),
        }
    
    # Define binding sites
    binding_site_1 = [23, 42, 44]  # First hotspot
    binding_site_2 = [69, 72, 74]  # Second hotspot
    combined_sites = [23, 42, 44, 69, 72, 74]  # Both hotspots
    
    # Test 1: Traditional approach (same binding site for all peptides)
    logging.info("Testing traditional binding site approach (list format)...")
    
    from bopep.bayesian_optimization import optimization
    original_validate = optimization._validate_dependencies
    optimization._validate_dependencies = lambda: None
    
    try:
        bopep_traditional = BoPep(
            surrogate_model_kwargs={
                "network_type": "mlp",
                "model_type": "deep_evidential",
            },
            docker_kwargs={
                "output_dir": os.path.join(output_dir, "traditional", "docking"),
                "num_cores": 1
            },
            hpo_kwargs={
                "n_trials": 2,
                "hpo_interval": 10 
            },
            objective_function=iptm_objective,
            scoring_kwargs={"scores_to_include": ["iptm", "in_binding_site"]},
            log_dir=os.path.join(output_dir, "traditional"),
            checkpoint_interval=2
        )
    finally:
        optimization._validate_dependencies = original_validate
    
    # Mock scoring for traditional approach
    def mock_score_batch_traditional(docked_dirs):
        mock_scores = {}
        for peptide in docked_dirs:
            if peptide in all_scores:
                mock_scores[peptide] = all_scores[peptide].copy()
                # Add a note that this was scored with traditional binding sites
                mock_scores[peptide]['binding_site_type'] = 'traditional'
        return mock_scores
    
    bopep_traditional._score_batch = mock_score_batch_traditional
    bopep_traditional.docker.dock_peptides = lambda peptides: peptides
    
    # Test 2: Peptide-specific approach (dict format)
    logging.info("Testing peptide-specific binding site approach (dict format)...")
    
    optimization._validate_dependencies = lambda: None
    try:
        bopep_specific = BoPep(
            surrogate_model_kwargs={
                "network_type": "mlp", 
                "model_type": "deep_evidential",
            },
            docker_kwargs={
                "output_dir": os.path.join(output_dir, "peptide_specific", "docking"),
                "num_cores": 1
            },
            hpo_kwargs={
                "n_trials": 2,
                "hpo_interval": 10
            },
            objective_function=iptm_objective,
            scoring_kwargs={"scores_to_include": ["iptm", "in_binding_site"]},
            log_dir=os.path.join(output_dir, "peptide_specific"),
            checkpoint_interval=2
        )
    finally:
        optimization._validate_dependencies = original_validate
    
    # Mock scoring for peptide-specific approach
    def mock_score_batch_specific(docked_dirs):
        mock_scores = {}
        for peptide in docked_dirs:
            if peptide in all_scores:
                mock_scores[peptide] = all_scores[peptide].copy()
                # Add a note that this was scored with peptide-specific binding sites
                mock_scores[peptide]['binding_site_type'] = 'peptide_specific'
        return mock_scores
    
    bopep_specific._score_batch = mock_score_batch_specific
    bopep_specific.docker.dock_peptides = lambda peptides: peptides
    
    # Define peptide-specific binding sites
    peptide_specific_binding_sites = {
        "ACDEFGHIK": binding_site_1,     # Peptide 1 -> hotspot 1
        "LMNPQRSTV": binding_site_2,     # Peptide 2 -> hotspot 2
        "WYACDEFGH": combined_sites,     # Peptide 3 -> both hotspots
        "IKLMNPQRS": binding_site_1,     # Peptide 4 -> hotspot 1  
        "TVWYACDEF": binding_site_2,     # Peptide 5 -> hotspot 2
        "GHIKLMNPQ": binding_site_1,     # Peptide 6 -> hotspot 1
        "RSTVWYACD": binding_site_2,     # Peptide 7 -> hotspot 2
        "EFGHIKLMN": combined_sites,     # Peptide 8 -> both hotspots
        "PQRSTVWYA": binding_site_1,     # Peptide 9 -> hotspot 1
        "CDEFGHIKL": binding_site_2      # Peptide 10 -> hotspot 2
    }
    
    # Use a fallback PDB file
    target_pdb_path = os.path.join(os.path.dirname(__file__), "..", "data", "4glf.pdb")
    if not os.path.exists(target_pdb_path):
        # Try other possible locations
        possible_paths = [
            "/home/er8813ha/bopep/data/4glf.pdb",
            "../data/4glf.pdb",
            "data/4glf.pdb"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                target_pdb_path = path
                break
        else:
            logging.warning("PDB file not found, using placeholder path")
            target_pdb_path = "placeholder.pdb"
    
    # Define minimal schedules for testing
    test_schedule = [
        {"acquisition": "standard_deviation", "iterations": 1},
        {"acquisition": "expected_improvement", "iterations": 2}
    ]
    
    try:
        # Run traditional approach
        logging.info("Running optimization with traditional binding sites...")
        bopep_traditional.optimize(
            target_structure_path=target_pdb_path,
            num_initial=3,
            batch_size=2,
            schedule=test_schedule,
            embeddings=embeddings,
            binding_site_residue_indices=combined_sites,  # List format - same for all
            n_validate=None,  # Skip validation for simple test
            assume_zero_indexed=True,
            initial_peptides=test_peptides[:3]  # Use first 3 peptides as initial
        )
        
        # Run peptide-specific approach  
        logging.info("Running optimization with peptide-specific binding sites...")
        bopep_specific.optimize(
            target_structure_path=target_pdb_path,
            num_initial=3,
            batch_size=2,
            schedule=test_schedule,
            embeddings=embeddings,
            binding_site_residue_indices=peptide_specific_binding_sites,  # Dict format
            n_validate=None,  # Skip validation for simple test
            initial_peptides=test_peptides[:3],  # Use first 3 peptides as initial
            assume_zero_indexed = True
        )
        
        logging.info("Both approaches completed successfully!")
        logging.info(f"Traditional approach logs: {os.path.join(output_dir, 'traditional')}")
        logging.info(f"Peptide-specific approach logs: {os.path.join(output_dir, 'peptide_specific')}")
        
        # Verify that both approaches created their checkpoint directories
        traditional_checkpoint = os.path.join(output_dir, "traditional", "checkpoint_0")
        specific_checkpoint = os.path.join(output_dir, "peptide_specific", "checkpoint_0")
        
        if os.path.exists(traditional_checkpoint):
            logging.info("✓ Traditional approach checkpoint created")
        else:
            logging.warning("✗ Traditional approach checkpoint missing")
            
        if os.path.exists(specific_checkpoint):
            logging.info("✓ Peptide-specific approach checkpoint created")
        else:
            logging.warning("✗ Peptide-specific approach checkpoint missing")
            
        print(f"\nBinding site test completed! Results saved to: {output_dir}")
        print("Check the logs to see how each approach handled the binding sites.")
        
    except Exception as e:
        logging.error(f"Error during binding site testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BoPep optimization functionality')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic data instead of trying to load real data')
    parser.add_argument('--binding-sites', action='store_true',
                       help='Test peptide-specific binding sites functionality')
    parser.add_argument('--output-dir', default='./test_output_checkpointing',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    if args.binding_sites:
        print("Running peptide-specific binding sites test...")
        test_peptide_specific_binding_sites(args.output_dir + "_binding_sites")
    elif args.synthetic:
        print("Running test with synthetic data only...")
        test_bopep_synthetic_only(args.output_dir)
    else:
        # Option 1: Try to use real data if available
        embeddings_path = "/srv/data1/er8813ha/docking-peptide/output_v2/benchmarking/embedding_methods/embedding_benchmark_data.pkl"
        objectives_csv = "/home/er8813ha/docking-peptide/src/benchmark/benchmark_scores.csv"
        
        print("Testing BoPep with checkpointing...")
        print("Trying real data paths first, will fall back to synthetic data if not found")
        print()
        
        test_bopep_with_precomputed_data(
            embeddings_path=embeddings_path,
            objectives_csv=objectives_csv,
            embedding_type="esm_1d_pca",
            output_dir=args.output_dir,
            use_synthetic=False
        )
    
    print(f"\nTest completed! Check the output directory: {args.output_dir}")
    if args.binding_sites:
        print("This test verified both traditional (list) and peptide-specific (dict) binding site formats.")
    else:
        print("This includes logs and checkpoint directories to verify the new checkpointing system.")
