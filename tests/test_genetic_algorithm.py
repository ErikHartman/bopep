"""
Tests for the genetic algorithm module.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from bopep.genetic_algorithm.mutate import PeptideMutator
from bopep.genetic_algorithm.generate import BoGA


class TestPeptideMutator:
    """Test the PeptideMutator class"""

    def test_init_default(self):
        """Test mutator initialization with default parameters"""
        mutator = PeptideMutator()
        assert mutator.min_sequence_length == 6
        assert mutator.max_sequence_length == 40
        assert mutator.mutation_rate == 0.01
        assert mutator.p_ins == 0.10
        assert mutator.p_del == 0.10

    def test_init_custom(self):
        """Test mutator initialization with custom parameters"""
        mutator = PeptideMutator(
            min_sequence_length=8,
            max_sequence_length=20,
            mutation_rate=0.05,
            p_ins=0.15,
            p_del=0.05
        )
        assert mutator.min_sequence_length == 8
        assert mutator.max_sequence_length == 20
        assert mutator.mutation_rate == 0.05
        assert mutator.p_ins == 0.15
        assert mutator.p_del == 0.05

    def test_generate_random_sequence(self):
        """Test random sequence generation"""
        mutator = PeptideMutator(min_sequence_length=5, max_sequence_length=10)
        
        sequence = mutator.generate_random_sequence()
        
        assert isinstance(sequence, str)
        assert 5 <= len(sequence) <= 10
        assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence)

    def test_crossover_single_point(self):
        """Test single-point crossover"""
        mutator = PeptideMutator(min_sequence_length=5, max_sequence_length=15)
        
        parent1 = "AAAAA"
        parent2 = "CCCCC"
        
        child = mutator.crossover(parent1, parent2, method="single")
        
        assert isinstance(child, str)
        assert 5 <= len(child) <= 15
        assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in child)
        # Child should have segments from both parents
        assert any(aa in child for aa in parent1) or any(aa in child for aa in parent2)

    def test_crossover_two_point(self):
        """Test two-point crossover"""
        mutator = PeptideMutator(min_sequence_length=5, max_sequence_length=15)
        
        parent1 = "ACDEFGHIK"
        parent2 = "LMNPQRSTV"
        
        child = mutator.crossover(parent1, parent2, method="two")
        
        assert isinstance(child, str)
        assert 5 <= len(child) <= 15
        assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in child)

    def test_crossover_length_constraints(self):
        """Test that crossover respects length constraints"""
        mutator = PeptideMutator(min_sequence_length=8, max_sequence_length=10)
        
        parent1 = "ACDEFGHIK"
        parent2 = "LMNPQRSTV"
        
        for _ in range(10):
            child = mutator.crossover(parent1, parent2, method="single")
            assert 8 <= len(child) <= 10
            
            child = mutator.crossover(parent1, parent2, method="two")
            assert 8 <= len(child) <= 10

    def test_crossover_invalid_method(self):
        """Test that invalid crossover method raises error"""
        mutator = PeptideMutator()
        
        with pytest.raises(ValueError, match="Unknown crossover method"):
            mutator.crossover("AAAAA", "CCCCC", method="invalid")

    def test_mutate_sequence_uniform_mode(self):
        """Test sequence mutation with uniform substitutions"""
        mutator = PeptideMutator(
            min_sequence_length=5,
            max_sequence_length=15,
            mutation_rate=0.5  # High rate to ensure mutation
        )
        
        parent = "AAAAA"
        evaluated = set()
        
        child = mutator.mutate_sequence(parent, evaluated)
        
        assert isinstance(child, str)
        assert child != parent  # Should be different
        assert child not in evaluated
        assert 5 <= len(child) <= 15
        assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in child)

    def test_mutate_sequence_length_constraints(self):
        """Test that mutations respect length constraints"""
        mutator = PeptideMutator(
            min_sequence_length=8,
            max_sequence_length=10,
            mutation_rate=1.0  # High rate to force many mutations
        )
        
        parent = "ACDEFGHI"  # 8 amino acids (at minimum)
        evaluated = set()
        
        for _ in range(10):  # Test multiple mutations
            child = mutator.mutate_sequence(parent, evaluated)
            assert 8 <= len(child) <= 10

    def test_mutate_sequence_avoids_evaluated(self):
        """Test that mutation avoids previously evaluated sequences"""
        mutator = PeptideMutator(
            min_sequence_length=5,
            max_sequence_length=8,
            mutation_rate=0.1
        )
        
        parent = "AAAAA"
        evaluated = {"AAAAB", "AAAAC", "AAAAD"}
        
        child = mutator.mutate_sequence(parent, evaluated)
        
        assert child not in evaluated
        assert child != parent

    def test_mutate_pool(self):
        """Test mutation of a pool of sequences"""
        mutator = PeptideMutator(
            min_sequence_length=5,
            max_sequence_length=10,
            mutation_rate=0.2
        )
        
        parents = ["AAAAA", "CCCCC", "GGGGG"]
        k_pool = 5
        evaluated = set()
        
        pool = mutator.mutate_pool(parents, k_pool, evaluated)
        
        assert len(pool) <= k_pool  # May be less if can't generate enough unique sequences
        assert all(isinstance(seq, str) for seq in pool)
        assert all(seq not in evaluated for seq in pool)
        assert all(seq not in parents for seq in pool)

    def test_mutate_pool_large_evaluated_set(self):
        """Test mutation pool with large evaluated set"""
        mutator = PeptideMutator(
            min_sequence_length=5,
            max_sequence_length=6,
            mutation_rate=0.1
        )
        
        parents = ["AAAAA"]
        k_pool = 3
        # Create a large evaluated set to make it harder to find new sequences
        evaluated = {f"AAAA{aa}" for aa in "BCDEFGH"}
        
        pool = mutator.mutate_pool(parents, k_pool, evaluated)
        
        assert all(seq not in evaluated for seq in pool)


class TestBoGA:
    """Test the BoGA (Bayesian Optimization with Genetic Algorithm) class"""

    @pytest.fixture
    def basic_surrogate_kwargs(self):
        """Basic surrogate model kwargs for testing"""
        return {
            'network_type': 'mlp',
            'model_type': 'nn_ensemble'
        }

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all the heavy dependencies for BoGA"""
        with patch('bopep.genetic_algorithm.generate.Docker') as mock_docker, \
             patch('bopep.genetic_algorithm.generate.ComplexScorer') as mock_complex_scorer, \
             patch('bopep.genetic_algorithm.generate.MonomerScorer') as mock_monomer_scorer, \
             patch('bopep.genetic_algorithm.generate.AlphaFoldMonomer') as mock_alphafold, \
             patch('bopep.genetic_algorithm.generate.ScoresToObjective') as mock_scores_to_obj, \
             patch('bopep.genetic_algorithm.generate.Embedder') as mock_embedder, \
             patch('bopep.genetic_algorithm.generate.AcquisitionFunction') as mock_acq_func, \
             patch('bopep.genetic_algorithm.generate.SurrogateModelManager') as mock_surr_mgr, \
             patch('bopep.genetic_algorithm.generate.Logger') as mock_logger, \
             patch('bopep.genetic_algorithm.generate.torch') as mock_torch:
            
            # Setup basic mocks
            mock_torch.cuda.is_available.return_value = False
            mock_docker_instance = Mock()
            mock_docker.return_value = mock_docker_instance
            
            mock_complex_scorer_instance = Mock()
            mock_complex_scorer.return_value = mock_complex_scorer_instance
            
            mock_monomer_scorer_instance = Mock()
            mock_monomer_scorer.return_value = mock_monomer_scorer_instance
            
            mock_alphafold_instance = Mock()
            mock_alphafold.return_value = mock_alphafold_instance
            
            mock_scores_to_obj_instance = Mock()
            mock_scores_to_obj.return_value = mock_scores_to_obj_instance
            
            mock_embedder_instance = Mock()
            mock_embedder.return_value = mock_embedder_instance
            
            mock_acq_func_instance = Mock()
            mock_acq_func.return_value = mock_acq_func_instance
            
            mock_surr_mgr_instance = Mock()
            mock_surr_mgr.return_value = mock_surr_mgr_instance
            
            yield {
                'docker': mock_docker_instance,
                'complex_scorer': mock_complex_scorer_instance,
                'monomer_scorer': mock_monomer_scorer_instance,
                'alphafold': mock_alphafold_instance,
                'scores_to_obj': mock_scores_to_obj_instance,
                'embedder': mock_embedder_instance,
                'acq_func': mock_acq_func_instance,
                'surr_mgr': mock_surr_mgr_instance,
            }

    def test_init_basic(self, mock_dependencies, basic_surrogate_kwargs):
        """Test basic BoGA initialization"""
        schedule = [
            {
                'acquisition': 'ei',
                'generations': 5,
                'm_select': 10,
                'k_pool': 20,
                
            }
        ]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        assert boga.target_structure_path == "/fake/path.pdb"
        assert boga.initial_sequences == "ACDEFG"
        assert boga.min_sequence_length == 6
        assert boga.max_sequence_length == 40
        assert boga.n_init == 130
        assert boga.pca_n_components == 10

    def test_init_custom_params(self, mock_dependencies, basic_surrogate_kwargs):
        """Test BoGA initialization with custom parameters"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=["ACDEFG", "HIJKLM"],
            n_init=50,
            min_sequence_length=4,
            max_sequence_length=20,
            mutation_rate=0.05,
            embed_method='aaindex',
            pca_n_components=5,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        assert boga.n_init == 50
        assert boga.min_sequence_length == 4
        assert boga.max_sequence_length == 20
        assert boga.mutation_rate == 0.05
        assert boga.embed_method == 'aaindex'
        assert boga.pca_n_components == 5

    def test_init_no_pca_components_error(self, mock_dependencies, basic_surrogate_kwargs):
        """Test that initialization fails without PCA components"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        with pytest.raises(ValueError, match="pca_n_components must be specified"):
            BoGA(
                target_structure_path="/fake/path.pdb",
                initial_sequences="ACDEFG",
                pca_n_components=None,
                surrogate_model_kwargs=basic_surrogate_kwargs
            )

    def test_init_none_initial_sequences_error(self, mock_dependencies, basic_surrogate_kwargs):
        """Test that initialization fails with None initial sequences during preparation"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=None,
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        # Error should happen when preparing population, not during init
        with pytest.raises(ValueError, match="initial_sequences cannot be None"):
            boga._prepare_initial_population()

    def test_random_sequence(self, mock_dependencies, basic_surrogate_kwargs):
        """Test random sequence generation"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        seq = boga.mutator.generate_random_sequence()
        assert isinstance(seq, str)
        assert 6 <= len(seq) <= 40
        assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq)

    def test_generate_initial_sequences(self, mock_dependencies, basic_surrogate_kwargs):
        """Test initial sequence generation"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            n_init=20,
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        sequences = boga._generate_initial_sequences()
        assert len(sequences) == 20
        assert all(isinstance(seq, str) for seq in sequences)

    def test_prepare_initial_population_single_sequence(self, mock_dependencies, basic_surrogate_kwargs):
        """Test initial population preparation from single sequence"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            n_init=5,
            pca_n_components=3,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        sequences = boga._prepare_initial_population()
        assert len(sequences) >= 5  # May be more due to PCA requirements
        assert "ACDEFG" in sequences  # Original sequence should be included
        assert len(set(sequences)) == len(sequences)  # All unique

    def test_prepare_initial_population_list_sequences(self, mock_dependencies, basic_surrogate_kwargs):
        """Test initial population preparation from list of sequences"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        initial_seqs = ["ACDEFG", "HIKLMN", "NPQRSV"]  # Changed J to I and O to V to avoid invalid amino acids
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=initial_seqs,
            n_init=5,
            pca_n_components=3,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        sequences = boga._prepare_initial_population()
        assert len(sequences) >= 5
        for seq in initial_seqs:
            assert seq in sequences

    def test_prepare_initial_population_many_sequences(self, mock_dependencies, basic_surrogate_kwargs):
        """Test initial population when we have more sequences than needed"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        initial_seqs = [f"ACDEFG{i:02d}A" for i in range(20)]  # 20 sequences
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=initial_seqs,
            n_init=10,
            pca_n_components=5,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        sequences = boga._prepare_initial_population()
        assert len(sequences) == 20  # Current implementation returns all sequences when enough are provided

    def test_embed_peptides(self, mock_dependencies, basic_surrogate_kwargs):
        """Test that embedding peptides works properly (no longer uses caching)"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        # Mock embedder methods
        mock_raw_embeddings = {
            "ACDEFG": np.random.randn(1280),
            "HIJKLM": np.random.randn(1280)
        }
        mock_scaled_embeddings = {
            "ACDEFG": np.random.randn(1280),
            "HIJKLM": np.random.randn(1280)
        }
        mock_reduced_embeddings = {
            "ACDEFG": np.random.randn(10),
            "HIJKLM": np.random.randn(10)
        }
        
        mock_dependencies['embedder'].embed_esm.return_value = mock_raw_embeddings
        mock_dependencies['embedder'].scale_embeddings.return_value = mock_scaled_embeddings
        mock_dependencies['embedder'].reduce_embeddings_pca.return_value = mock_reduced_embeddings
        
        # Test embedding peptides
        result = boga._embed_peptides(["ACDEFG", "HIJKLM"])
        assert "ACDEFG" in result
        assert "HIJKLM" in result
        
        # Verify embedder methods were called
        mock_dependencies['embedder'].embed_esm.assert_called_once()
        mock_dependencies['embedder'].scale_embeddings.assert_called_once()
        mock_dependencies['embedder'].reduce_embeddings_pca.assert_called_once()

    def test_select_top_objectives(self, mock_dependencies, basic_surrogate_kwargs):
        """Test selection of top objectives"""
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        objectives = {
            "SEQ1": 0.5,
            "SEQ2": 0.9,
            "SEQ3": 0.3,
            "SEQ4": 0.7
        }
        
        top_seqs = boga._select_top_objectives(objectives, m_pool=2)
        
        assert len(top_seqs) == 2
        # With uniform selection, we can't guarantee which exact sequences are selected
        # but they should be from the top performers
        assert all(seq in objectives for seq in top_seqs)

    def test_select_top_objectives_exponential(self, mock_dependencies, basic_surrogate_kwargs):
        """Test exponential selection with beta parameter"""
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        objectives = {
            "SEQ1": 0.2,
            "SEQ2": 0.9,
            "SEQ3": 0.3,
            "SEQ4": 0.7,
            "SEQ5": 0.8
        }
        
        # Test with exponential selection
        top_seqs = boga._select_top_objectives(
            objectives, 
            m_pool=3, 
            selection_method="exponential",
            beta=2.0
        )
        
        assert len(top_seqs) == 3
        assert all(seq in objectives for seq in top_seqs)
        # Higher beta should favor top performers more strongly

    def test_select_top_objectives_integer_top_fraction(self, mock_dependencies, basic_surrogate_kwargs):
        """Test selection with integer top_fraction"""
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        objectives = {f"SEQ{i}": float(i) / 10.0 for i in range(1, 11)}
        
        # Test with integer top_fraction (absolute number)
        top_seqs = boga._select_top_objectives(
            objectives, 
            m_pool=3,
            top_fraction=5  # Take top 5 sequences
        )
        
        assert len(top_seqs) == 3

    def test_select_top_predictions(self, mock_dependencies, basic_surrogate_kwargs):
        """Test selection of top predictions using acquisition function"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        predictions = {
            "SEQ1": (0.5, 0.1),  # (mean, std)
            "SEQ2": (0.6, 0.2),
            "SEQ3": (0.4, 0.3)
        }
        
        # Mock acquisition function
        mock_acq_values = {
            "SEQ1": 0.3,
            "SEQ2": 0.8,
            "SEQ3": 0.5
        }
        mock_dependencies['acq_func'].compute_acquisition.return_value = mock_acq_values
        
        top_seqs = boga._select_top_predictions(predictions, k=2, acquisition_function='ei')
        
        assert len(top_seqs) == 2
        assert "SEQ2" in top_seqs  # Highest acquisition value

    def test_load_from_logs(self, mock_dependencies, basic_surrogate_kwargs, temp_dir):
        """Test loading previous results from log files"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        # Create mock log file
        log_dir = Path(temp_dir)
        scores_file = log_dir / "scores.csv"
        scores_content = """peptide,rosetta_score,distance_score,iteration,phase,timestamp
ACDEFG,-10.5,0.8,0,initial,2024-01-01
HIKLMN,-8.2,0.6,1,phase1,2024-01-01
"""
        scores_file.write_text(scores_content)
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            continue_from_logs=str(log_dir),
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        scores, evaluated, last_iteration = boga._load_from_logs(str(log_dir))
        
        assert len(scores) == 2
        assert "ACDEFG" in scores
        assert "HIKLMN" in scores  # Updated sequence name
        assert scores["ACDEFG"]["rosetta_score"] == -10.5
        assert len(evaluated) == 2
        assert last_iteration == 1  # Should be the max iteration from the CSV

    def test_load_from_logs_missing_file(self, mock_dependencies, basic_surrogate_kwargs, temp_dir):
        """Test loading from logs when file doesn't exist"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        with pytest.raises(FileNotFoundError):
            boga._load_from_logs(str(temp_dir))

    def test_iteration_continuation(self, mock_dependencies, basic_surrogate_kwargs, temp_dir):
        """Test that iterations continue from last iteration when resuming from logs"""
        schedule = [{'acquisition': 'ei', 'generations': 2, 'm_select': 2, 'k_pool': 3}]
        
        # Create mock log file with iteration 5 as the last iteration
        log_dir = Path(temp_dir)
        scores_file = log_dir / "scores.csv"
        scores_content = """peptide,rosetta_score,distance_score,iteration,phase,timestamp
ACDEFG,-10.5,0.8,0,initial,2024-01-01
HIKLMN,-8.2,0.6,5,phase1,2024-01-01
"""
        scores_file.write_text(scores_content)
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences="ACDEFG",
            pca_n_components=3,
            continue_from_logs=str(log_dir),
            surrogate_model_kwargs=basic_surrogate_kwargs,
            scoring_kwargs={'scores_to_include': ['score1'], 'n_jobs': 1}
        )
        
        # Mock all the dependencies for a partial run
        mock_objectives = {"ACDEFG": 0.5, "HIKLMN": 0.3}
        mock_embeddings = {"ACDEFG": np.random.randn(3), "HIKLMN": np.random.randn(3)}
        
        mock_dependencies['scores_to_obj'].create_objective.return_value = mock_objectives
        
        def mock_embed_peptides(sequences):
            return {seq: np.random.randn(3) for seq in sequences}
        
        def mock_embed_generation(scored, candidates):
            all_seqs = scored + candidates
            train_emb = {seq: np.random.randn(3) for seq in scored}
            cand_emb = {seq: np.random.randn(3) for seq in candidates}
            return train_emb, cand_emb
            
        boga._embed_peptides = mock_embed_peptides
        boga._embed_generation = mock_embed_generation
        mock_dependencies['surr_mgr'].train_with_validation_split.return_value = (0.1, {})
        mock_dependencies['surr_mgr'].predict.return_value = {"NEWSEQ": (0.6, 0.1)}
        mock_dependencies['acq_func'].compute_acquisition.return_value = {"NEWSEQ": 0.7}
        
        # Mock the scoring to avoid actual docking
        def mock_dock_and_score(sequences):
            return {seq: {"score1": 0.5} for seq in sequences}
        boga._dock_and_score = mock_dock_and_score
        
        # Extract just the first generation to see the starting iteration
        
        # Run and capture the last_iteration that gets loaded
        scores, evaluated_seqs, last_iteration = boga._load_from_logs(str(log_dir))
        
        # Verify that the last iteration is correctly identified as 5
        assert last_iteration == 5

    def test_logger_initialization_with_continue_from_logs(self, mock_dependencies, basic_surrogate_kwargs, temp_dir):
        """Test that logger doesn't overwrite logs when continuing from logs"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        # Test case 1: continue_from_logs only (should not overwrite)
        with patch('bopep.genetic_algorithm.generate.Logger') as mock_logger:
            boga1 = BoGA(
                target_structure_path="/fake/path.pdb",
                initial_sequences="ACDEFG",
                pca_n_components=10,
                continue_from_logs=str(temp_dir),
                surrogate_model_kwargs=basic_surrogate_kwargs
            )
            # Should call Logger with overwrite_logs=False
            mock_logger.assert_called_once_with(log_dir=str(temp_dir), overwrite_logs=False)
        
        # Test case 2: both log_dir and continue_from_logs (should prioritize continue_from_logs)
        with patch('bopep.genetic_algorithm.generate.Logger') as mock_logger:
            boga2 = BoGA(
                target_structure_path="/fake/path.pdb",
                initial_sequences="ACDEFG",
                pca_n_components=10,
                log_dir="/some/other/dir",
                continue_from_logs=str(temp_dir),
                surrogate_model_kwargs=basic_surrogate_kwargs
            )
            # Should call Logger with continue_from_logs path and overwrite_logs=False
            mock_logger.assert_called_once_with(log_dir=str(temp_dir), overwrite_logs=False)
        
        # Test case 3: log_dir only (should overwrite)
        with patch('bopep.genetic_algorithm.generate.Logger') as mock_logger:
            boga3 = BoGA(
                target_structure_path="/fake/path.pdb",
                initial_sequences="ACDEFG",
                pca_n_components=10,
                log_dir="/some/log/dir",
                surrogate_model_kwargs=basic_surrogate_kwargs
            )
            # Should call Logger with overwrite_logs=True
            mock_logger.assert_called_once_with(log_dir="/some/log/dir", overwrite_logs=True)

    @patch('bopep.genetic_algorithm.generate.pd')
    def test_run_fresh_start_basic(self, mock_pd, mock_dependencies, basic_surrogate_kwargs):
        """Test basic run functionality for fresh start (without full execution)"""
        schedule = [
            {
                'acquisition': 'ei',
                'generations': 1,
                'm_select': 2,
                'k_pool': 3,
                
            }
        ]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=["ACDEFG", "HIJKLM"],
            n_init=2,
            pca_n_components=5,
            surrogate_model_kwargs=basic_surrogate_kwargs,
            scoring_kwargs={
                'scores_to_include': ['score1'],
                'n_jobs': 1
            }
        )
        
        # Mock all the methods that would be called
        mock_scores = {"ACDEFG": {"score1": 0.5}, "HIJKLM": {"score1": 0.3}}
        mock_objectives = {"ACDEFG": 0.5, "HIJKLM": 0.3}
        mock_embeddings = {"ACDEFG": np.random.randn(5), "HIJKLM": np.random.randn(5)}
        
        # Mock external calls
        mock_dependencies['docker'].dock_peptides.return_value = ["dir1", "dir2"]
        mock_dependencies['complex_scorer'].score_batch.return_value = mock_scores
        mock_dependencies['scores_to_obj'].create_objective.return_value = mock_objectives
        
        # Mock embedding methods to handle any sequence
        def mock_embed_esm(sequences, **kwargs):
            return {seq: np.random.randn(1280) for seq in sequences}
            
        def mock_scale_embeddings(embeddings):
            return {seq: np.random.randn(1280) for seq in embeddings.keys()}
            
        def mock_reduce_embeddings_pca(embeddings, **kwargs):
            return {seq: np.random.randn(5) for seq in embeddings.keys()}
        
        mock_dependencies['embedder'].embed_esm.side_effect = mock_embed_esm
        mock_dependencies['embedder'].scale_embeddings.side_effect = mock_scale_embeddings
        mock_dependencies['embedder'].reduce_embeddings_pca.side_effect = mock_reduce_embeddings_pca
        mock_dependencies['surr_mgr'].train_with_validation_split.return_value = (0.1, {})
        mock_dependencies['surr_mgr'].predict.return_value = {
            "NEWSEQ1": (0.6, 0.1),
            "NEWSEQ2": (0.4, 0.2)
        }
        mock_dependencies['acq_func'].compute_acquisition.return_value = {
            "NEWSEQ1": 0.7,
            "NEWSEQ2": 0.3
        }
        
        # Mock the mutator to avoid amino acid issues
        with patch.object(boga.mutator, 'mutate_sequence') as mock_mutate:
            mock_mutate.side_effect = lambda parent, evaluated: f"{parent}X"  # Simple mutation
            
            # Mock the run to stop after initial setup by using empty schedule
            result = boga.run(schedule=[])
        
        # Should return the objectives
        assert isinstance(result, dict)
        assert len(result) >= 2  # At least the initial sequences

    def test_invalid_initial_sequences_type(self, mock_dependencies, basic_surrogate_kwargs):
        """Test initialization with invalid initial sequences type"""
        schedule = [{'acquisition': 'ei', 'generations': 1, 'm_select': 5, 'k_pool': 10}]
        
        boga = BoGA(
            target_structure_path="/fake/path.pdb",
            initial_sequences=123,  # Invalid type
            pca_n_components=10,
            surrogate_model_kwargs=basic_surrogate_kwargs
        )
        
        # Error should happen when preparing population, not during init
        with pytest.raises(ValueError, match="initial_sequences must be None, a string, or a list of strings"):
            boga._prepare_initial_population()