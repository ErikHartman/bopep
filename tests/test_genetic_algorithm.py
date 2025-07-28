import tempfile
from bopep.genetic_algorithm.generate import BoGA


class TestBoGABasics:
    """Test basic BoGA functionality"""

    def test_init_with_required_params(self):
        """Test BoGA initialization with required parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boga = BoGA(
                target_structure_path="./data/1ssc.pdb",
                sequence_length=8,
                docker_kwargs={"output_dir": temp_dir},
                surrogate_model_kwargs={"model_type": "nn_ensemble", "network_type": "mlp"}
            )
            
            assert boga.target_structure_path == "./data/1ssc.pdb"
            assert boga.sequence_length == 8
            assert boga.n_init == 100  # default value
            assert boga.m_select == 50  # default value

    def test_random_sequence_generation(self):
        """Test random sequence generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boga = BoGA(
                target_structure_path="./data/1ssc.pdb",
                sequence_length=10,
                docker_kwargs={"output_dir": temp_dir},
                surrogate_model_kwargs={"model_type": "nn_ensemble", "network_type": "mlp"}
            )
            
            seq = boga._random_sequence()
            assert len(seq) == 10
            assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq)

    def test_generate_initial_sequences(self):
        """Test initial sequence generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boga = BoGA(
                target_structure_path="./data/1ssc.pdb",
                sequence_length=8,
                n_init=5,
                docker_kwargs={"output_dir": temp_dir},
                surrogate_model_kwargs={"model_type": "nn_ensemble", "network_type": "mlp"}
            )
            
            seqs = boga._generate_initial_sequences()
            assert len(seqs) == 5
            assert all(len(seq) == 8 for seq in seqs)

    def test_select_top(self):
        """Test selection of top sequences"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boga = BoGA(
                target_structure_path="./data/1ssc.pdb",
                sequence_length=8,
                docker_kwargs={"output_dir": temp_dir},
                surrogate_model_kwargs={"model_type": "nn_ensemble", "network_type": "mlp"}
            )
            
            data = {"seq1": 0.9, "seq2": 0.5, "seq3": 0.8, "seq4": 0.2}
            top_2 = boga._select_top(data, 2)
            
            assert len(top_2) == 2
            assert "seq1" in top_2  # highest score
            assert "seq3" in top_2  # second highest score

    def test_mutate_sequence(self):
        """Test sequence mutation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            boga = BoGA(
                target_structure_path="./data/1ssc.pdb",
                sequence_length=8,
                mutation_rate=1.0,  # 100% mutation rate for testing
                docker_kwargs={"output_dir": temp_dir},
                surrogate_model_kwargs={"model_type": "nn_ensemble", "network_type": "mlp"}
            )
            
            original = "AAAAAAAA"
            mutated = boga._mutate_sequence(original)
            
            assert len(mutated) == len(original)
            # With 100% mutation rate, should be different
            assert mutated != original
