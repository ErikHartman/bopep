import random
from typing import List, Dict, Any, Optional, Callable, Union
from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import (
    tune_hyperparams,
    NeuralNetworkEnsemble,
    MonteCarloDropout,
    DeepEvidentialRegression,
    MVE
)
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.logging.logger import Logger
import torch
import numpy as np
_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

class BoGA:
    """
    Genetic algorithm for peptide binder discovery using surrogate modeling.

    In BoGA, we specify a target and binding site, then:

    1. Generate/prepare initial population of N sequences:
       - If initial_sequences is None: generate N random sequences
       - If initial_sequences is a string: treat as single sequence and generate N mutations
       - If initial_sequences is a list with ≥N sequences: use first N sequences
       - If initial_sequences is a list with <N sequences: use all + fill remainder with mutations/random

    2. Dock and score initial population
    3. Train surrogate model on scores
    4. For each generation:
       - Select M best sequences based on predicted affinity
       - Mutate selected sequences to generate new K candidate sequences
       - Evaluate candidates and update population
    5. Repeat until convergence or max generations

    TODO: 
    - Add validation of binding site indices
    - Add more detailed logging
    - Abstract out things from BoPep and use here (eg binding site validation, hparam opt)
    """
    def __init__(
        self,
        target_structure_path: str,
        max_sequence_length: int,
        initial_sequences: Optional[Union[str, List[str]]] = None,
        min_sequence_length: int = 6,
        n_init: int = 100,
        m_select: int = 50,
        k_pool: int = 5_000,
        generations: int = 100,
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        mutation_rate: float = 0.01,
        random_seed: Optional[int] = None,
        # Embedding options
        embed_method: str = 'esm',               # 'esm' or 'aaindex'
        embed_average: bool = True,
        embed_model_path: Optional[str] = None,
        embed_batch_size: int = 128,
        embed_device: Optional[str] = None,
        # PCA reduction
        pca_explained_variance_ratio: float = 0.95,
        pca_n_components: Optional[int] = None,
        # Hyperparameter tuning interval
        hpo_interval: int = 10,
        # Logging options
        log_dir: Optional[str] = None,
        enable_logging: bool = True,
        # Testing options
        use_dummy_scoring: bool = False,
    ):
        """
        Parameters
        ----------
        target_structure_path : str
            Path to target protein structure for docking
        max_sequence_length : int
            Maximum length of generated peptide sequences
        initial_sequences : Optional[Union[str, List[str]]], default=None
            Initial population specification:
            - None: generate n_init random sequences
            - str: single sequence to mutate n_init times  
            - List[str]: list of sequences (truncated to n_init if too many, 
              extended with mutations/random if too few)
        min_sequence_length : int, default=6
            Minimum length of generated peptide sequences
        n_init : int, default=100
            Size of initial population
        hpo_interval : int, default=10
            Hyperparameter optimization interval (every N generations)
        log_dir : Optional[str], default=None
            Directory for logging files. If None, logging is disabled.
        enable_logging : bool, default=True
            Whether to enable logging (requires log_dir to be set)
        use_dummy_scoring : bool, default=False
            Whether to use dummy scoring instead of real docking for testing
        ... (other parameters)
        """
        
        self.initial_sequences = initial_sequences

        # Validate surrogate model config
        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
        _validate_surrogate_model_kwargs(self.surrogate_model_kwargs)

        # Store GA parameters
        self.target_structure_path = target_structure_path
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.n_init = n_init
        self.m_select = m_select
        self.k_pool = k_pool
        self.generations = generations
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.scoring_kwargs = scoring_kwargs or {}
        self.mutation_rate = mutation_rate
        self.hpo_interval = hpo_interval

        # Embedding configuration
        self.embed_method = embed_method.lower()
        self.embed_average = embed_average
        self.embed_model_path = embed_model_path
        self.embed_batch_size = embed_batch_size
        self.embed_device = embed_device
        self.pca_explained_variance_ratio = pca_explained_variance_ratio
        self.pca_n_components = pca_n_components

        # Initialize components
        self.docker = Docker(docker_kwargs or {})
        self.scorer = Scorer()
        self.scores_to_objective = ScoresToObjective()
        self.embedder = Embedder()

        # Initialize logging
        self.enable_logging = enable_logging
        if self.enable_logging and log_dir is not None:
            self.logger = Logger(log_dir=log_dir, overwrite_logs=True)
        else:
            self.logger = None

        # Store testing options
        self.use_dummy_scoring = use_dummy_scoring

        # Placeholders for model and hyperparameters
        self.best_hyperparams = None
        self.previous_study = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if random_seed is not None:
            random.seed(random_seed)

    def _random_sequence(self) -> str:
        return ''.join(random.choice(_AMINO_ACIDS) for _ in range(random.randint(self.min_sequence_length, self.max_sequence_length)))

    def _generate_initial_sequences(self) -> List[str]:
        return [self._random_sequence() for _ in range(self.n_init)]

    def _prepare_initial_population(self) -> List[str]:
        """
        Prepare the initial population based on the initial_sequences parameter:
        - If None: generate n_init random sequences
        - If single string: treat as single sequence and mutate n_init times
        - If list with enough sequences (>= n_init): use first n_init sequences
        - If list with few sequences (< n_init): use all + fill remainder with mutations/random
        """
        if self.initial_sequences is None:
            # No initial sequences provided - generate random
            return self._generate_initial_sequences()
        
        elif isinstance(self.initial_sequences, str):
            # Single sequence provided - mutate it n_init times
            base_sequence = self.initial_sequences
            sequences = [base_sequence]  # Include original
            # Generate n_init-1 mutations of the base sequence
            for _ in range(self.n_init - 1):
                sequences.append(self._mutate_sequence(base_sequence))
            return sequences
        
        elif isinstance(self.initial_sequences, list):
            if len(self.initial_sequences) >= self.n_init:
                # Enough sequences provided - use first n_init
                return self.initial_sequences[:self.n_init]
            else:
                # Not enough sequences - use all and fill remainder
                sequences = list(self.initial_sequences)
                remaining = self.n_init - len(sequences)
                
                # Fill remainder with mutations of existing sequences and random sequences
                for _ in range(remaining):
                    if random.random() < 0.7:  # 70% chance to mutate existing sequence
                        parent = random.choice(self.initial_sequences)
                        sequences.append(self._mutate_sequence(parent))
                    else:  # 30% chance to generate completely random sequence
                        sequences.append(self._random_sequence())
                
                return sequences
        
        else:
            raise ValueError("initial_sequences must be None, a string, or a list of strings")

    def _embed(self, peptides: List[str]) -> Dict[str, Any]:
        """
        Embed peptides (ESM or AAIndex), scale, and apply PCA reduction. Returns
        reduced embeddings directly.
        """
        # Embed
        if self.embed_method == 'esm':
            raw = self.embedder.embed_esm(
                peptides,
                average=self.embed_average,
                model_path=self.embed_model_path,
                batch_size=self.embed_batch_size,
                filter=True,
                device=self.embed_device
            )
        elif self.embed_method == 'aaindex':
            raw = self.embedder.embed_aaindex(
                peptides,
                average=self.embed_average,
                filter=True
            )
        else:
            raise ValueError("embed_method must be 'esm' or 'aaindex'")
        # Scale + PCA
        scaled = self.embedder.scale_embeddings(raw)
        reduced = self.embedder.reduce_embeddings_pca(
            scaled,
            explained_variance_ratio=self.pca_explained_variance_ratio,
            n_components=self.pca_n_components
        )
        return reduced

    def _dock_and_score(self, sequences: List[str]) -> Dict[str, float]:
        dock_dirs = self.docker.dock_peptides(sequences)
        return self.scorer.score_batch(
            scores_to_include=self.scoring_kwargs.get('scores_to_include', []),
            inputs=dock_dirs,
            input_type='colab_dir',
            binding_site_residue_indices=self.scoring_kwargs.get('binding_site_residue_indices'),
            n_jobs=self.scoring_kwargs.get('n_jobs', 12),
            binding_site_distance_threshold=self.scoring_kwargs.get('binding_site_distance_threshold', 5),
            required_n_contact_residues=self.scoring_kwargs.get('required_n_contact_residues', 5),
        )

    def _dock_and_score_dummy(self, sequences: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Dummy docking and scoring for testing purposes.
        Generates realistic-looking fake scores based on sequence properties.
        """
        import hashlib
        import time
        
        print(f"  [DUMMY] Docking and scoring {len(sequences)} sequences...")
        time.sleep(0.5)  # Simulate some processing time
        
        results = {}
        for seq in sequences:
            # Use sequence hash for reproducible "random" scores
            hash_val = int(hashlib.md5(seq.encode()).hexdigest()[:8], 16)
            np.random.seed(hash_val % 2**31)  # Ensure reproducible results
            
            # Generate realistic score ranges based on typical values
            seq_len = len(seq)
            
            # Rosetta score: typically -1000 to 400
            rosetta_score = np.random.uniform(-800, 200) - seq_len * 10
            
            # Interface dG: typically -100 to 20 
            interface_dG = np.random.uniform(-80, 10) + np.random.normal(0, 15)
            
            # Distance score: typically 5 to 8
            distance_score = np.random.uniform(5.5, 7.5)
            
            # IPTM: typically 0.1 to 0.95
            iptm = np.random.beta(2, 3) * 0.85 + 0.1
            
            # Peptide PAE: typically 3 to 30
            peptide_pae = np.random.gamma(2, 3) + 3
            if peptide_pae > 30:
                peptide_pae = 30
                
            # Interface SASA: typically 0 to 2000
            interface_sasa = np.random.exponential(400)
            if interface_sasa > 2000:
                interface_sasa = 2000
                
            # Number of contacts: typically 0 to 20
            n_contacts = np.random.poisson(5)
            if n_contacts > 20:
                n_contacts = 20
                
            # Peptide plDDT: typically 40 to 90
            peptide_plddt = np.random.normal(70, 15)
            peptide_plddt = max(40, min(90, peptide_plddt))
            
            # In binding site: make it sequence-dependent but mostly true
            in_binding_site = (hash_val % 10) < 8  # 80% chance of being in binding site
            
            results[seq] = {
                'rosetta_score': rosetta_score,
                'interface_dG': interface_dG,
                'distance_score': distance_score,
                'iptm': iptm,
                'peptide_pae': peptide_pae,
                'interface_sasa': interface_sasa,
                'n_contacts': n_contacts,
                'peptide_plddt': peptide_plddt,
                'in_binding_site': in_binding_site
            }
            
        return results

    def _optimize_hyperparameters(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> None:
        """
        Hyperparameter tuning and model training.
        Called only when generation % hpo_interval == 0.
        """
        self.best_hyperparams, self.previous_study = tune_hyperparams(
            model_type=self.surrogate_model_kwargs['model_type'],
            embedding_dict=embeddings,
            objective_dict=objectives,
            network_type=self.surrogate_model_kwargs['network_type'],
            n_trials=self.surrogate_model_kwargs.get('n_trials', 20),
            n_splits=self.surrogate_model_kwargs.get('n_splits', 3),
            random_state=self.surrogate_model_kwargs.get('random_state', 42),
            previous_study=self.previous_study,
            device=self.device
        )

    def _train_model(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> None:
        """
        Train with existing hyperparameters without tuning.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized before training")
        self.model.fit_dict(embeddings, objectives, device=self.device)

    def _initialize_model(self, hyperparams: Dict[str, Any]) -> None:
        model_type = self.surrogate_model_kwargs['model_type']
        network_type = self.surrogate_model_kwargs['network_type']
        input_dim = hyperparams.get('input_dim')
        hidden_dims = hyperparams.get('hidden_dims')
        hidden_dim = hyperparams.get('hidden_dim')
        num_layers = hyperparams.get('num_layers', 2)
        uncertainty_param = hyperparams.get('uncertainty_param')

        if model_type == 'mve':
            self.model = MVE(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                mve_regularization=uncertainty_param
            )
        elif model_type == 'deep_evidential':
            self.model = DeepEvidentialRegression(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                evidential_regularization=uncertainty_param
            )
        elif model_type == 'mc_dropout':
            self.model = MonteCarloDropout(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                dropout_rate=uncertainty_param
            )
        elif model_type == 'nn_ensemble':
            self.model = NeuralNetworkEnsemble(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                network_type=network_type,
                n_networks=int(uncertainty_param)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model.to(self.device)

    def _select_top(self, data: Dict[str, float], k: int) -> List[str]:
        return [seq for seq, _ in sorted(data.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _mutate_sequence(self, seq: str) -> str:
        """
        Apply substitution, insertion, or deletion to the sequence based on mutation_rate.
        """
        seq_list = list(seq)
        new_seq = []
        for aa in seq_list:
            r = random.random()
            if r < self.mutation_rate:
                # Choose mutation type
                op = random.choice(['sub', 'del', 'ins'])
                if op == 'sub':
                    # substitution
                    new_seq.append(random.choice(_AMINO_ACIDS))
                elif op == 'del':
                    # deletion: skip this residue
                    continue
                else:
                    # insertion: insert a random AA before current
                    new_seq.append(random.choice(_AMINO_ACIDS))
                    new_seq.append(aa)
            else:
                new_seq.append(aa)
        # Additionally, random insertion at end
        if random.random() < self.mutation_rate:
            new_seq.append(random.choice(_AMINO_ACIDS))
        # Truncate or pad to desired max_sequence_length
        if len(new_seq) > self.max_sequence_length:
            return ''.join(new_seq[:self.max_sequence_length])
        else:
            # pad by random AAs if too short
            while len(new_seq) < self.min_sequence_length:
                new_seq.append(random.choice(_AMINO_ACIDS))
            return ''.join(new_seq)

    def _mutate_pool(self, parents: List[str]) -> List[str]:
        """Generate new pool by mutating selected parent sequences."""
        pool = []
        for _ in range(self.k_pool):
            parent = random.choice(parents)
            pool.append(self._mutate_sequence(parent))
        return pool

    def _calculate_model_accuracy(self, predictions: Dict[str, float], actual: Dict[str, float]) -> Dict[str, float]:
        """Calculate model accuracy metrics by comparing predictions to actual values."""
        if not predictions or not actual:
            return {}
        
        # Find common sequences
        common_seqs = set(predictions.keys()) & set(actual.keys())
        if not common_seqs:
            return {}
        
        pred_values = np.array([predictions[seq] for seq in common_seqs])
        actual_values = np.array([actual[seq] for seq in common_seqs])
        
        # Calculate metrics
        mse = np.mean((pred_values - actual_values) ** 2)
        mae = np.mean(np.abs(pred_values - actual_values))
        
        # Correlation coefficient (if variance exists)
        if np.var(pred_values) > 0 and np.var(actual_values) > 0:
            correlation = np.corrcoef(pred_values, actual_values)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'n_samples': len(common_seqs)
        }

    def run(self) -> Dict[str, float]:
        # Initial population and embedding/reduction
        init_seqs = self._prepare_initial_population()
        print(f"Generated initial population of {len(init_seqs)} sequences")

        init_reduced = self._embed(init_seqs)

        # Dock and score initial
        print("Docking and scoring initial population...")
        if self.use_dummy_scoring:
            scores = self._dock_and_score_dummy(init_seqs)
        else:
            scores = self._dock_and_score(init_seqs)

        # Convert initial scores to objectives
        objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

        # Log initial population
        if self.logger:
            self.logger.log_scores(scores, iteration=0, acquisition_name="initial")
            self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")

        print(f"Initial population - Best objective: {max(objectives.values()):.4f}")

        # Initial hyperparameter tuning
        print("Optimizing initial hyperparameters...")
        self._optimize_hyperparameters(init_reduced, objectives)

        for gen in range(1, self.generations + 1):
            print(f"\n=== Generation {gen}/{self.generations} ===")
            
            # Init fresh model
            self._initialize_model(self.best_hyperparams) 
            # Embed and reduce current peptides
            seqs = list(scores.keys())
            reduced_embs = self._embed(seqs)

            # Convert scores to objectives
            objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

            # Train surrogate or model only based on interval
            if gen % self.hpo_interval == 0:
                print(f"Re-optimizing hyperparameters (generation {gen})")
                self._optimize_hyperparameters(reduced_embs, objectives)

            print("Training surrogate model...")
            self.model.fit_dict(reduced_embs, objectives, device=self.device)

            # Generate new pool via mutation of top M (use objectives for selection)
            parents = self._select_top(objectives, self.m_select)
            print(f"Selected top {len(parents)} parents for mutation")
            
            pool = self._mutate_pool(parents)
            print(f"Generated candidate pool of {len(pool)} sequences")

            # Embed and reduce pool
            pool_embs = self._embed(pool)

            # Predict and select top
            preds = self.model.predict_dict(pool_embs, device=self.device)
            candidates = self._select_top(preds, self.m_select)

            print(f"Selected {len(candidates)} candidates for evaluation")

            # Dock, score, and update
            if self.use_dummy_scoring:
                new_scores = self._dock_and_score_dummy(candidates)
            else:
                new_scores = self._dock_and_score(candidates)
            scores.update(new_scores)
            
            # Calculate new objectives for logging
            new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
            
            # Log generation data
            if self.logger:
                # Log new scores and objectives
                self.logger.log_scores(new_scores, iteration=gen, acquisition_name=f"generation_{gen}")
                self.logger.log_objectives(new_objectives, iteration=gen, acquisition_name=f"generation_{gen}")
                
                # Calculate and log model accuracy
                accuracy_metrics = self._calculate_model_accuracy(
                    {seq: preds[seq] for seq in candidates if seq in preds}, 
                    new_objectives
                )
                if accuracy_metrics:
                    self.logger.log_model_metrics(
                        loss=accuracy_metrics.get('mse', 0), 
                        iteration=gen, 
                        metrics={
                            'mae': accuracy_metrics.get('mae', 0),
                            'correlation': accuracy_metrics.get('correlation', 0),
                            'n_samples': accuracy_metrics.get('n_samples', 0)
                        }
                    )
                
                # Log hyperparameters if they were updated
                if gen % self.hpo_interval == 0 and self.best_hyperparams:
                    self.logger.log_hyperparameters(
                        iteration=gen,
                        hyperparams=self.best_hyperparams,
                        model_type=self.surrogate_model_kwargs['model_type'],
                        network_type=self.surrogate_model_kwargs['network_type']
                    )

            # Report generation progress
            current_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
            best_objective = max(current_objectives.values())
            best_sequence = max(current_objectives.items(), key=lambda x: x[1])[0]
            
            print(f"Generation {gen} - Best objective: {best_objective:.4f}")
            print(f"Best sequence so far: {best_sequence}")
            
            if accuracy_metrics:
                print(f"Model accuracy - MAE: {accuracy_metrics.get('mae', 0):.4f}, "
                      f"Correlation: {accuracy_metrics.get('correlation', 0):.4f}")

        # Return final objectives instead of raw scores
        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(final_objectives)}")
        print(f"Best final objective: {max(final_objectives.values()):.4f}")
        
        return final_objectives
