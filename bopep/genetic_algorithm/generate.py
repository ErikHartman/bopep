import random
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.search.acquisition_functions import AcquisitionFunction
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
    """
    def __init__(
        self,
        target_structure_path: str,
        max_sequence_length: int,
        schedule: List[Dict[str, Any]],
        initial_sequences: Optional[Union[str, List[str]]] = None,
        min_sequence_length: int = 6,
        n_init: int = 100,
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        mutation_rate: float = 0.01,
        random_seed: Optional[int] = None,
        # Embedding options
        embed_method: str = 'esm',               # 'esm' or 'aaindex'
        embed_model_path: Optional[str] = None,
        embed_batch_size: int = 128,
        embed_device: Optional[str] = None,
        # PCA reduction
        pca_n_components: Optional[int] = None,
        # Hyperparameter tuning interval
        hpo_interval: int = 10,
        # Validation options
        n_validate: Optional[int] = None,
        validation_split: float = 0.2,
        # Logging options
        log_dir: Optional[str] = None,
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
        schedule : List[Dict[str, Any]]
            List of dictionaries defining the optimization phases.
            Each dict should have 'acquisition', 'generations', 'm_select', and 'k_pool' keys.
            Example: [{'acquisition': 'expected_improvement', 'generations': 50, 'm_select': 50, 'k_pool': 5000}]
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
        self.schedule = schedule
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.scoring_kwargs = scoring_kwargs or {}
        self.mutation_rate = mutation_rate
        self.hpo_interval = hpo_interval
        self.n_validate = n_validate
        self.validation_split = validation_split

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Embedding configuration
        self.embed_method = embed_method.lower()
        # Auto-detect embedding averaging based on network type
        network_type = self.surrogate_model_kwargs.get('network_type', 'mlp').lower()
        if network_type == "mlp":
            self.embed_average = True
        else:
            self.embed_average = False
        self.embed_model_path = embed_model_path
        self.embed_batch_size = embed_batch_size
        self.embed_device = embed_device
        self.pca_n_components = pca_n_components

        # Initialize components
        self.docker = Docker(docker_kwargs or {})
        self.scorer = Scorer()
        self.scores_to_objective = ScoresToObjective()
        self.embedder = Embedder()
        self.acquisition_function_obj = AcquisitionFunction()
        
        # Initialize surrogate model manager
        self.surrogate_manager = SurrogateModelManager(
            surrogate_model_kwargs=self.surrogate_model_kwargs,
            device=self.device
        )

        # Initialize logging
        if log_dir is not None:
            self.logger = Logger(log_dir=log_dir, overwrite_logs=True)
        else:
            self.logger = None

        # Store testing options
        self.use_dummy_scoring = use_dummy_scoring
        
        # Store raw embeddings for dynamic PCA fitting
        self.raw_embeddings_cache = {}
        self.actual_pca_components = None  # Fixed after first embedding
        
        # Store fitted transformers to ensure consistent dimensionality
        self.fitted_scaler = None
        self.fitted_pca = None
        self.final_input_dim = None

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
        - If single string: treat as single sequence and mutate until we have n_init UNIQUE sequences
        - If list with enough sequences (>= n_init): use first n_init sequences
        - If list with few sequences (< n_init): use all + fill remainder with mutations/random
        
        Ensures we have enough unique sequences for PCA (at least pca_n_components if specified).
        """
        # Determine minimum required sequences for PCA
        # Need at least pca_n_components + 1 samples to get pca_n_components dimensions
        min_required_for_pca = (self.pca_n_components + 1) if self.pca_n_components else 0
        actual_target = max(self.n_init, min_required_for_pca)
        
        if actual_target > self.n_init:
            print(f"Increasing initial population from {self.n_init} to {actual_target} to support {self.pca_n_components} PCA components")
            self.n_init = actual_target  # Update n_init to ensure we generate enough
        
        if self.initial_sequences is None:
            # No initial sequences provided - generate random
            sequences = set()
            while len(sequences) < self.n_init:
                sequences.add(self._random_sequence())
            return list(sequences)
        
        elif isinstance(self.initial_sequences, str):
            # Single sequence provided - mutate until we have enough UNIQUE sequences
            base_sequence = self.initial_sequences
            sequences = {base_sequence}  # Use set to ensure uniqueness
            
            # Keep mutating until we have enough unique sequences
            max_attempts = self.n_init * 10  # Prevent infinite loops
            attempts = 0
            while len(sequences) < self.n_init and attempts < max_attempts:
                # Mutate a random sequence from our current set (more diversity)
                parent = random.choice(list(sequences))
                new_seq = self._mutate_sequence(parent)  # Always forces change now
                sequences.add(new_seq)
                attempts += 1
            
            # If we still don't have enough, fill with random sequences
            while len(sequences) < self.n_init:
                sequences.add(self._random_sequence())
            
            sequence_list = list(sequences)
            return sequence_list
        
        elif isinstance(self.initial_sequences, list):
            if len(self.initial_sequences) >= self.n_init:
                # Enough sequences provided - use first n_init
                return self.initial_sequences[:self.n_init]
            else:
                # Not enough sequences - use all and fill remainder
                sequences = set(self.initial_sequences)  # Ensure uniqueness
                
                # Fill remainder with mutations of existing sequences and random sequences
                max_attempts = self.n_init * 10
                attempts = 0
                while len(sequences) < self.n_init and attempts < max_attempts:
                    if random.random() < 0.7:  # 70% chance to mutate existing sequence
                        parent = random.choice(list(sequences))
                        sequences.add(self._mutate_sequence(parent))
                    else:  # 30% chance to generate completely random sequence
                        sequences.add(self._random_sequence())
                    attempts += 1
                
                # Fill any remaining with random sequences
                while len(sequences) < self.n_init:
                    sequences.add(self._random_sequence())
                
                sequence_list = list(sequences)
                return sequence_list
        
        else:
            raise ValueError("initial_sequences must be None, a string, or a list of strings")

    def _embed(self, peptides: List[str]) -> Dict[str, Any]:
        """
        Embed peptides (ESM or AAIndex), scale, and apply PCA reduction. Returns
        reduced embeddings directly.
        
        Uses cached raw embeddings and recomputes PCA on the complete dataset each time.
        """
        # First, embed any new peptides that aren't in our cache
        new_peptides = [p for p in peptides if p not in self.raw_embeddings_cache]
        
        if new_peptides:
            # Embed new peptides
            if self.embed_method == 'esm':
                new_raw = self.embedder.embed_esm(
                    new_peptides,
                    average=self.embed_average,
                    model_path=self.embed_model_path,
                    batch_size=self.embed_batch_size,
                    filter=True,
                    device=self.embed_device
                )
            elif self.embed_method == 'aaindex':
                new_raw = self.embedder.embed_aaindex(
                    new_peptides,
                    average=self.embed_average,
                    filter=True
                )
            else:
                raise ValueError("embed_method must be 'esm' or 'aaindex'")
            
            # Add to cache
            self.raw_embeddings_cache.update(new_raw)
        
        # Get the subset of embeddings we need for this batch
        batch_raw = {p: self.raw_embeddings_cache[p] for p in peptides}
        
        # Scale all embeddings in the cache (for consistency)
        all_scaled = self.embedder.scale_embeddings(self.raw_embeddings_cache)
        
        # Apply PCA to all cached embeddings
        n_samples = len(all_scaled)
        if n_samples > 0:
            sample_embedding = next(iter(all_scaled.values()))
            if hasattr(sample_embedding, 'shape'):
                if len(sample_embedding.shape) == 2:
                    n_features = sample_embedding.shape[1]
                else:
                    n_features = sample_embedding.shape[0]
            else:
                n_features = len(sample_embedding)
            
            # Set PCA components once and stick with it
            if self.actual_pca_components is None:
                max_components = min(n_samples, n_features)
                self.actual_pca_components = self.pca_n_components
                
                if self.actual_pca_components and self.actual_pca_components >= max_components:
                    print(f"Info: Adjusting PCA components from {self.actual_pca_components} to {max_components - 1} (n_samples={n_samples}, n_features={n_features})")
                    self.actual_pca_components = max_components - 1
                print(f"Info: Fixed PCA components to {self.actual_pca_components} for all future embeddings")
        else:
            self.actual_pca_components = None
        
        # Apply PCA to all cached embeddings
        all_reduced = self.embedder.reduce_embeddings_pca(
            all_scaled,
            n_components=self.actual_pca_components
        )
        
        # Return only the embeddings requested for this batch
        batch_reduced = {p: all_reduced[p] for p in peptides}
        return batch_reduced

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

    def _optimize_hyperparameters(self, embeddings: Dict[str, Any], objectives: Dict[str, float], iteration: Optional[int] = None) -> None:
        """
        Hyperparameter tuning using the surrogate model manager.
        """
        self.surrogate_manager.optimize_hyperparameters(
            embeddings=embeddings,
            objectives=objectives,
            n_trials=self.surrogate_model_kwargs.get('n_trials', 20),
            n_splits=self.surrogate_model_kwargs.get('n_splits', 3),
            random_state=self.surrogate_model_kwargs.get('random_state', 42),
            iteration=iteration
        )

    def _initialize_model(self, embeddings: Dict[str, Any]) -> None:
        """Initialize the model using the surrogate model manager."""
        self.surrogate_manager.initialize_model(embeddings=embeddings)

    def _train_model(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Train the model with validation.
        
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        if self.n_validate is not None and len(embeddings) > self.n_validate:
            return self.surrogate_manager.train_with_validation(
                embeddings=embeddings,
                objectives=objectives,
                n_validate=self.n_validate,
                validation_split=self.validation_split,
                random_state=self.surrogate_model_kwargs.get('random_state', 42)
            )
        else:
            # Not enough data for validation or validation disabled
            self.surrogate_manager.train_model(embeddings, objectives)
            return 0.0, {}

    def _predict(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the surrogate model manager."""
        return self.surrogate_manager.predict(embeddings)

    def _select_top(self, data: Dict[str, float], k: int, acquisition_function: str = "mean", predictions: Optional[Dict[str, tuple]] = None) -> List[str]:
        """
        Select top k sequences based on acquisition function.
        
        Args:
            data: Dictionary of {sequence: objective_value} for sorting fallback
            k: Number of sequences to select
            acquisition_function: Acquisition function to use
            predictions: Dictionary of {sequence: (mean, std)} predictions from model
        
        Returns:
            List of selected sequences
        """
        if acquisition_function == "mean" or predictions is None:
            # Fall back to simple objective-based selection
            return [seq for seq, _ in sorted(data.items(), key=lambda x: x[1], reverse=True)[:k]]
        
        # Use acquisition function
        acquisition_values = self.acquisition_function_obj.compute_acquisition(
            predictions, acquisition_function
        )
        
        # Sort by acquisition value (higher is better)
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _mutate_sequence(self, seq: str) -> str:
        """
        Apply substitution, insertion, or deletion to the sequence based on mutation_rate.
        Always ensures the returned sequence is different from the input.
        """
        max_attempts = 100  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            seq_list = list(seq)
            new_seq = []
            mutation_occurred = False
            
            for aa in seq_list:
                r = random.random()
                if r < self.mutation_rate or (not mutation_occurred and aa == seq_list[-1]):
                    # Choose mutation type
                    op = random.choice(['sub', 'del', 'ins'])
                    if op == 'sub':
                        # substitution
                        new_aa = random.choice([a for a in _AMINO_ACIDS if a != aa])  # Ensure change
                        new_seq.append(new_aa)
                        mutation_occurred = True
                    elif op == 'del' and len(seq_list) > self.min_sequence_length:
                        # deletion: skip this residue (only if we won't go below min length)
                        mutation_occurred = True
                        continue
                    else:
                        # insertion: insert a random AA before current
                        new_seq.append(random.choice(_AMINO_ACIDS))
                        new_seq.append(aa)
                        mutation_occurred = True
                else:
                    new_seq.append(aa)
            
            # Additionally, random insertion at end
            if random.random() < self.mutation_rate or not mutation_occurred:
                new_seq.append(random.choice(_AMINO_ACIDS))
                mutation_occurred = True
            
            # Ensure we have at least one change
            if not mutation_occurred and len(new_seq) > 0:
                # Force a substitution at a random position
                pos = random.randint(0, len(new_seq) - 1)
                new_seq[pos] = random.choice([a for a in _AMINO_ACIDS if a != new_seq[pos]])
                mutation_occurred = True
            
            # Truncate or pad to desired length constraints
            if len(new_seq) > self.max_sequence_length:
                result = ''.join(new_seq[:self.max_sequence_length])
            else:
                # pad by random AAs if too short
                while len(new_seq) < self.min_sequence_length:
                    new_seq.append(random.choice(_AMINO_ACIDS))
                result = ''.join(new_seq)
            
            # Check if result is different from original
            if result != seq:
                return result
            
            attempt += 1
        
        # Last resort fallback: substitute first amino acid
        if len(seq) > 0:
            result_list = list(seq)
            result_list[0] = random.choice([a for a in _AMINO_ACIDS if a != result_list[0]])
            return ''.join(result_list)
        
        return seq  # Should never reach here with valid input

    def _mutate_pool(self, parents: List[str], k_pool: int) -> List[str]:
        """Generate new pool by mutating selected parent sequences."""
        pool = []
        for _ in range(k_pool):
            parent = random.choice(parents)
            pool.append(self._mutate_sequence(parent))
        return pool

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

        # Run through schedule phases
        global_generation = 0
        for phase_index, phase in enumerate(self.schedule, start=1):
            acquisition_function = phase['acquisition']
            generations = phase['generations']
            m_select = phase['m_select']
            k_pool = phase['k_pool']
            
            print(f"\n=== Phase {phase_index}: {acquisition_function} for {generations} generations ===")
            print(f"Selection: {m_select}, Pool: {k_pool}")

            for gen in range(1, generations + 1):
                global_generation += 1
                print(f"\n--- Generation {global_generation} (Phase {phase_index}, Gen {gen}/{generations}) ---")
                
                # Embed and reduce current peptides
                seqs = list(scores.keys())
                reduced_embs = self._embed(seqs)

                # Init fresh model with embeddings to determine input_dim
                self._initialize_model(reduced_embs)

                # Convert scores to objectives
                objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

                # Train surrogate or model only based on interval
                if global_generation % self.hpo_interval == 0:
                    print(f"Re-optimizing hyperparameters (generation {global_generation})")
                    self._optimize_hyperparameters(reduced_embs, objectives, iteration=global_generation)

                print("Training surrogate model...")
                val_loss, metrics = self._train_model(reduced_embs, objectives)

                # Generate new pool via mutation of top M
                # For parent selection, always use objectives (exploitation)
                parents = self._select_top(objectives, m_select)
                print(f"Selected top {len(parents)} parents for mutation")
                
                pool = self._mutate_pool(parents, k_pool)
                print(f"Generated candidate pool of {len(pool)} sequences")

                # Embed and reduce pool
                pool_embs = self._embed(pool)

                # Predict on pool
                preds = self._predict(pool_embs)
                
                # Convert predictions to proper format for acquisition function
                if isinstance(list(preds.values())[0], tuple):
                    # Uncertainty model - predictions are (mean, std) tuples
                    pred_tuples = preds
                else:
                    # Non-uncertainty model - convert to (mean, 0) tuples
                    pred_tuples = {seq: (pred, 0.0) for seq, pred in preds.items()}

                # Select candidates using acquisition function
                candidates = self._select_top(
                    data={seq: pred_tuples[seq][0] for seq in pred_tuples},  # fallback data
                    k=m_select,
                    acquisition_function=acquisition_function,
                    predictions=pred_tuples
                )

                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")

                # Dock, score, and update
                if self.use_dummy_scoring:
                    new_scores = self._dock_and_score_dummy(candidates)
                else:
                    new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                # Calculate new objectives for logging
                new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
                
                if self.logger:
                    # Log new scores and objectives
                    self.logger.log_scores(new_scores, iteration=global_generation, acquisition_name=acquisition_function)
                    self.logger.log_objectives(new_objectives, iteration=global_generation, acquisition_name=acquisition_function)
                    
                    # Log hyperparameters if they were updated
                    if global_generation % self.hpo_interval == 0 and self.surrogate_manager.best_hyperparams:
                        self.logger.log_hyperparameters(
                            iteration=global_generation,
                            hyperparams=self.surrogate_manager.best_hyperparams,
                            model_type=self.surrogate_model_kwargs['model_type'],
                            network_type=self.surrogate_model_kwargs['network_type']
                        )

                # Report generation progress
                current_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
                best_objective = max(current_objectives.values())
                best_sequence = max(current_objectives.items(), key=lambda x: x[1])[0]
                
                print(f"Generation {global_generation} - Best objective: {best_objective:.4f}")
                print(f"Best sequence so far: {best_sequence}")

        # Return final objectives instead of raw scores
        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(final_objectives)}")
        print(f"Best final objective: {max(final_objectives.values()):.4f}")
        
        return final_objectives
