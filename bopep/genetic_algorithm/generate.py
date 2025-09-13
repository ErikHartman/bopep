import random
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

import numpy as np
from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.search.acquisition_functions import AcquisitionFunction
from bopep.logging.logger import Logger
import torch

_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

class BoGA:
    """
    Genetic algorithm for peptide binder discovery using surrogate modeling.
    """
    def __init__(
        self,
        target_structure_path: str,
        schedule: List[Dict[str, Any]],
        initial_sequences: Union[str, List[str]],

        n_init : int = 50,

        min_sequence_length: int = 6,
        max_sequence_length: int = 40,
        
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
    ):
        self.initial_sequences = initial_sequences
        
        # Validate surrogate model config
        self.surrogate_model_kwargs = surrogate_model_kwargs
        _validate_surrogate_model_kwargs(self.surrogate_model_kwargs)

        # Store GA parameters
        self.target_structure_path = target_structure_path
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.n_init = n_init
        self.schedule = schedule
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs
        self.scoring_kwargs = scoring_kwargs
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
        self.docker = Docker(docker_kwargs)
        self.docker.set_target_structure(self.target_structure_path)
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

        self._evaluated_sequences  = set()
        
        # Store raw embeddings for dynamic PCA fitting
        self.raw_embeddings_cache = {}

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
                    filter=False,
                    device=self.embed_device
                )
            elif self.embed_method == 'aaindex':
                new_raw = self.embedder.embed_aaindex(
                    new_peptides,
                    average=self.embed_average,
                    filter=False
                )
            else:
                raise ValueError("embed_method must be 'esm' or 'aaindex'")
            
            # Add to cache
            self.raw_embeddings_cache.update(new_raw)
        
        # Scale all embeddings in the cache (for consistency)
        all_scaled = self.embedder.scale_embeddings(self.raw_embeddings_cache)
        
        # Apply PCA to all cached embeddings
        all_reduced = self.embedder.reduce_embeddings_pca(
            all_scaled,
            n_components=self.pca_n_components
        )
        
        # Return only the embeddings requested for this batch
        batch_reduced = {p: all_reduced[p] for p in peptides}
        return batch_reduced

    def _dock_and_score(self, sequences: List[str]) -> Dict[str, float]:
        dock_dirs = self.docker.dock_peptides(sequences)
        scores =  self.scorer.score_batch(
            scores_to_include=self.scoring_kwargs.get('scores_to_include', []),
            inputs=dock_dirs,
            input_type='processed_dir',
            binding_site_residue_indices=self.scoring_kwargs.get('binding_site_residue_indices'),
            n_jobs=self.scoring_kwargs.get('n_jobs', 12),
            binding_site_distance_threshold=self.scoring_kwargs.get('binding_site_distance_threshold', 5),
            required_n_contact_residues=self.scoring_kwargs.get('required_n_contact_residues', 5),
        )
        self._evaluated_sequences.update(sequences)
        return scores

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
        return self.surrogate_manager.predict(embeddings)

    def _select_top_objectives(self, objectives: Dict[str, float], k: int) -> List[str]:
        return [seq for seq, _ in sorted(objectives.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _select_top_predictions(self, predictions: Dict[str, tuple], k: int, acquisition_function: str) -> List[str]:
        acquisition_values = self.acquisition_function_obj.compute_acquisition(predictions, acquisition_function)
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _mutate_sequence(self, seq: str) -> str:
        """
        - n_edits ~ Poisson(len(seq) * mutation_rate), min 1
        - ops: substitution, deletion, insertion
        - respects min/max length at every step
        - guarantees a novel child not seen in _evaluated_sequences
        """
        max_attempts = 10_000
        ops_space = np.array(["sub", "del", "ins"], dtype=object)

        for _ in range(max_attempts):
            child = list(seq)

            lam = max(1e-9, len(child) * self.mutation_rate)
            n_edits = int(np.random.poisson(lam))
            if n_edits < 1:
                n_edits = 1

            for _ in range(n_edits):
                can_del = len(child) > self.min_sequence_length
                can_ins = len(child) < self.max_sequence_length

                # probabilities for [sub, del, ins], normalized
                probs = np.array([1.0, 1.0 if can_del else 0.0, 1.0 if can_ins else 0.0], dtype=float)
                probs /= probs.sum()  # if both del/ins illegal, this becomes [1,0,0]

                op = np.random.choice(ops_space, p=probs)

                if op == "sub":
                    i = np.random.randint(len(child))
                    old = child[i]
                    # choose a different amino acid
                    # fast path avoids while-loop
                    choices = [a for a in _AMINO_ACIDS if a != old]
                    child[i] = choices[np.random.randint(len(choices))]

                elif op == "del":
                    if len(child) <= self.min_sequence_length:
                        # degrade to substitution
                        i = np.random.randint(len(child))
                        old = child[i]
                        choices = [a for a in _AMINO_ACIDS if a != old]
                        child[i] = choices[np.random.randint(len(choices))]
                    else:
                        i = np.random.randint(len(child))
                        del child[i]

                else:  # "ins"
                    if len(child) >= self.max_sequence_length:
                        # degrade to substitution
                        i = np.random.randint(len(child))
                        old = child[i]
                        choices = [a for a in _AMINO_ACIDS if a != old]
                        child[i] = choices[np.random.randint(len(choices))]
                    else:
                        # insert at a random gap [0..len]
                        i = np.random.randint(len(child) + 1)
                        child.insert(i, _AMINO_ACIDS[np.random.randint(len(_AMINO_ACIDS))])

            result = "".join(child)
            if self.min_sequence_length <= len(result) <= self.max_sequence_length \
            and result != seq and result not in self._evaluated_sequences:
                return result

        return seq

    def _mutate_pool(self, parents: List[str], k_pool: int) -> List[str]:
        """
        - m_select: how many top-performing *parents* are selected from the evaluated set.
        - k_pool:   how many *offspring candidates* to generate in this mutation step.
        """
        # Generate a unique pool and avoid sequences we’ve already evaluated
        pool: set[str] = set()
        attempts = 0
        max_attempts = max(k_pool * 20, 10_000)

        while len(pool) < k_pool and attempts < max_attempts:
            parent = random.choice(parents)
            child = self._mutate_sequence(parent)
            if child not in self._evaluated_sequences:
                pool.add(child)
            attempts += 1

        return list(pool)

    def run(self) -> Dict[str, float]:
        # Initial population and embedding/reduction
        init_seqs = self._prepare_initial_population()
        print(f"Generated initial population of {len(init_seqs)} sequences")

        init_reduced = self._embed(init_seqs)

        # Dock and score initial population
        print("Docking and scoring initial population...")
        scores = self._dock_and_score(init_seqs)
        
        print("Initial scores:")
        print(scores)

        # Convert initial scores to objectives
        objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

        print("Initial objectives:")
        print(objectives)

        # Log initial population
        if self.logger:
            self.logger.log_scores(scores, iteration=0, acquisition_name="initial")
            self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")

        print(f"Initial population - best objective: {max(objectives.values()):.4f}")

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
                parents = self._select_top_objectives(objectives, m_select)
                print(f"Selected top {len(parents)} parents for mutation")
                
                pool = self._mutate_pool(parents, k_pool)
                print(f"Generated candidate pool of {len(pool)} sequences")

                # Embed and reduce pool
                pool_embs = self._embed(pool)

                # Predict on pool
                preds = self._predict(pool_embs)
                
                # Select candidates using acquisition function
                candidates = self._select_top_predictions(preds, m_select, acquisition_function)

                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")

                # Dock, score, and update
                new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
                
                if self.logger:
                    # Log new scores and objectives
                    self.logger.log_model_metrics(val_loss, iteration=global_generation, metrics=metrics)
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

                print(f"Generation {global_generation} leaderboard:")
                sorted_leaderboard = sorted(current_objectives.items(), key=lambda x: x[1], reverse=True)[:5]
                for rank, (seq, obj) in enumerate(sorted_leaderboard, start=1):
                    print(f"  {rank}. {seq} - Objective: {obj:.4f}")


        # Return final objectives instead of raw scores
        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(final_objectives)}")
        print(f"Best final objective: {max(final_objectives.values()):.4f}")
        
        return final_objectives
