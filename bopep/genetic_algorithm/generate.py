import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import pandas as pd
import numpy as np
        
from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.bayes.acquisition import AcquisitionFunction
from bopep.logging.logger import Logger
from bopep.genetic_algorithm.mutate import PeptideMutator
import torch

class BoGA:
    """
    Genetic algorithm for peptide binder discovery using surrogate modeling.
    
    Use surrogate_model_kwargs to configure the surrogate model, including:
    - model_type, network_type, multi_model (separate models per objective), etc.
    """
    def __init__(
        self,
        target_structure_path: str,
        initial_sequences: Union[str, List[str]],

        n_init : int = 130,

        min_sequence_length: int = 6,
        max_sequence_length: int = 40,
        
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        mutation_rate: float = 0.01,
        # Embedding options
        embed_method: str = 'esm',               # 'esm' or 'aaindex'
        embed_model_path: Optional[str] = None,
        embed_batch_size: int = 128,
        embed_device: Optional[str] = None,
        # PCA reduction
        pca_n_components: int = 128,
        # Validation options
        n_validate: Optional[Union[float, int]] = 0.2,  # Number (int) or fraction (float<1) for validation. None=no validation
        min_validation_samples: int = 20,
        min_training_samples: int = 100,
        # Logging options
        log_dir: Optional[str] = None,
        # Continuation options
        continue_from_logs: Optional[str] = None,
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
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        self.scoring_kwargs = scoring_kwargs
        self.mutation_rate = mutation_rate
        self.n_validate = n_validate
        self.min_validation_samples = min_validation_samples
        self.min_training_samples = min_training_samples
        self.continue_from_logs = continue_from_logs

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
        
        # Enforce fixed PCA dimensions for consistent neural network training
        if self.pca_n_components is None:
            raise ValueError(
                "pca_n_components must be specified to ensure dimensional consistency. "
                "Use a fixed value (e.g., pca_n_components=20) for stable neural network training."
            )

        # Initialize components
        self.docker = Docker(docker_kwargs)
        self.docker.set_target_structure(self.target_structure_path)
        self.scorer = Scorer()
        self.scores_to_objective = ScoresToObjective()
        self.embedder = Embedder()
        self.acquisition_function_obj = AcquisitionFunction()
        
        # Initialize peptide mutator
        self.mutator = PeptideMutator(
            min_sequence_length=self.min_sequence_length,
            max_sequence_length=self.max_sequence_length,
            mutation_rate=self.mutation_rate,
        )
        
        # Initialize surrogate model manager
        self.surrogate_manager = SurrogateModelManager(
            surrogate_model_kwargs=self.surrogate_model_kwargs,
            device=self.device
        )

        # Initialize logging
        if continue_from_logs is not None:
            # When continuing from logs, never overwrite existing logs
            self.logger = Logger(log_dir=continue_from_logs, overwrite_logs=False)
        elif log_dir is not None:
            # For fresh runs, overwrite logs
            self.logger = Logger(log_dir=log_dir, overwrite_logs=True)
        else:
            self.logger = None

        self._evaluated_sequences  = set()

    def _print_leaderboard(self, objectives: Dict[str, Any], generation: int, top_n: int = 5, objective_directions: Dict[str, str] = None):
        """Print leaderboard for both single and multi-objective cases."""
        if not objectives:
            return
        
        print(f"Generation {generation} leaderboard:")
        
        sample_obj = next(iter(objectives.values()))
        if isinstance(sample_obj, dict):
            # Multi-objective case: show top performers for each objective (like optimization.py)
            obj_names = list(sample_obj.keys())
            print(f"Top {top_n} peptides (multiobjective):")
            
            for obj_name in obj_names:
                print(f"\n--- {obj_name} ---")
                
                # Sort by direction
                if objective_directions and obj_name in objective_directions:
                    reverse_sort = objective_directions[obj_name] == "max"
                else:
                    reverse_sort = True  # Default to maximization
                
                sorted_peptides = sorted(objectives.items(), 
                                       key=lambda x: x[1][obj_name], 
                                       reverse=reverse_sort)[:top_n]
                print(f"{'Peptide':<20} | {obj_name:<15}")
                print("-" * 40)
                for peptide, obj_dict in sorted_peptides:
                    print(f"{peptide:<20} | {obj_dict[obj_name]:<15.4f}")
        else:
            # Single objective case: original behavior
            sorted_leaderboard = sorted(objectives.items(), key=lambda x: x[1], reverse=True)[:top_n]
            print(f"{'Peptide':<20} | {'Objective':<10} ")
            print("-" * 60)
            for rank, (seq, obj) in enumerate(sorted_leaderboard, start=1):
                print(f"  {rank}. {seq} - Objective: {obj:.4f}")

    def _generate_initial_sequences(self) -> List[str]:
        return [self.mutator.generate_random_sequence() for _ in range(self.n_init)]

    def _prepare_initial_population(self) -> List[str]:
        """
        Prepare the initial population based on the initial_sequences parameter:
        - If single string: treat as single sequence and mutate until we have n_init UNIQUE sequences
        - If list with enough sequences (>= n_init): use first n_init sequences
        - If list with few sequences (< n_init): use all + fill remainder with mutations
        
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
            raise ValueError("initial_sequences cannot be None. Please provide a sequence or list of sequences.")
        
        elif isinstance(self.initial_sequences, str):
            # Single sequence provided - mutate until we have enough UNIQUE sequences
            base_sequence = self.initial_sequences
            sequences = {base_sequence}
            
            # Keep mutating until we have enough unique sequences
            while len(sequences) < self.n_init:
                # Mutate a random sequence from our current set (more diversity)
                parent = random.choice(list(sequences))
                new_seq = self.mutator.mutate_sequence(parent, self._evaluated_sequences)  # Always forces change now
                sequences.add(new_seq)
            return list(sequences)
        
        elif isinstance(self.initial_sequences, list):
            if len(self.initial_sequences) >= self.n_init:
                return self.initial_sequences
            else:
                sequences = set(self.initial_sequences)  # Ensure uniqueness

                while len(sequences) < self.n_init:
                    parent = random.choice(list(sequences))
                    sequences.add(self.mutator.mutate_sequence(parent, self._evaluated_sequences))
                return list(sequences)
        else:
            raise ValueError("initial_sequences must be None, a string, or a list of strings")

    def _embed_peptides(self, peptides: List[str]) -> Dict[str, Any]:
        """
        Embed, scale, and apply PCA to a list of peptides.
        """
        if not peptides:
            return {}
        
        # Fresh embed all peptides
        if self.embed_method == 'esm':
            raw_embeddings = self.embedder.embed_esm(
                peptides,
                average=self.embed_average,
                model_path=self.embed_model_path,
                batch_size=self.embed_batch_size,
                filter=False,
                device=self.embed_device
            )
        elif self.embed_method == 'aaindex':
            raw_embeddings = self.embedder.embed_aaindex(
                peptides,
                average=self.embed_average,
                filter=False
            )
        else:
            raise ValueError("embed_method must be 'esm' or 'aaindex'")
        
        # Scale and reduce the embeddings
        scaled_embeddings = self.embedder.scale_embeddings(raw_embeddings)
        reduced_embeddings = self.embedder.reduce_embeddings_pca(
            scaled_embeddings,
            n_components=self.pca_n_components
        )
        
        return reduced_embeddings

    def _embed_generation(self, scored_peptides: List[str], candidate_peptides: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Embed and reduce peptides for this generation, ensuring consistent scaling/PCA.
        """
        # Combine all peptides for consistent scaling and PCA
        all_peptides = scored_peptides + candidate_peptides
        all_embeddings = self._embed_peptides(all_peptides)
        
        # Split back into training and candidate sets
        training_embeddings = {p: all_embeddings[p] for p in scored_peptides if p in all_embeddings}
        candidate_embeddings = {p: all_embeddings[p] for p in candidate_peptides if p in all_embeddings}
        
        return training_embeddings, candidate_embeddings

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
            iteration=iteration
        )

    def _train_model(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> Dict[str, Any]:
        """
        Train the model with automatic validation split. Manager decides whether to validate based on sample size.
        """
        return self.surrogate_manager.train_with_validation_split(
            embeddings=embeddings,
            objectives=objectives,
            validation_size=self.n_validate,
            min_training_samples=self.min_training_samples,
            min_validation_samples=self.min_validation_samples
        )

    def _select_top_objectives(
        self, 
        objectives: Dict[str, Any], 
        m_pool: int, 
        objective_directions: Dict[str, str] = None, 
        top_fraction: float = 0.3,
        selection_method: str = "uniform",
        beta: float = 1.0
    ) -> List[str]:
        """
        Select top m_pool sequences based on objectives, with support for sampling from top contenders.
        
        Args:
            objectives: Dictionary mapping sequences to objectives (single value or dict of objectives)
            m_pool: Number of sequences to select
            objective_directions: Optional dict specifying 'max' or 'min' for each objective
            top_fraction: If float < 1.0, fraction of sequences to consider as top candidates.
                         If int >= 1, absolute number of top sequences per objective to consider.
            selection_method: 'uniform' for uniform random sampling, 'exponential' for fitness-weighted sampling
            beta: Selection intensity for exponential method (higher = stronger selection pressure)
        """
        if not objectives:
            return []
        
        # Check if single or multi-objective
        sample_obj = next(iter(objectives.values()))
        
        if isinstance(sample_obj, dict):
            # Multi-objective case: matrix-based sampling approach
            obj_names = list(sample_obj.keys())
            peptides = list(objectives.keys())
            
            # Create ranking matrix: each column is a different objective, sorted by rank
            rankings = {}  # {objective_name: [peptide_list_sorted_by_that_objective]}
            
            for obj_name in obj_names:
                # Sort peptides by this objective
                if objective_directions and obj_name in objective_directions:
                    reverse_sort = objective_directions[obj_name] == "max"
                else:
                    reverse_sort = True  # Default to maximization
                
                sorted_peptides = sorted(peptides, 
                                       key=lambda p: objectives[p][obj_name], 
                                       reverse=reverse_sort)
                rankings[obj_name] = sorted_peptides
            
            # Sample from the top portion of each objective ranking
            top_candidates = set()
            
            # Handle both fraction and integer specifications
            if isinstance(top_fraction, float) and top_fraction < 1.0:
                # Fraction mode: take top fraction of sequences
                n_top_per_objective = max(1, int(len(peptides) * top_fraction))
            elif isinstance(top_fraction, int) or (isinstance(top_fraction, float) and top_fraction >= 1.0):
                # Integer mode: take exact number of top sequences
                n_top_per_objective = int(top_fraction)
            else:
                raise ValueError(f"top_fraction must be a positive number, got {top_fraction}")
            
            for obj_name, ranked_peptides in rankings.items():
                # Take top performers for this objective
                top_for_this_obj = ranked_peptides[:n_top_per_objective]
                top_candidates.update(top_for_this_obj)
            
            # Sample m_pool sequences from the combined top candidates
            top_candidates = list(top_candidates)
            if len(top_candidates) <= m_pool:
                return top_candidates
            
            if selection_method == "uniform":
                return random.sample(top_candidates, m_pool)
            elif selection_method == "exponential":
                # For multi-objective, use aggregated normalized objectives
                return self._exponential_selection(top_candidates, objectives, m_pool, beta, objective_directions)
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")
        else:
            # Single objective case
            sorted_sequences = sorted(objectives.items(), key=lambda x: x[1], reverse=True)
            
            # Handle both fraction and integer specifications
            if isinstance(top_fraction, float) and top_fraction < 1.0:
                # Fraction mode
                n_top = max(1, int(len(sorted_sequences) * top_fraction))
            elif isinstance(top_fraction, int) or (isinstance(top_fraction, float) and top_fraction >= 1.0):
                # Integer mode
                n_top = int(top_fraction)
            else:
                raise ValueError(f"top_fraction must be a positive number, got {top_fraction}")
            
            n_top = max(n_top, m_pool)  # Ensure we have at least m_pool candidates
            top_candidates = [seq for seq, _ in sorted_sequences[:n_top]]
            
            if len(top_candidates) <= m_pool:
                return top_candidates
            
            if selection_method == "uniform":
                return random.sample(top_candidates, m_pool)
            elif selection_method == "exponential":
                # Create objectives dict for top candidates only
                top_objectives = {seq: obj for seq, obj in sorted_sequences[:n_top]}
                return self._exponential_selection(top_candidates, top_objectives, m_pool, beta)
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")

    def _exponential_selection(
        self, 
        candidates: List[str], 
        objectives: Dict[str, Any], 
        m_pool: int, 
        beta: float,
        objective_directions: Dict[str, str] = None
    ) -> List[str]:
        """
        Select sequences using exponential weighting based on normalized objectives.
        
        Args:
            candidates: List of candidate sequences to select from
            objectives: Dictionary mapping sequences to objectives
            m_pool: Number of sequences to select
            beta: Selection intensity (higher = stronger preference for top performers)
            objective_directions: Optional dict for multi-objective direction handling
        """
        sample_obj = objectives[candidates[0]]
        
        if isinstance(sample_obj, dict):
            # Multi-objective: aggregate normalized objectives
            obj_names = list(sample_obj.keys())
            
            # Normalize each objective to [0, 1]
            normalized_objs = {}
            for obj_name in obj_names:
                values = [objectives[seq][obj_name] for seq in candidates]
                min_val, max_val = min(values), max(values)
                
                # Handle direction
                if objective_directions and obj_name in objective_directions:
                    reverse = objective_directions[obj_name] == "max"
                else:
                    reverse = True
                
                for seq in candidates:
                    if seq not in normalized_objs:
                        normalized_objs[seq] = 0.0
                    
                    if max_val > min_val:
                        norm_val = (objectives[seq][obj_name] - min_val) / (max_val - min_val)
                        if not reverse:  # For minimization, invert
                            norm_val = 1.0 - norm_val
                    else:
                        norm_val = 0.5  # All equal
                    
                    normalized_objs[seq] += norm_val / len(obj_names)  # Average across objectives
            
            fitness_values = normalized_objs
        else:
            # Single objective: normalize to [0, 1]
            values = [objectives[seq] for seq in candidates]
            min_val, max_val = min(values), max(values)
            
            fitness_values = {}
            for seq in candidates:
                if max_val > min_val:
                    fitness_values[seq] = (objectives[seq] - min_val) / (max_val - min_val)
                else:
                    fitness_values[seq] = 0.5  # All equal
        
        # Apply exponential weighting
        weights = np.array([np.exp(beta * fitness_values[seq]) for seq in candidates])
        
        # Handle potential overflow/underflow
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            # Fallback to uniform if numerical issues
            return random.sample(candidates, m_pool)
        
        # Normalize to probabilities
        probs = weights / weights.sum()
        
        # Sample without replacement
        selected_indices = np.random.choice(len(candidates), size=m_pool, replace=False, p=probs)
        return [candidates[i] for i in selected_indices]

    def _select_top_predictions(self, predictions: Dict[str, tuple], k: int, acquisition_function: str, acquisition_kwargs: Dict[str, Any] = None) -> List[str]:
        if acquisition_kwargs is None:
            acquisition_kwargs = {}
        acquisition_values = self.acquisition_function_obj.compute_acquisition(
            predictions, 
            acquisition_function, 
            **acquisition_kwargs
        )
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]


    def _load_from_logs(self, log_dir: str) -> Tuple[Dict[str, Dict[str, float]], set, int]:
        """
        Load scores and evaluated sequences from existing log files.
        Returns scores, evaluated_sequences, and last_iteration.
        """

        log_path = Path(log_dir)
        scores_file = log_path / "scores.csv"
        
        if not scores_file.exists():
            raise FileNotFoundError(f"No scores.csv found in {log_dir}")
        
        print(f"Loading scores from {scores_file}")
        df = pd.read_csv(scores_file)
        
        scores = {}
        for _, row in df.iterrows():
            sequence = row['peptide']
            score_columns = [col for col in df.columns 
                           if col not in ['peptide', 'iteration', 'phase', 'timestamp']]
            scores[sequence] = {col: row[col] for col in score_columns}

        evaluated_sequences = set(scores.keys())
        
        # Get the last iteration number to continue from
        last_iteration = df['iteration'].max() if 'iteration' in df.columns and not df.empty else 0
        
        print(f"Loaded {len(scores)} previously evaluated sequences")
        print(f"Last iteration was: {last_iteration}")
        
        return scores, evaluated_sequences, last_iteration

    def run(self, schedule: List[Dict[str, Any]]) -> Dict[str, float]:
        if self.continue_from_logs:
            print(f"Loading previous results from {self.continue_from_logs}")
            scores, self._evaluated_sequences, last_iteration = self._load_from_logs(self.continue_from_logs)
            print("Skipping initial population generation - using loaded sequences")
        else:
            # Fresh start - initial population and embedding/reduction
            init_seqs = self._prepare_initial_population()
            print(f"Generated initial population of {len(init_seqs)} sequences")

            init_reduced = self._embed_peptides(init_seqs)

            # Dock and score initial population
            print("Docking and scoring initial population...")
            scores = self._dock_and_score(init_seqs)
            
            print("Initial scores:")
            print(scores)
            
            last_iteration = 0  # Fresh start

        # Convert initial scores to objectives
        objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

        first_phase_kwargs = schedule[0].get("acquisition_kwargs", {}) if schedule else {}

        if not self.continue_from_logs:
            print("Initial objectives:")
            print(objectives)

            # Log initial population for fresh runs only
            if self.logger:
                self.logger.log_scores(scores, iteration=0, acquisition_name="initial")
                self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")

            # Show initial population leaderboard
            print("Initial population results:")
            self._print_leaderboard(objectives, 0, objective_directions=first_phase_kwargs.get("objective_directions", {}))

            # Initial hyperparameter tuning for fresh runs
            init_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing initial hyperparameters...")
            self._optimize_hyperparameters(init_reduced, objectives)
        else:
            print("Loaded objectives:")
            print(f"Total sequences: {len(objectives)}")
            print("Best existing performers:")
            self._print_leaderboard(objectives, last_iteration, objective_directions=first_phase_kwargs.get("objective_directions", {}))
            
            # For continued runs, optimize hyperparameters on existing data
            existing_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing hyperparameters on existing data...")
            self._optimize_hyperparameters(existing_reduced, objectives)

        # Run through schedule phases
        global_generation = last_iteration  # Continue from last iteration when resuming
        objective_directions = {}  # Default value in case schedule is empty
        for phase_index, phase in enumerate(schedule, start=1):
            acquisition_function = phase['acquisition']
            generations = phase['generations']
            m_select = phase['m_select']
            k_pool = phase['k_pool']
            
            print(f"\n=== Phase {phase_index}: {acquisition_function} for {generations} generations ===")
            print(f"Selection: {m_select}, Pool: {k_pool}")

            for gen in range(1, generations + 1):
                global_generation += 1
                print(f"\n--- Generation {global_generation} (Phase {phase_index}, Gen {gen}/{generations}) ---")
                
                # Convert scores to objectives FIRST to ensure mutation uses current data
                objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

                # Generate new pool via mutation of top M
                # For parent selection, always use objectives (exploitation)
                acquisition_kwargs = phase.get("acquisition_kwargs", {})
                objective_directions = acquisition_kwargs.get("objective_directions", {})
                parents = self._select_top_objectives(
                    objectives, 
                    m_select, 
                    objective_directions=objective_directions,
                    top_fraction=acquisition_kwargs.get("top_fraction", 0.3),
                    selection_method=acquisition_kwargs.get("selection_method", "uniform"),
                    beta=acquisition_kwargs.get("beta", 1.0)
                )
                print(f"Selected top {len(parents)} parents for mutation")
                
                pool = self.mutator.mutate_pool(parents, k_pool, self._evaluated_sequences, objectives)
                print(f"Generated candidate pool of {len(pool)} sequences")

                # Embed scored peptides + candidates together with fresh scaling/PCA
                scored_seqs = list(scores.keys())
                training_embeddings, candidate_embeddings = self._embed_generation(scored_seqs, pool)

                # Init fresh model with training embeddings to determine input_dim
                self.surrogate_manager.initialize_model(embeddings=training_embeddings, objectives=objectives)

                # Train surrogate or model only based on interval
                if global_generation % self.surrogate_model_kwargs['hpo_interval'] == 0:
                    print(f"Re-optimizing hyperparameters (generation {global_generation})")
                    self._optimize_hyperparameters(training_embeddings, objectives, iteration=global_generation)

                print("Training surrogate model...")
                metrics = self._train_model(training_embeddings, objectives)

                # Extract loss - use validation loss if available, otherwise training loss
                loss = metrics["val_mse"] if metrics["val_mse"] is not None else metrics["train_mse"]

                # Predict on pool using candidate embeddings
                preds = self.surrogate_manager.predict(candidate_embeddings)

                # Select candidates using acquisition function
                acquisition_kwargs = phase.get("acquisition_kwargs", {})
                candidates = self._select_top_predictions(preds, m_select, acquisition_function, acquisition_kwargs)

                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")

                # Dock, score, and update
                new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
                
                if self.logger:
                    self.logger.log_model_metrics(loss, iteration=global_generation, metrics=metrics)
                    self.logger.log_scores(new_scores, iteration=global_generation, acquisition_name=acquisition_function)
                    self.logger.log_objectives(new_objectives, iteration=global_generation, acquisition_name=acquisition_function)
                    
                    if global_generation % self.surrogate_model_kwargs['hpo_interval'] == 0 and self.surrogate_manager.best_hyperparams:
                        self.logger.log_hyperparameters(
                            iteration=global_generation,
                            hyperparams=self.surrogate_manager.best_hyperparams,
                            model_type=self.surrogate_model_kwargs['model_type'],
                            network_type=self.surrogate_model_kwargs['network_type']
                        )

                current_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

                self._print_leaderboard(current_objectives, global_generation, objective_directions=objective_directions)

        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(final_objectives)}")
        print("Final leaderboard:")
        self._print_leaderboard(final_objectives, global_generation, objective_directions=objective_directions)
        
        return final_objectives
