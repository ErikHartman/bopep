import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import pandas as pd
        
from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.scorer import Scorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.search.acquisition_functions import AcquisitionFunction
from bopep.logging.logger import Logger
from bopep.genetic_algorithm.mutate import PeptideMutator
import torch

class BoGA:
    """
    Genetic algorithm for peptide binder discovery using surrogate modeling.
    """
    def __init__(
        self,
        target_structure_path: str,
        schedule: List[Dict[str, Any]],
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
        self.schedule = schedule
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
            mode="uniform",  # Default mode, will be set per phase
            tau=1.0,         # Default tau
            lam=0.3,         # Default lam
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

    def _train_model(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
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

    def _select_top_objectives(self, objectives: Dict[str, float], k: int) -> List[str]:
        return [seq for seq, _ in sorted(objectives.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _select_top_predictions(self, predictions: Dict[str, tuple], k: int, acquisition_function: str) -> List[str]:
        acquisition_values = self.acquisition_function_obj.compute_acquisition(predictions, acquisition_function)
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]

    def _configure_mutation_for_phase(self, phase: Dict[str, Any], phase_index: int, objectives: Dict[str, float]) -> None:
        """
        Validate and configure mutation parameters for a phase.
        """
        mutation_mode = phase["mutation_mode"]
        mutation_tau = phase.get('mutation_tau', 1.0)  # Default tau
        mutation_lam = phase.get('mutation_lam', 0.3)  # Default lam
        
        # Update mutator parameters
        self.mutator.set_mode(mutation_mode)
        self.mutator.tau = max(1e-6, float(mutation_tau))
        self.mutator.lam = float(mutation_lam)
        
        print(f"Mutation: mode={mutation_mode}, tau={mutation_tau:.3f}, lam={mutation_lam:.3f}")

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

    def run(self) -> Dict[str, float]:
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

        if not self.continue_from_logs:
            print("Initial objectives:")
            print(objectives)

            # Log initial population for fresh runs only
            if self.logger:
                self.logger.log_scores(scores, iteration=0, acquisition_name="initial")
                self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")

            print(f"Initial population - best objective: {max(objectives.values()):.4f}")

            # Initial hyperparameter tuning for fresh runs
            init_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing initial hyperparameters...")
            self._optimize_hyperparameters(init_reduced, objectives)
        else:
            print("Loaded objectives:")
            print(f"Total sequences: {len(objectives)}")
            print(f"Best existing objective: {max(objectives.values()):.4f}")
            
            # For continued runs, optimize hyperparameters on existing data
            existing_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing hyperparameters on existing data...")
            self._optimize_hyperparameters(existing_reduced, objectives)

        # Run through schedule phases
        global_generation = last_iteration  # Continue from last iteration when resuming
        for phase_index, phase in enumerate(self.schedule, start=1):
            acquisition_function = phase['acquisition']
            generations = phase['generations']
            m_select = phase['m_select']
            k_pool = phase['k_pool']
            
            # Configure mutation parameters for this phase
            self._configure_mutation_for_phase(phase, phase_index, objectives)
            
            print(f"\n=== Phase {phase_index}: {acquisition_function} for {generations} generations ===")
            print(f"Selection: {m_select}, Pool: {k_pool}")

            for gen in range(1, generations + 1):
                global_generation += 1
                print(f"\n--- Generation {global_generation} (Phase {phase_index}, Gen {gen}/{generations}) ---")
                
                # Generate new pool via mutation of top M
                # For parent selection, always use objectives (exploitation)
                parents = self._select_top_objectives(objectives, m_select)
                print(f"Selected top {len(parents)} parents for mutation")
                
                pool = self.mutator.mutate_pool(parents, k_pool, self._evaluated_sequences, objectives)
                print(f"Generated candidate pool of {len(pool)} sequences")

                # Embed scored peptides + candidates together with fresh scaling/PCA
                scored_seqs = list(scores.keys())
                training_embeddings, candidate_embeddings = self._embed_generation(scored_seqs, pool)

                # Init fresh model with training embeddings to determine input_dim
                self.surrogate_manager.initialize_model(embeddings=training_embeddings)

                # Convert scores to objectives
                objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

                # Train surrogate or model only based on interval
                if global_generation % self.surrogate_model_kwargs['hpo_interval'] == 0:
                    print(f"Re-optimizing hyperparameters (generation {global_generation})")
                    self._optimize_hyperparameters(training_embeddings, objectives, iteration=global_generation)

                print("Training surrogate model...")
                val_loss, metrics = self._train_model(training_embeddings, objectives)

                # Predict on pool using candidate embeddings
                preds = self.surrogate_manager.predict(candidate_embeddings)

                # Select candidates using acquisition function
                candidates = self._select_top_predictions(preds, m_select, acquisition_function)

                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")

                # Dock, score, and update
                new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
                
                if self.logger:
                    self.logger.log_model_metrics(val_loss, iteration=global_generation, metrics=metrics)
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

                print(f"Generation {global_generation} leaderboard:")
                sorted_leaderboard = sorted(current_objectives.items(), key=lambda x: x[1], reverse=True)[:5]
                for rank, (seq, obj) in enumerate(sorted_leaderboard, start=1):
                    print(f"  {rank}. {seq} - Objective: {obj:.4f}")

        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(final_objectives)}")
        print(f"Best final objective: {max(final_objectives.values()):.4f}")
        
        return final_objectives
