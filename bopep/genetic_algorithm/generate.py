import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import pandas as pd
import numpy as np
        
from bopep.docking.docker import Docker
from bopep.folding.alphafold_monomer import AlphaFoldMonomer
from bopep.embedding.embedder import Embedder
from bopep.scoring.complex_scorer import ComplexScorer
from bopep.scoring.monomer_scorer import MonomerScorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs
from bopep.bayes.acquisition import AcquisitionFunction
from bopep.logging.logger import Logger
from bopep.genetic_algorithm.mutate import PeptideMutator
from bopep.config import Config
import torch


class BoGA:
    """
    Evoluationary algorithm for protein discovery using surrogate modeling.
    """
    def __init__(
        self,
        initial_sequences: Union[str, List[str]],
        mode: str,
        objective_function: Callable = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        target_structure_path: Optional[str] = None,

        n_init: Optional[int] = None,

        min_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,

        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        mutation_rate: Optional[float] = None,
        # Embedding options
        embed_method: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        embed_device: Optional[str] = None,
        # PCA reduction
        pca_n_components: Optional[int] = None,
        # Validation options
        n_validate: Optional[Union[float, int]] = None,
        min_validation_samples: Optional[int] = None,
        min_training_samples: Optional[int] = None,
        # Selection options
        m_select: Optional[int] = None,
        k_propose: Optional[int] = None,
        selection_method: Optional[str] = None,
        top_n_or_frac: Optional[float] = None,
        beta: Optional[float] = None,
        adalead_k: Optional[float] = None,
        selection_pool: Optional[str] = None,
        # Logging options
        log_dir: Optional[str] = None,
        # Continuation options
        continue_from_logs: Optional[str] = None,
        # Config object 
        config: Optional[Config] = None,
    ):
        # Initialize or load config
        if config is None:
            config = Config(script="BoGA")  # Load defaults
        self.config = config
        
        # Get flattened config for easy parameter access
        cfg = self.config.flatten()
        
        # Helper function to get parameter value (user override > config)
        def get_param(user_val, config_key):
            return user_val if user_val is not None else cfg.get(config_key)
        
        self.initial_sequences = initial_sequences
        
        # Get mode (now REQUIRED parameter, not from config)
        if mode is None:
            raise ValueError("'mode' is a required parameter. Must be 'binding', 'unconditional', or 'sequence'.")
        self.mode = mode.lower()
        if self.mode not in ['binding', 'unconditional', 'sequence']:
            raise ValueError(f"mode must be 'binding', 'unconditional', or 'sequence', got '{self.mode}'")
        
        # Validate mode-specific requirements
        self.target_structure_path = target_structure_path
        if self.mode == 'binding' and self.target_structure_path is None:
            raise ValueError("target_structure_path is required for mode='binding'")
        
        # Get GA parameters from config or user overrides
        self.n_init = get_param(n_init, 'n_init')
        self.min_sequence_length = get_param(min_sequence_length, 'min_sequence_length')
        self.max_sequence_length = get_param(max_sequence_length, 'max_sequence_length')
        self.mutation_rate = get_param(mutation_rate, 'mutation_rate')
        
        # Get surrogate model kwargs
        if surrogate_model_kwargs is not None:
            self.surrogate_model_kwargs = surrogate_model_kwargs
        else:
            # Extract from flattened config
            self.surrogate_model_kwargs = {
                'model_type': cfg.get('surrogate_model.model_type'),
                'network_type': cfg.get('surrogate_model.network_type'),
                'n_trials': cfg.get('surrogate_model.n_trials'),
                'n_splits': cfg.get('surrogate_model.n_splits'),
                'hpo_interval': cfg.get('surrogate_model.hpo_interval'),
            }
        _validate_surrogate_model_kwargs(self.surrogate_model_kwargs)
        
        # Get scoring kwargs
        if scoring_kwargs is not None:
            self.scoring_kwargs = scoring_kwargs
        else:
            # Extract all scoring.* keys from flattened config
            self.scoring_kwargs = {
                k.replace('scoring.', ''): v 
                for k, v in cfg.items() 
                if k.startswith('scoring.')
            }
        
        # Objective function (must be provided by user, not in config)
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        
        # Get validation parameters
        self.n_validate = get_param(n_validate, 'validation.n_validate')
        self.min_validation_samples = get_param(min_validation_samples, 'validation.min_validation_samples')
        self.min_training_samples = get_param(min_training_samples, 'validation.min_training_samples')
        
        # Get selection parameters from config or user overrides
        self.m_select = get_param(m_select, 'selection.m_select')
        self.k_propose = get_param(k_propose, 'selection.k_propose')
        self.selection_method = get_param(selection_method, 'selection.method')
        self.top_n_or_frac = get_param(top_n_or_frac, 'selection.top_n_or_frac')
        self.beta = get_param(beta, 'selection.beta')
        self.adalead_k = get_param(adalead_k, 'selection.adalead_k')
        self.selection_pool = get_param(selection_pool, 'selection.selection_pool')
        
        # Logging options
        self.continue_from_logs = get_param(continue_from_logs, 'logging.continue_from_logs')
        log_dir = get_param(log_dir, 'logging.log_dir')

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get embedding configuration
        self.embed_method = get_param(embed_method, 'embedding.method').lower()
        self.embed_model_path = get_param(embed_model_path, 'embedding.model_path')
        self.embed_batch_size = get_param(embed_batch_size, 'embedding.batch_size')
        self.embed_device = get_param(embed_device, 'embedding.device')
        self.pca_n_components = get_param(pca_n_components, 'embedding.pca_n_components')
        
        # Auto-detect embedding averaging based on network type
        network_type = self.surrogate_model_kwargs.get('network_type', 'mlp').lower()
        if network_type == "mlp":
            self.embed_average = True
        else:
            self.embed_average = False
        
        # Enforce fixed PCA dimensions for consistent neural network training
        if self.pca_n_components is None:
            raise ValueError(
                "pca_n_components must be specified to ensure dimensional consistency. "
                "Use a fixed value (e.g., pca_n_components=20) for stable neural network training."
            )
        
        # Get docker/folding kwargs from config or user overrides
        if docker_kwargs is not None:
            final_docker_kwargs = docker_kwargs
        else:
            if self.mode == 'binding':
                # Extract all docker.* keys from flattened config
                final_docker_kwargs = {
                    k.replace('docker.', ''): v 
                    for k, v in cfg.items() 
                    if k.startswith('docker.')
                }
            else:  # unconditional mode uses folding kwargs
                # Extract all folding.* keys from flattened config
                final_docker_kwargs = {
                    k.replace('folding.', ''): v 
                    for k, v in cfg.items() 
                    if k.startswith('folding.')
                }
        print("Using docker/folding kwargs:")
        print(final_docker_kwargs)
        
        # Initialize components based on mode
        if self.mode == 'binding':
            self.docker = Docker(final_docker_kwargs)
            self.docker.set_target_structure(self.target_structure_path)
            self.folder = None
            self.scorer = ComplexScorer()
        elif self.mode == 'unconditional':
            self.docker = None
            self.folder = AlphaFoldMonomer(
                output_dir=final_docker_kwargs['output_dir'],
                num_models=final_docker_kwargs.get('num_models', 5),
                num_recycles=final_docker_kwargs.get('num_recycles', 3),
                save_raw=final_docker_kwargs.get('save_raw', False),
                force=final_docker_kwargs.get('force', False),
            )
            self.scorer = MonomerScorer()
        else:  # mode == 'sequence'
            self.docker = None
            self.folder = None
            self.scorer = MonomerScorer()
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
        # Set max_seq_len based on max_sequence_length if not already specified
        if 'max_seq_len' not in self.surrogate_model_kwargs:
            self.surrogate_model_kwargs['max_seq_len'] = self.max_sequence_length
        
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

    def _print_leaderboard(self, objectives: Dict[str, Any], generation: int, print_n: int = 5, objective_directions: Dict[str, str] = None):
        """Print leaderboard for both single and multi-objective cases."""
        if not objectives:
            return
        
        print(f"Generation {generation} leaderboard:")
        
        sample_obj = next(iter(objectives.values()))
        if isinstance(sample_obj, dict):
            # Multi-objective case: show top performers for each objective (like optimization.py)
            obj_names = list(sample_obj.keys())
            print(f"Top {print_n} peptides (multiobjective):")
            
            for obj_name in obj_names:
                print(f"\n--- {obj_name} ---")
                
                # Sort by direction
                if objective_directions and obj_name in objective_directions:
                    reverse_sort = objective_directions[obj_name] == "max"
                else:
                    reverse_sort = True  # Default to maximization
                
                sorted_peptides = sorted(objectives.items(), 
                                       key=lambda x: x[1][obj_name], 
                                       reverse=reverse_sort)[:print_n]
                print(f"{'Peptide':<20} | {obj_name:<15}")
                print("-" * 40)
                for peptide, obj_dict in sorted_peptides:
                    print(f"{peptide:<20} | {obj_dict[obj_name]:<15.4f}")
        else:
            # Single objective case: original behavior
            sorted_leaderboard = sorted(objectives.items(), key=lambda x: x[1], reverse=True)[:print_n]
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
        if self.mode == 'unconditional':
            # Unconditional mode: fold monomers and score intrinsic properties
            fold_dirs = self.folder.fold(sequences)
            scores = self.scorer.score_batch(
                scores_to_include=self.scoring_kwargs.get('scores_to_include', []),
                inputs=fold_dirs,
                input_type='processed_dir',
                n_jobs=self.scoring_kwargs.get('n_jobs', 12),
            )
        elif self.mode == 'sequence':
            # Sequence mode: fast sequence-only scoring without folding
            scores = self.scorer.score_batch(
                scores_to_include=self.scoring_kwargs.get('scores_to_include', []),
                inputs=sequences,
                input_type='sequence',
                n_jobs=self.scoring_kwargs.get('n_jobs', 12),
            )
        else:  # mode == 'binding'
            # Binding mode: dock and score complexes
            dock_dirs = self.docker.dock_peptides(sequences)
            scores = self.scorer.score_batch(
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
        top_n_or_frac: float = 0.3,
        selection_method: str = "uniform",
        beta: float = 1.0,
        adalead_k: float = 0.05,
        last_batch_objectives: Optional[Dict[str, Any]] = None,
        selection_pool: str = "full_history"
    ) -> List[str]:
        """
        Select top m_pool sequences based on objectives, with support for sampling from top contenders.
        """
        if not objectives:
            return []
        
        # Determine which objectives to select from based on selection_pool
        if selection_pool == "last_batch" and last_batch_objectives:
            selection_objectives = last_batch_objectives
        elif selection_pool == "full_history":
            selection_objectives = objectives
        else:
            # Invalid selection_pool or last_batch not available, default to full history
            selection_objectives = objectives
        
        # Check if single or multi-objective
        sample_obj = next(iter(selection_objectives.values()))
        
        if isinstance(sample_obj, dict):
            # Multi-objective case: matrix-based sampling approach
            obj_names = list(sample_obj.keys())
            peptides = list(selection_objectives.keys())
            
            # Create ranking matrix: each column is a different objective, sorted by rank
            rankings = {}  # {objective_name: [peptide_list_sorted_by_that_objective]}
            
            for obj_name in obj_names:
                # Sort peptides by this objective
                if objective_directions and obj_name in objective_directions:
                    reverse_sort = objective_directions[obj_name] == "max"
                else:
                    reverse_sort = True  # Default to maximization
                
                sorted_peptides = sorted(peptides, 
                                       key=lambda p: selection_objectives[p][obj_name], 
                                       reverse=reverse_sort)
                rankings[obj_name] = sorted_peptides
            
            # Sample from the top portion of each objective ranking
            top_candidates = set()
            
            # Handle both fraction and integer specifications
            if isinstance(top_n_or_frac, float) and top_n_or_frac < 1.0:
                # Fraction mode: take top fraction of sequences
                n_top_per_objective = max(1, int(len(peptides) * top_n_or_frac))
            elif isinstance(top_n_or_frac, int) or (isinstance(top_n_or_frac, float) and top_n_or_frac >= 1.0):
                # Integer mode: take exact number of top sequences
                n_top_per_objective = int(top_n_or_frac)
            else:
                raise ValueError(f"top_n_or_frac must be a positive number, got {top_n_or_frac}")
            
            for obj_name, ranked_peptides in rankings.items():
                # Take top performers for this objective
                top_for_this_obj = ranked_peptides[:n_top_per_objective]
                top_candidates.update(top_for_this_obj)
            
            if selection_method == "adalead":
                raise ValueError("AdaLead selection is only supported for single-objective optimization")
            
            top_candidates = list(top_candidates)
            if len(top_candidates) <= m_pool:
                return top_candidates
            
            if selection_method == "uniform":
                return random.sample(top_candidates, m_pool)
            elif selection_method == "exponential":
                # For multi-objective, use aggregated normalized objectives
                return self._exponential_selection(top_candidates, selection_objectives, m_pool, beta, objective_directions)
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")
        else:
            sorted_sequences = sorted(selection_objectives.items(), key=lambda x: x[1], reverse=True)
            
            # Handle both fraction and integer specifications
            if isinstance(top_n_or_frac, float) and top_n_or_frac < 1.0:
                n_top = max(1, int(len(sorted_sequences) * top_n_or_frac))
            elif isinstance(top_n_or_frac, int) or (isinstance(top_n_or_frac, float) and top_n_or_frac >= 1.0):
                n_top = int(top_n_or_frac)
            else:
                raise ValueError(f"top_n_or_frac must be a positive number, got {top_n_or_frac}")
            
            n_top = max(n_top, m_pool)
            top_candidates = [seq for seq, _ in sorted_sequences[:n_top]]
            
            if len(top_candidates) <= m_pool:
                return top_candidates
            
            if selection_method == "uniform":
                return random.sample(top_candidates, m_pool)
            elif selection_method == "exponential":
                top_objectives = {seq: obj for seq, obj in sorted_sequences[:n_top]}
                return self._exponential_selection(top_candidates, top_objectives, m_pool, beta)
            elif selection_method == "adalead":
                return self._adalead_selection(selection_objectives, adalead_k)
            else:
                raise ValueError(f"Unknown selection_method: {selection_method}")

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
        # Update config with actually-used values and save to output directory
        if self.logger and hasattr(self.logger, 'log_dir') and isinstance(self.logger.log_dir, (str, Path)):
            # Update config with all instance values that may have been user-overridden
            self.config.update_from_used_values(
                mode=self.mode,
                n_init=self.n_init,
                min_sequence_length=self.min_sequence_length,
                max_sequence_length=self.max_sequence_length,
                mutation_rate=self.mutation_rate,
                **{'selection.m_select': self.m_select},
                **{'selection.k_propose': self.k_propose},
                **{'selection.selection_method': self.selection_method},
                **{'selection.top_n_or_frac': self.top_n_or_frac},
                **{'selection.beta': self.beta},
                **{'selection.adalead_k': self.adalead_k},
                **{'selection.selection_pool': self.selection_pool},
                **{'embedding.method': self.embed_method},
                **{'embedding.model_path': self.embed_model_path},
                **{'embedding.batch_size': self.embed_batch_size},
                **{'embedding.device': self.embed_device},
                **{'embedding.pca_n_components': self.pca_n_components},
                **{'validation.n_validate': self.n_validate},
                **{'validation.min_validation_samples': self.min_validation_samples},
                **{'validation.min_training_samples': self.min_training_samples},
            )
            
            output_dir = self.logger.log_dir
            config_path = self.config.save(output_dir)
            print(f"Saved configuration to {config_path}")
        
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
            print("Evaluating initial population...")
            scores = self._dock_and_score(init_seqs)
            last_iteration = 0  # Fresh start

        # Convert initial scores to objectives
        objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

        # Get objective directions for leaderboard display (from scoring_kwargs if available)
        objective_directions = self.scoring_kwargs.get("objective_directions", {}) if self.scoring_kwargs else {}

        if not self.continue_from_logs:
            print("Initial objectives:")
            print(objectives)

            # Log initial population for fresh runs only
            if self.logger:
                self.logger.log_scores(scores, iteration=0, acquisition_name="initial")
                self.logger.log_objectives(objectives, iteration=0, acquisition_name="initial")

            # Show initial population leaderboard
            print("Initial population results:")
            self._print_leaderboard(objectives, 0, objective_directions=objective_directions)

            # Initial hyperparameter tuning for fresh runs
            init_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing initial hyperparameters...")
            self._optimize_hyperparameters(init_reduced, objectives)
            if self.logger:
                self.logger.log_hyperparameters(
                                iteration=0,
                                hyperparams=self.surrogate_manager.best_hyperparams,
                                model_type=self.surrogate_model_kwargs['model_type'],
                                network_type=self.surrogate_model_kwargs['network_type']
                            )

        else:
            print("Loaded objectives:")
            print(f"Total sequences: {len(objectives)}")
            print("Best existing performers:")
            self._print_leaderboard(objectives, last_iteration, objective_directions=objective_directions)
            
            # For continued runs, optimize hyperparameters on existing data
            existing_reduced = self._embed_peptides(list(scores.keys()))
            print("Optimizing hyperparameters on existing data...")
            self._optimize_hyperparameters(existing_reduced, objectives)

        # Run through schedule phases
        global_generation = last_iteration  # Continue from last iteration when resuming
        last_batch_objectives = None  # Track last batch for AdaLead
        
        for phase_index, phase in enumerate(schedule, start=1):
            acquisition_function = phase['acquisition']
            generations = phase['generations']
            
            print(f"\n=== Phase {phase_index}: {acquisition_function} for {generations} generations ===")
            print(f"Selection: {self.m_select}, Pool: {self.k_propose}")

            for gen in range(1, generations + 1):
                global_generation += 1
                print(f"\n--- Generation {global_generation} (Phase {phase_index}, Gen {gen}/{generations}) ---")
                
                # Convert scores to objectives FIRST to ensure mutation uses current data
                objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

                # Get acquisition_kwargs from phase for acquisition-specific parameters
                acquisition_kwargs = phase.get("acquisition_kwargs", {})
                
                # For objective_directions, check scoring_kwargs first (multi-objective case)
                objective_directions = self.scoring_kwargs.get("objective_directions", {})
                
                parents = self._select_top_objectives(
                    objectives, 
                    self.m_select, 
                    objective_directions=objective_directions,
                    top_n_or_frac=self.top_n_or_frac,
                    selection_method=self.selection_method,
                    beta=self.beta,
                    adalead_k=self.adalead_k,
                    last_batch_objectives=last_batch_objectives,
                    selection_pool=self.selection_pool
                )
                print(f"Selected top {len(parents)} parents for mutation")
                
                pool = self.mutator.mutate_pool(parents, self.k_propose, self._evaluated_sequences, objectives)
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
                candidates = self._select_top_predictions(preds, self.m_select, acquisition_function, acquisition_kwargs)

                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")

                # Dock, score, and update
                new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                new_objectives = self.scores_to_objective.create_objective(new_scores, self.objective_function, **self.objective_function_kwargs)
                last_batch_objectives = new_objectives  # Track for AdaLead
                
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
        """
        sample_obj = objectives[candidates[0]]
        
        if isinstance(sample_obj, dict):
            obj_names = list(sample_obj.keys())
            normalized_objs = {}
            for obj_name in obj_names:
                values = [objectives[seq][obj_name] for seq in candidates]
                min_val, max_val = min(values), max(values)

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
            values = [objectives[seq] for seq in candidates]
            min_val, max_val = min(values), max(values)
            
            fitness_values = {}
            for seq in candidates:
                if max_val > min_val:
                    fitness_values[seq] = (objectives[seq] - min_val) / (max_val - min_val)
                else:
                    fitness_values[seq] = 0.5  # All equal
        weights = np.array([np.exp(beta * fitness_values[seq]) for seq in candidates])
        if np.any(np.isinf(weights)) or np.any(np.isnan(weights)):
            return random.sample(candidates, m_pool)
        probs = weights / weights.sum()
        selected_indices = np.random.choice(len(candidates), size=m_pool, replace=False, p=probs)
        return [candidates[i] for i in selected_indices]

    def _adalead_selection(
        self,
        objectives: Dict[str, Any],
        adalead_k: float,
        last_batch_objectives: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        AdaLead selection: adaptive threshold-based resampling (single-objective only).
        """
        if not objectives:
            return []
        
        if not (0 <= adalead_k <= 1):
            raise ValueError(f"adalead_k must be in [0, 1], got {adalead_k}")
        
        # Check for multi-objective (reject early)
        sample_obj = next(iter(objectives.values()))
        if isinstance(sample_obj, dict):
            raise ValueError("AdaLead selection is only supported for single-objective optimization")
        
        # Determine which objectives to use for threshold calculation
        threshold_objectives = last_batch_objectives if last_batch_objectives else objectives
        
        # Find best value and compute threshold
        best_value = max(threshold_objectives.values())
        threshold = (1 - adalead_k) * best_value
        
        # Select ALL sequences from full objectives that meet threshold
        candidates = [seq for seq, val in objectives.items() if val >= threshold]
        
        print(f"AdaLead: Selected {len(candidates)} sequences above threshold (threshold={threshold:.4f}, max={best_value:.4f}, k={adalead_k})")
        return candidates

    def _select_top_predictions(self, predictions: Dict[str, tuple], k: int, acquisition_function: str, acquisition_kwargs: Dict[str, Any] = None) -> List[str]:
        if acquisition_kwargs is None:
            acquisition_kwargs = {}
        acquisition_values = self.acquisition_function_obj.compute_acquisition(
            predictions, 
            acquisition_function, 
            **acquisition_kwargs
        )
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]
