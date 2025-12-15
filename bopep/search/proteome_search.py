import random
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np

from bopep.docking.docker import Docker
from bopep.embedding.embedder import Embedder
from bopep.scoring.complex_scorer import ComplexScorer
from bopep.surrogate_model import SurrogateModelManager
from bopep.scoring.scores_to_objective import ScoresToObjective
from bopep.search.utils import _validate_surrogate_model_kwargs, print_leaderboard
from bopep.bayes.acquisition import AcquisitionFunction
from bopep.logging.logger import Logger
from bopep.config import Config
import torch


class ProteomeSearch:
    """
    Proteome-based search for peptide discovery using surrogate modeling.
    Samples peptides from a provided proteome and uses Bayesian optimization
    to guide the search toward promising candidates.
    """
    
    def __init__(
        self,
        proteome: Dict[str, str],
        target_structure_path: str,
        
        # Peptide sampling parameters
        min_peptide_length: Optional[int] = None,
        max_peptide_length: Optional[int] = None,
        length_distribution: Optional[str] = None,
        
        # Initial sampling
        n_init: Optional[int] = None,
        
        # Surrogate model parameters
        surrogate_model_kwargs: Optional[Dict[str, Any]] = None,
        
        # Objective function
        objective_function: Optional[Callable] = None,
        objective_function_kwargs: Optional[Dict[str, Any]] = None,
        
        # Scoring and docking
        scoring_kwargs: Optional[Dict[str, Any]] = None,
        docker_kwargs: Optional[Dict[str, Any]] = None,
        
        # Embedding options
        embed_method: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        embed_device: Optional[str] = None,
        pca_n_components: Optional[int] = None,
        
        # Validation options
        n_validate: Optional[float] = None,
        min_validation_samples: Optional[int] = None,
        min_training_samples: Optional[int] = None,
        
        # Selection options
        m_select: Optional[int] = None,
        k_propose: Optional[int] = None,
        
        # Logging options
        log_dir: Optional[str] = None,
        
        # Continuation options
        continue_from_logs: Optional[str] = None,
        
        # Config object
        config: Optional[Config] = None,
    ):
        """
        Initialize the ProteomeSearch optimizer.
        
        Args:
            proteome: Dictionary mapping protein IDs to protein sequences
            target_structure_path: Path to target structure for docking
            min_peptide_length: Minimum length of sampled peptides
            max_peptide_length: Maximum length of sampled peptides
            length_distribution: Distribution for sampling peptide lengths ('uniform', 'normal')
            n_init: Number of initial peptides to sample and dock
            surrogate_model_kwargs: Configuration for surrogate model
            objective_function: Custom objective function
            objective_function_kwargs: Parameters for objective function
            scoring_kwargs: Configuration for scoring
            docker_kwargs: Configuration for docking
            embed_method: Embedding method ('esm' or 'aaindex')
            embed_model_path: Path to embedding model
            embed_batch_size: Batch size for embedding
            embed_device: Device for embedding ('cuda' or 'cpu')
            pca_n_components: Number of PCA components
            n_validate: Fraction or number of samples for validation
            min_validation_samples: Minimum samples for validation
            min_training_samples: Minimum samples for training
            m_select: Number of candidates to select and dock per iteration
            k_propose: Number of candidates to sample from proteome per iteration
            log_dir: Directory for logging
            continue_from_logs: Path to previous log directory to resume from
            config: Optional Config object. If not provided, defaults will be loaded.
        """
        # Initialize or load config
        if config is None:
            config = Config(script="ProteomeSearch")
        self.config = config
        
        # Get flattened config
        cfg = self.config.flatten()
        
        # Helper function to get parameter value (user override > config)
        def get_param(user_val, config_key):
            return user_val if user_val is not None else cfg.get(config_key)
        
        # Store proteome
        if not proteome:
            raise ValueError("proteome cannot be empty. Provide a dictionary of protein_id: sequence")
        
        self.proteome = proteome
        self.protein_ids = list(proteome.keys())
        print(f"Loaded proteome with {len(self.proteome)} proteins")
        
        # Target structure
        self.target_structure_path = target_structure_path
        
        # Peptide sampling parameters
        self.min_peptide_length = get_param(min_peptide_length, 'min_peptide_length')
        self.max_peptide_length = get_param(max_peptide_length, 'max_peptide_length')
        self.length_distribution = get_param(length_distribution, 'length_distribution')
        
        # Initial sampling
        self.n_init = get_param(n_init, 'n_init')
        
        # Get surrogate model kwargs
        if surrogate_model_kwargs is not None:
            self.surrogate_model_kwargs = surrogate_model_kwargs
        else:
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
            self.scoring_kwargs = {
                k.replace('scoring.', ''): v 
                for k, v in cfg.items() 
                if k.startswith('scoring.')
            }
        
        # Objective function
        self.objective_function = objective_function
        self.objective_function_kwargs = objective_function_kwargs or {}
        
        # Get validation parameters
        self.n_validate = get_param(n_validate, 'validation.n_validate')
        self.min_validation_samples = get_param(min_validation_samples, 'validation.min_validation_samples')
        self.min_training_samples = get_param(min_training_samples, 'validation.min_training_samples')
        
        # Get selection parameters
        self.m_select = get_param(m_select, 'selection.m_select')
        self.k_propose = get_param(k_propose, 'selection.k_propose')
        
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
        
        # Enforce fixed PCA dimensions
        if self.pca_n_components is None:
            raise ValueError(
                "pca_n_components must be specified to ensure dimensional consistency. "
            )
        
        # Get docker kwargs
        if docker_kwargs is not None:
            final_docker_kwargs = docker_kwargs
        else:
            final_docker_kwargs = {
                k.replace('docker.', ''): v 
                for k, v in cfg.items() 
                if k.startswith('docker.')
            }
        
        # Initialize components
        self.docker = Docker(final_docker_kwargs)
        self.docker.set_target_structure(self.target_structure_path)
        self.scorer = ComplexScorer()
        self.scores_to_objective = ScoresToObjective()
        self.embedder = Embedder()
        self.acquisition_function_obj = AcquisitionFunction()
        
        # Set max_seq_len based on max_peptide_length if not already specified
        if 'max_seq_len' not in self.surrogate_model_kwargs:
            self.surrogate_model_kwargs['max_seq_len'] = self.max_peptide_length
        
        # Initialize surrogate model manager
        self.surrogate_manager = SurrogateModelManager(
            surrogate_model_kwargs=self.surrogate_model_kwargs,
            device=self.device
        )
        
        # Logging and continuation
        self.continue_from_logs = get_param(continue_from_logs, 'logging.continue_from_logs')
        
        if continue_from_logs is not None:
            # When continuing from logs, never overwrite existing logs
            self.logger = Logger(log_dir=continue_from_logs, overwrite_logs=False)
        elif log_dir is not None:
            # For fresh runs, overwrite logs
            self.logger = Logger(log_dir=log_dir, overwrite_logs=True)
        else:
            self.logger = None
        
        # Track evaluated sequences
        self._evaluated_sequences = set()
        
        # Track peptide metadata (source protein and position)
        self._peptide_metadata = {}  # {peptide_sequence: {"protein_id": str, "start_pos": int}}
    
    def _sample_peptide_length(self) -> int:
        """Sample a peptide length from the configured distribution."""
        if self.length_distribution == 'uniform':
            return random.randint(self.min_peptide_length, self.max_peptide_length)
        elif self.length_distribution == 'normal':
            # Use normal distribution centered at midpoint
            mean = (self.min_peptide_length + self.max_peptide_length) / 2
            std = (self.max_peptide_length - self.min_peptide_length) / 6  # ~99.7% within range
            length = int(np.random.normal(mean, std))
            # Clip to valid range
            return max(self.min_peptide_length, min(self.max_peptide_length, length))
        else:
            raise ValueError(f"Unknown length_distribution: {self.length_distribution}")
    
    def _sample_peptide_from_proteome(self) -> Tuple[str, str, int]:
        """
        Sample a single peptide from the proteome.
        """
        # Pick random protein
        protein_id = random.choice(self.protein_ids)
        protein_seq = self.proteome[protein_id]
        
        # Sample peptide length
        peptide_length = self._sample_peptide_length()
        
        # Check if protein is long enough
        if len(protein_seq) < peptide_length:
            # If protein too short, just use the whole protein
            return protein_seq, protein_id, 0
        
        # Sample random midpoint
        # Midpoint should be positioned such that peptide fits within protein
        half_length = peptide_length // 2
        min_midpoint = half_length
        max_midpoint = len(protein_seq) - (peptide_length - half_length)
        
        if min_midpoint > max_midpoint:
            # Edge case: very short protein
            return protein_seq, protein_id, 0
        
        midpoint = random.randint(min_midpoint, max_midpoint)
        
        # Extract peptide around midpoint
        start = midpoint - half_length
        end = start + peptide_length
        peptide = protein_seq[start:end]
        
        return peptide, protein_id, start
    
    def _sample_peptides_from_proteome(self, n_sample: int) -> List[str]:
        """
        Sample n unique peptides from the proteome.
        """
        peptides = set()
        max_attempts = n_sample * 10  # Prevent infinite loop
        attempts = 0
        
        while len(peptides) < n_sample and attempts < max_attempts:
            peptide, protein_id, start_pos = self._sample_peptide_from_proteome()
            if peptide not in self._evaluated_sequences:
                peptides.add(peptide)
                # Store metadata for this peptide
                self._peptide_metadata[peptide] = {
                    "protein_id": protein_id,
                    "start_pos": start_pos
                }
            attempts += 1
        
        if len(peptides) < n_sample:
            print(f"Warning: Only sampled {len(peptides)} unique peptides (requested {n_sample})")
        
        return list(peptides)
    
    def _embed_sequences(self, sequences: List[str]) -> Dict[str, Any]:
        """Embed, scale, and apply PCA to sequences."""
        if not sequences:
            return {}
        
        # Embed sequences
        if self.embed_method == 'esm':
            raw_embeddings = self.embedder.embed_esm(
                sequences,
                average=self.embed_average,
                model_path=self.embed_model_path,
                batch_size=self.embed_batch_size,
                filter=False,
                device=self.embed_device
            )
        elif self.embed_method == 'aaindex':
            raw_embeddings = self.embedder.embed_aaindex(
                sequences,
                average=self.embed_average,
                filter=False
            )
        else:
            raise ValueError("embed_method must be 'esm' or 'aaindex'")
        
        # Scale and reduce
        scaled_embeddings = self.embedder.scale_embeddings(raw_embeddings)
        reduced_embeddings = self.embedder.reduce_embeddings_pca(
            scaled_embeddings,
            n_components=self.pca_n_components
        )
        
        return reduced_embeddings
    
    def _embed_generation(self, scored_sequences: List[str], candidate_sequences: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Embed and reduce sequences for this generation, ensuring consistent scaling/PCA.
        """
        # Combine all sequences for consistent scaling and PCA
        all_sequences = scored_sequences + candidate_sequences
        all_embeddings = self._embed_sequences(all_sequences)
        
        # Split back into training and candidate sets
        training_embeddings = {p: all_embeddings[p] for p in scored_sequences if p in all_embeddings}
        candidate_embeddings = {p: all_embeddings[p] for p in candidate_sequences if p in all_embeddings}
        
        return training_embeddings, candidate_embeddings
    
    def _dock_and_score(self, sequences: List[str]) -> Dict[str, Dict[str, float]]:
        """Dock sequences and compute scores."""
        dock_dirs = self.docker.dock_sequences(sequences)
        scores = self.scorer.score_batch(
            scores_to_include=self.scoring_kwargs.get('scores_to_include', []),
            inputs=dock_dirs,
            input_type='processed_dir',
            binding_site_residue_indices=self.scoring_kwargs.get('binding_site_residue_indices'),
            n_jobs=self.scoring_kwargs.get('n_jobs', 12),
            binding_site_distance_threshold=self.scoring_kwargs.get('binding_site_distance_threshold', 5),
            required_n_contact_residues=self.scoring_kwargs.get('required_n_contact_residues', 5),
        )
        
        # Add metadata to scores
        for seq in scores:
            if seq in self._peptide_metadata:
                scores[seq]['protein_id'] = self._peptide_metadata[seq]['protein_id']
                scores[seq]['start_pos'] = self._peptide_metadata[seq]['start_pos']
        
        self._evaluated_sequences.update(sequences)
        return scores
    
    def _optimize_hyperparameters(self, embeddings: Dict[str, Any], objectives: Dict[str, float], iteration: Optional[int] = None) -> None:
        """Hyperparameter tuning using the surrogate model manager."""
        self.surrogate_manager.optimize_hyperparameters(
            embeddings=embeddings,
            objectives=objectives,
            n_trials=self.surrogate_model_kwargs.get('n_trials', 20),
            n_splits=self.surrogate_model_kwargs.get('n_splits', 3),
            iteration=iteration
        )
    
    def _train_model(self, embeddings: Dict[str, Any], objectives: Dict[str, float]) -> Dict[str, Any]:
        """Train the model with automatic validation split."""
        return self.surrogate_manager.train_with_validation_split(
            embeddings=embeddings,
            objectives=objectives,
            validation_size=self.n_validate,
            min_training_samples=self.min_training_samples,
            min_validation_samples=self.min_validation_samples
        )
    
    def _select_top_predictions(self, predictions: Dict[str, tuple], k: int, acquisition_function: str, acquisition_kwargs: Dict[str, Any] = None) -> List[str]:
        """Select top k candidates using acquisition function."""
        if acquisition_kwargs is None:
            acquisition_kwargs = {}
        acquisition_values = self.acquisition_function_obj.compute_acquisition(
            predictions, 
            acquisition_function, 
            **acquisition_kwargs
        )
        return [seq for seq, _ in sorted(acquisition_values.items(), key=lambda x: x[1], reverse=True)[:k]]
    
    def run(self, schedule: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Run the proteome search optimization.
        """
        print("=== Starting Proteome Search ===")
        print(f"Proteome size: {len(self.proteome)} proteins")
        print(f"Peptide length range: {self.min_peptide_length}-{self.max_peptide_length}")
        print(f"k_propose: {self.k_propose}, m_select: {self.m_select}")
        
        # Get objective directions for multi-objective
        objective_directions = self.scoring_kwargs.get("objective_directions", {})
        
        if self.continue_from_logs:
            print(f"\n--- Loading previous results from {self.continue_from_logs} ---")
            scores, self._evaluated_sequences, last_iteration = self._load_from_logs(self.continue_from_logs)
            print("Skipping initial population generation - using loaded sequences")
            
            # Convert scores to objectives
            objectives = self.scores_to_objective.create_objective(
                scores, 
                self.objective_function, 
                **self.objective_function_kwargs
            )
            
            print("Loaded objectives:")
            print(f"Total sequences: {len(objectives)}")
            print("Best existing performers:")
            self._print_leaderboard(objectives, last_iteration, objective_directions=objective_directions)
            
            # Load previous hyperparameters if available
            previous_hyperparams = self._load_hyperparameters_from_logs(self.continue_from_logs)
            if previous_hyperparams:
                print("Using previous hyperparameters to initialize surrogate model")
                self.surrogate_manager.best_hyperparams = previous_hyperparams
            else:
                print("No previous hyperparameters found, will optimize on first HPO interval")
            
        else:
            # Fresh start - sample and evaluate initial peptides
            print(f"\n--- Initialization: sampling {self.n_init} peptides ---")
            initial_peptides = self._sample_peptides_from_proteome(self.n_init)
            print(f"Sampled {len(initial_peptides)} initial peptides from proteome")
            
            print("Docking and scoring initial peptides...")
            scores = self._dock_and_score(initial_peptides)
            
            # Convert to objectives
            objectives = self.scores_to_objective.create_objective(
                scores, 
                self.objective_function, 
                **self.objective_function_kwargs
            )
            
            print(f"Initial evaluation complete. {len(objectives)} sequences scored.")
            
            # Embed initial sequences
            initial_embeddings = self._embed_sequences(list(scores.keys()))
            
            # Optimize hyperparameters on initial data
            print("Optimizing hyperparameters on initial data...")
            self._optimize_hyperparameters(initial_embeddings, objectives, iteration=0)
            
            if self.logger:
                self.logger.log_scores(scores, iteration=0, acquisition_name='initial')
                self.logger.log_objectives(objectives, iteration=0, acquisition_name='initial')
                if self.surrogate_manager.best_hyperparams:
                    self.logger.log_hyperparameters(
                        iteration=0,
                        hyperparams=self.surrogate_manager.best_hyperparams,
                        model_type=self.surrogate_model_kwargs['model_type'],
                        network_type=self.surrogate_model_kwargs['network_type']
                    )
            
            self._print_leaderboard(objectives, 0, objective_directions=objective_directions)
            last_iteration = 0
        
        # Run through schedule phases
        global_iteration = last_iteration  # Continue from last iteration when resuming
        
        for phase_index, phase in enumerate(schedule, start=1):
            acquisition_function = phase['acquisition']
            iterations = phase['generations']
            
            print(f"\n=== Phase {phase_index}: {acquisition_function} for {iterations} iterations ===")
            print(f"Selection: {self.m_select}, Pool: {self.k_propose}")
            
            for iter_num in range(1, iterations + 1):
                global_iteration += 1
                print(f"\n--- Iteration {global_iteration} (Phase {phase_index}, Iter {iter_num}/{iterations}) ---")
                
                # Sample k_propose peptides from proteome
                print(f"Sampling {self.k_propose} peptides from proteome...")
                candidate_pool = self._sample_peptides_from_proteome(self.k_propose)
                print(f"Sampled {len(candidate_pool)} candidate peptides")
                
                # Embed scored sequences + candidates together
                scored_seqs = list(scores.keys())
                training_embeddings, candidate_embeddings = self._embed_generation(scored_seqs, candidate_pool)
                
                # Initialize model with training embeddings
                self.surrogate_manager.initialize_model(embeddings=training_embeddings, objectives=objectives)
                
                # HPO if needed
                if global_iteration % self.surrogate_model_kwargs['hpo_interval'] == 0:
                    print(f"Re-optimizing hyperparameters (iteration {global_iteration})")
                    self._optimize_hyperparameters(training_embeddings, objectives, iteration=global_iteration)
                
                # Train surrogate model
                print("Training surrogate model...")
                metrics = self._train_model(training_embeddings, objectives)
                loss = metrics["val_mse"] if metrics["val_mse"] is not None else metrics["train_mse"]
                
                # Predict on candidate pool
                preds = self.surrogate_manager.predict(candidate_embeddings)
                
                # Select top candidates using acquisition function
                acquisition_kwargs = phase.get("acquisition_kwargs", {})
                candidates = self._select_top_predictions(
                    preds, 
                    self.m_select, 
                    acquisition_function, 
                    acquisition_kwargs
                )
                print(f"Selected {len(candidates)} candidates for evaluation using {acquisition_function}")
                
                # Dock and score selected candidates
                new_scores = self._dock_and_score(candidates)
                scores.update(new_scores)
                
                # Update objectives
                new_objectives = self.scores_to_objective.create_objective(
                    new_scores, 
                    self.objective_function, 
                    **self.objective_function_kwargs
                )
                objectives.update(new_objectives)
                
                # Logging
                if self.logger:
                    self.logger.log_model_metrics(loss, iteration=global_iteration, metrics=metrics)
                    self.logger.log_scores(new_scores, iteration=global_iteration, acquisition_name=acquisition_function)
                    self.logger.log_objectives(new_objectives, iteration=global_iteration, acquisition_name=acquisition_function)
                    
                    if global_iteration % self.surrogate_model_kwargs['hpo_interval'] == 0 and self.surrogate_manager.best_hyperparams:
                        self.logger.log_hyperparameters(
                            iteration=global_iteration,
                            hyperparams=self.surrogate_manager.best_hyperparams,
                            model_type=self.surrogate_model_kwargs['model_type'],
                            network_type=self.surrogate_model_kwargs['network_type']
                        )
                
                # Print progress
                self._print_leaderboard(objectives, global_iteration, objective_directions=objective_directions)
        
        # Final results
        print(f"\n=== Final Results ===")
        print(f"Total sequences evaluated: {len(objectives)}")
        print("Final leaderboard:")
        self._print_leaderboard(objectives, global_iteration, objective_directions=objective_directions)
        
        return objectives
    
    def _print_leaderboard(self, objectives: Dict[str, Any], iteration: int, print_n: int = 5, objective_directions: Dict[str, str] = None):    
        """Print leaderboard for both single and multi-objective cases."""
        print_leaderboard(
            objectives=objectives,
            iteration=iteration,
            print_n=print_n,
            objective_directions=objective_directions,
            iteration_label="Iteration",
            use_logging=False
        )
    
    def _load_from_logs(self, log_dir: str) -> Tuple[Dict[str, Dict[str, float]], set, int]:
        """
        Load scores and evaluated sequences from existing log files.
        Returns scores, evaluated_sequences, and last_iteration.
        """
        import pandas as pd
        from pathlib import Path
        
        log_path = Path(log_dir)
        scores_file = log_path / "scores.csv"
        
        if not scores_file.exists():
            raise FileNotFoundError(f"No scores.csv found in {log_dir}")
        
        print(f"Loading scores from {scores_file}")
        df = pd.read_csv(scores_file)
        
        scores = {}
        for _, row in df.iterrows():
            sequence = row['sequence']
            score_columns = [col for col in df.columns 
                           if col not in ['sequence', 'iteration', 'phase', 'timestamp']]
            scores[sequence] = {col: row[col] for col in score_columns}
            
            # Load peptide metadata if available
            if 'protein_id' in row and pd.notna(row['protein_id']):
                self._peptide_metadata[sequence] = {
                    'protein_id': row['protein_id'],
                    'start_pos': int(row['start_pos']) if 'start_pos' in row and pd.notna(row['start_pos']) else 0
                }

        evaluated_sequences = set(scores.keys())
        
        # Get the last iteration number to continue from
        last_iteration = df['iteration'].max() if 'iteration' in df.columns and not df.empty else 0
        
        print(f"Loaded {len(scores)} previously evaluated sequences")
        print(f"Last iteration was: {last_iteration}")
        
        return scores, evaluated_sequences, last_iteration
    
    def _load_hyperparameters_from_logs(self, log_dir: str) -> Optional[Dict[str, Any]]:
        """
        Load the most recent hyperparameters from existing log files.
        Returns hyperparameters dict or None if not found.
        """
        import pandas as pd
        from pathlib import Path
        
        log_path = Path(log_dir)
        hyper_file = log_path / "hyperparameters.csv"
        
        if not hyper_file.exists():
            print(f"No hyperparameters.csv found in {log_dir}, will optimize from scratch")
            return None
        
        try:
            df = pd.read_csv(hyper_file)
            if df.empty:
                return None
            
            # Get the most recent hyperparameters (last row)
            last_row = df.iloc[-1]
            
            # Extract hyperparameters (skip metadata columns)
            hyperparams = {}
            skip_cols = ['timestamp', 'iteration', 'model_type', 'network_type']
            for col in df.columns:
                if col not in skip_cols:
                    value = last_row[col]
                    # Convert back to appropriate types
                    if pd.notna(value):
                        # Try to convert to float/int if possible
                        try:
                            if '.' in str(value):
                                hyperparams[col] = float(value)
                            else:
                                hyperparams[col] = int(value)
                        except (ValueError, TypeError):
                            hyperparams[col] = value
            
            print(f"Loaded hyperparameters from iteration {last_row['iteration']}")
            print(f"Hyperparameters: {hyperparams}")
            return hyperparams
            
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            return None