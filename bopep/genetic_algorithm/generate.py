import random
from typing import List, Dict, Any, Optional, Callable
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
import torch
_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

class BoGA:
    """
    TODO: 
    - Add validation of binding site indices
    - Abstract out sequence generation logic
    - Add more detailed logging
    - Abstract out things from BoPep and use here.
        - Eg binding site validataion
        - Hparam opt
    

    Genetic algorithm for peptide binder discovery using surrogate modeling.

    In BoGA, we specify a target and binding site. 

    Then we generate N sequences that are potential binders.

    A surrogate model is trained on these sequences to predict binding affinity.

    M sequences are selected based on their predicted binding affinity.

    Then, we mutate these M sequences to generate new sequences.

    This process is repeated for a specified number of generations.
    The goal is to evolve sequences that have high binding affinity to the target.

    Pseudo-code:
        Given Target.

        Generate N_init (e.g 100) peptide sequences using some algorithm.
            - Create new algorithm to generate sequences using some randomized method.

        Init by docking N_init sequences to the target.
            - Use docker in docking module to dock sequences.

        Train a surrogate model on the scores (objective, binding aff.) of these sequences.
            - Use surrogate model and scorer to train on the scores.

        Select M (e.g 50) sequences based on their predicted binding affinity.

        Mutate these M sequences to generate new N (e.g 10,000) sequences. 

        Repeat until convergence or max generations.
    """
    def __init__(
        self,
        target_structure_path: str,
        max_sequence_length: int,
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
    ):
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
            template_pdbs=self.scoring_kwargs.get('template_pdbs')
        )

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
            previous_study=self.previous_study
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
        for _ in range(self.n_pool):
            parent = random.choice(parents)
            pool.append(self._mutate_sequence(parent))
        return pool

    def run(self) -> Dict[str, float]:
        # Initial population and embedding/reduction
        init_seqs = self._generate_initial_sequences()
        init_reduced = self._embed(init_seqs)

        # Dock and score initial
        scores = self._dock_and_score(init_seqs)

        # Convert initial scores to objectives
        objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

        # Initial hyperparameter tuning
        self._optimize_hyperparameters(init_reduced, objectives)

        for gen in range(1, self.generations + 1):
            # Init fresh model
            self._initialize_model(self.best_hyperparams) 
            # Embed and reduce current peptides
            seqs = list(scores.keys())
            reduced_embs = self._embed(seqs)

            # Convert scores to objectives
            objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)

            # Train surrogate or model only based on interval
            if gen % self.hpo_interval == 0:
                self._optimize_hyperparameters(reduced_embs, objectives)

            self.model.fit_dict(reduced_embs, objectives, device=self.device)

            # Generate new pool via mutation of top M (use objectives for selection)
            parents = self._select_top(objectives, self.m_select)
            pool = self._mutate_pool(parents)

            # Embed and reduce pool
            pool_embs = self._embed(pool)

            # Predict and select top
            preds = self.model.predict_dict(pool_embs, device=self.device)
            candidates = self._select_top(preds, self.m_select)

            # Dock, score, and update
            new_scores = self._dock_and_score(candidates)
            scores.update(new_scores)

        # Return final objectives instead of raw scores
        final_objectives = self.scores_to_objective.create_objective(scores, self.objective_function, **self.objective_function_kwargs)
        return final_objectives
