import random
from typing import List, Set, Dict
import numpy as np
from Bio.Align import substitution_matrices 

_BLOSUM62 = substitution_matrices.load("BLOSUM62")
_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
_AA2IDX = {a: i for i, a in enumerate(_AMINO_ACIDS)}


class PeptideMutator:
    """
    Modular peptide mutation with flexible probability combinations.
    
    Modes:
      'uniform'        -> standard uniform substitution
      'blosum'         -> BLOSUM62 softmax with temperature tau
      'blosum_elite'   -> mix of BLOSUM softmax and elite frequency prior with weight lam
      custom dict      -> custom combination weights, e.g. {'uniform': 0.2, 'blosum': 0.5, 'elite': 0.3}
    """
    def __init__(
        self,
        min_sequence_length: int = 6,
        max_sequence_length: int = 40,
        mutation_rate: float = 0.01,
        mode: str = "uniform",
        tau: float = 1.0,      # higher = flatter, lower = more conservative
        lam: float = 0.3,      # weight on elite prior in blosum_elite mode
        p_ins: float = 0.10,
        p_del: float = 0.10,
    ):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.mutation_rate = mutation_rate

        self.mode = mode
        self.tau = max(1e-6, float(tau))
        self.lam = float(lam)
        self.p_ins = float(p_ins)
        self.p_del = float(p_del)

        # Build a symmetric 20x20 BLOSUM matrix if available
        self._blosum = self._build_blosum_matrix() if _BLOSUM62 is not None else None

        # Global elite prior over residues
        self._elite_prior = np.full(20, 1.0 / 20.0, dtype=float)

    def set_mode(self, mode):
        """Set mutation mode. Can be string ('uniform', 'blosum', 'blosum_elite') or dict with custom weights."""
        if isinstance(mode, str):
            assert mode in {"uniform", "blosum", "blosum_elite"}
        elif isinstance(mode, dict):
            # Validate that all keys are valid probability types
            valid_types = {'uniform', 'blosum', 'elite'}
            assert all(k in valid_types for k in mode.keys()), f"Invalid probability types. Must be in {valid_types}"
        else:
            raise ValueError("Mode must be string or dict")
        self.mode = mode

    def set_elite_prior_from_sequences(self, sequences: List[str], alpha: float = 0.1):
        """Update global elite prior from top sequences. Alpha adds light smoothing."""
        counts = np.zeros(20, dtype=float)
        for s in sequences:
            for ch in s:
                idx = _AA2IDX.get(ch)
                if idx is not None:
                    counts[idx] += 1.0
        if counts.sum() == 0:
            self._elite_prior[:] = 1.0 / 20.0
        else:
            p = counts / counts.sum()
            self._elite_prior = (1 - alpha) * p + alpha * (1.0 / 20.0) # smooth

    def _should_update_elite(self) -> bool:
        """Decide if elite prior should be updated based on current mode."""
        if isinstance(self.mode, str):
            return self.mode == 'blosum_elite'
        elif isinstance(self.mode, dict):
            return 'elite' in self.mode
        return False
    
    def _select_top_sequences(self, objectives: Dict[str, float], n_top: int) -> List[str]:
        """Select top n sequences from objectives."""
        sorted_seqs = sorted(objectives.items(), key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in sorted_seqs[:min(n_top, len(sorted_seqs))]]
    
    def update_elite_prior(self, objectives: Dict[str, float]) -> None:
        """Update elite prior from current objectives if using elite-based modes."""
        if self._should_update_elite() and len(objectives) > 0:
            top_sequences = self._select_top_sequences(objectives, 50)
            self.set_elite_prior_from_sequences(top_sequences)
            print(f"Updated elite prior from top {len(top_sequences)} sequences")

    def generate_random_sequence(self) -> str:
        L = random.randint(self.min_sequence_length, self.max_sequence_length)
        return ''.join(random.choice(_AMINO_ACIDS) for _ in range(L))

    def mutate_sequence(self, seq: str, evaluated_sequences: Set[str], objectives: Dict[str, float] = None) -> str:
        """
        - n_edits ~ Poisson(len(seq) * mutation_rate), min 1
        - ops: substitution, deletion, insertion
        - respects min and max length
        - guarantees a novel child not seen in evaluated_sequences
        - if objectives provided and mode requires elite updates, automatically updates elite prior
        """
        # Auto-update elite prior if needed
        if objectives is not None and self._should_update_elite():
            self.update_elite_prior(objectives)
        max_attempts = 10_000
        ops_space = np.array(["sub", "del", "ins"], dtype=object)

        for _ in range(max_attempts):
            child = list(seq)
            poiss_lam = max(1e-9, len(child) * self.mutation_rate)
            n_edits = max(1, int(np.random.poisson(poiss_lam)))

            for _ in range(n_edits):
                can_del = len(child) > self.min_sequence_length
                can_ins = len(child) < self.max_sequence_length

                w_sub = 1.0
                w_del = self.p_del if can_del else 0.0
                w_ins = self.p_ins if can_ins else 0.0
                probs = np.array([w_sub, w_del, w_ins], dtype=float)
                probs /= probs.sum()

                op = np.random.choice(ops_space, p=probs)

                if op == "sub":
                    i = np.random.randint(len(child))
                    old = child[i]
                    child[i] = self._sample_sub(old)

                elif op == "del":
                    i = np.random.randint(len(child))
                    del child[i]

                else:  # "ins"
                    i = np.random.randint(len(child) + 1)
                    aa = np.random.choice(_AMINO_ACIDS, p=self._elite_prior)  # bias insertions with elite prior
                    child.insert(i, aa)

            result = "".join(child)
            if (self.min_sequence_length <= len(result) <= self.max_sequence_length
                and result != seq and result not in evaluated_sequences):
                return result

        return seq

    def mutate_pool(self, parents: List[str], k_pool: int, evaluated_sequences: Set[str], objectives: Dict[str, float] = None) -> List[str]:
        pool = set()
        attempts = 0
        max_attempts = max(k_pool * 20, 10_000)

        while len(pool) < k_pool and attempts < max_attempts:
            parent = random.choice(parents)
            child = self.mutate_sequence(parent, evaluated_sequences, objectives)
            if child not in evaluated_sequences:
                pool.add(child)
            attempts += 1

        return list(pool)

    def _build_blosum_matrix(self) -> np.ndarray:
        mat = _BLOSUM62
        M = np.zeros((20, 20), dtype=float)

        for i, a in enumerate(_AMINO_ACIDS):
            for j, b in enumerate(_AMINO_ACIDS):
                val = 0.0
                try:
                    val = float(mat[(a, b)])
                except Exception:
                    try:
                        val = float(mat[(b, a)])
                    except Exception:
                        raise ValueError(f"BLOSUM62 missing entry for pair: {a}, {b}")
                M[i, j] = val
        return M

    def _uniform_probs(self, old: str) -> np.ndarray:
        """Return uniform probability vector (excluding old amino acid)."""
        probs = np.ones(20, dtype=float)
        probs[_AA2IDX[old]] = 0.0
        probs /= probs.sum()
        return probs
    
    def _blosum_probs(self, old: str) -> np.ndarray:
        """Return BLOSUM62-based probability vector with temperature scaling."""
        if self._blosum is None:
            return self._uniform_probs(old)
        
        row = self._blosum[_AA2IDX[old]].astype(float).copy()
        row[_AA2IDX[old]] = -np.inf  # forbid identity
        s = row / self.tau
        s -= np.nanmax(s)
        probs = np.exp(s)
        probs[np.isinf(row)] = 0.0
        
        if probs.sum() == 0:
            return self._uniform_probs(old)
        
        probs /= probs.sum()
        return probs
    
    def _elite_probs(self, old: str) -> np.ndarray:
        """Return elite frequency-based probability vector."""
        probs = self._elite_prior.copy()
        probs[_AA2IDX[old]] = 0.0
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            return self._uniform_probs(old)
        return probs
    
    def _combine_probs(self, **prob_weights) -> np.ndarray:
        """
        Combine multiple probability vectors with configurable weights.
        
        Args:
            **prob_weights: keyword arguments where keys are probability types
                          and values are (probability_vector, weight) tuples
                          
        Example:
            _combine_probs(
                uniform=(uniform_probs, 0.2),
                blosum=(blosum_probs, 0.5), 
                elite=(elite_probs, 0.3)
            )
        """
        combined = np.zeros(20, dtype=float)
        total_weight = 0.0
        
        for prob_type, (probs, weight) in prob_weights.items():
            combined += weight * probs
            total_weight += weight
            
        if total_weight > 0:
            combined /= total_weight
        else:
            # Fallback to uniform if no weights
            combined = np.ones(20, dtype=float) / 20.0
            
        return combined


    def _sample_sub(self, old: str) -> str:
        """Sample a substitution using modular probability computation."""
        
        if isinstance(self.mode, dict):
            # Custom mode with specified weights
            return self._sample_sub_custom(old, **self.mode)
        
        elif self.mode == "uniform":
            probs = self._uniform_probs(old)
        elif self.mode == "blosum":
            probs = self._blosum_probs(old)
        elif self.mode == "blosum_elite":
            blosum_probs = self._blosum_probs(old)
            elite_probs = self._elite_probs(old)
            probs = self._combine_probs(
                blosum=(blosum_probs, 1 - self.lam),
                elite=(elite_probs, self.lam)
            )
        else:
            # Fallback to uniform for unknown modes
            probs = self._uniform_probs(old)
        
        return np.random.choice(_AMINO_ACIDS, p=probs)
    
    def _sample_sub_custom(self, old: str, **prob_weights) -> str:
        """
        Sample a substitution using custom probability combination.
        
        Args:
            old: The amino acid being replaced
            **prob_weights: keyword arguments where keys are probability types
                          ('uniform', 'blosum', 'elite') and values are weights
                          
        Example:
            _sample_sub_custom(old, uniform=0.2, blosum=0.5, elite=0.3)
        """
        prob_vectors = {}
        
        if 'uniform' in prob_weights:
            prob_vectors['uniform'] = (self._uniform_probs(old), prob_weights['uniform'])
            
        if 'blosum' in prob_weights:
            prob_vectors['blosum'] = (self._blosum_probs(old), prob_weights['blosum'])
            
        if 'elite' in prob_weights:
            prob_vectors['elite'] = (self._elite_probs(old), prob_weights['elite'])
        
        if not prob_vectors:
            # Fallback to uniform if no valid weights provided
            probs = self._uniform_probs(old)
        else:
            probs = self._combine_probs(**prob_vectors)
        
        return np.random.choice(_AMINO_ACIDS, p=probs)

if __name__ == "__main__":
    mutator = PeptideMutator(mode="uniform", tau=0.5, lam=0.2)
    parent = "ACDEFGHIKLMNPQRSTVWY"
    evaluated = {parent}
    print(f"Parent: {parent}")
    for _ in range(10):
        child = mutator.mutate_sequence(parent, evaluated)
        evaluated.add(child)
        print(f"Child:  {child}")