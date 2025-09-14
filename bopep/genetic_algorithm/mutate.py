import random
from typing import List, Set
import numpy as np
from Bio.Align import substitution_matrices 

_BLOSUM62 = substitution_matrices.load("BLOSUM62")
_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
_AA2IDX = {a: i for i, a in enumerate(_AMINO_ACIDS)}


class PeptideMutator:
    """
    Modes:
      'uniform'        -> standard uniform substitution
      'blosum'         -> BLOSUM62 softmax with temperature tau
      'blosum_elite'   -> mix of BLOSUM softmax and elite frequency prior with weight lam
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
        p_shakeup: float = 0.05,  # chance to ignore mode and use uniform once
    ):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.mutation_rate = mutation_rate

        self.mode = mode
        self.tau = max(1e-6, float(tau))
        self.lam = float(lam)
        self.p_ins = float(p_ins)
        self.p_del = float(p_del)
        self.p_shakeup = float(p_shakeup)

        # Build a symmetric 20x20 BLOSUM matrix if available
        self._blosum = self._build_blosum_matrix() if _BLOSUM62 is not None else None

        # Global elite prior over residues
        self._elite_prior = np.full(20, 1.0 / 20.0, dtype=float)

    def set_mode(self, mode: str):
        assert mode in {"uniform", "blosum", "blosum_elite"}
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
            self._elite_prior = (1 - alpha) * p + alpha * (1.0 / 20.0)

    def generate_random_sequence(self) -> str:
        L = random.randint(self.min_sequence_length, self.max_sequence_length)
        return ''.join(random.choice(_AMINO_ACIDS) for _ in range(L))

    def mutate_sequence(self, seq: str, evaluated_sequences: Set[str]) -> str:
        """
        - n_edits ~ Poisson(len(seq) * mutation_rate), min 1
        - ops: substitution, deletion, insertion
        - respects min and max length
        - guarantees a novel child not seen in evaluated_sequences
        """
        max_attempts = 10_000
        ops_space = np.array(["sub", "del", "ins"], dtype=object)

        for _ in range(max_attempts):
            child = list(seq)
            lam = max(1e-9, len(child) * self.mutation_rate)
            n_edits = max(1, int(np.random.poisson(lam)))

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
                    if len(child) <= self.min_sequence_length:
                        i = np.random.randint(len(child))
                        old = child[i]
                        child[i] = self._sample_sub(old)
                    else:
                        i = np.random.randint(len(child))
                        del child[i]

                else:  # "ins"
                    if len(child) >= self.max_sequence_length:
                        i = np.random.randint(len(child))
                        old = child[i]
                        child[i] = self._sample_sub(old)
                    else:
                        i = np.random.randint(len(child) + 1)
                        # bias insertions with elite prior a bit
                        aa = np.random.choice(_AMINO_ACIDS, p=self._elite_prior)
                        child.insert(i, aa)

            result = "".join(child)
            if (self.min_sequence_length <= len(result) <= self.max_sequence_length
                and result != seq and result not in evaluated_sequences):
                return result

        return seq

    def mutate_pool(self, parents: List[str], k_pool: int, evaluated_sequences: Set[str]) -> List[str]:
        pool = set()
        attempts = 0
        max_attempts = max(k_pool * 20, 10_000)

        while len(pool) < k_pool and attempts < max_attempts:
            parent = random.choice(parents)
            child = self.mutate_sequence(parent, evaluated_sequences)
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


    def _sample_sub(self, old: str) -> str:
        # occasional uniform shakeup
        if random.random() < self.p_shakeup or self.mode == "uniform":
            probs = np.ones(20, dtype=float)
            probs[_AA2IDX[old]] = 0.0
            probs /= probs.sum()
            return np.random.choice(_AMINO_ACIDS, p=probs)

        # BLOSUM softmax over candidates
        row = self._blosum[_AA2IDX[old]].astype(float).copy()
        row[_AA2IDX[old]] = -np.inf  # forbid identity
        s = row / self.tau
        s -= np.nanmax(s)
        bl = np.exp(s)
        bl[np.isinf(row)] = 0.0
        if bl.sum() == 0:
            bl = np.ones(20, dtype=float)
            bl[_AA2IDX[old]] = 0.0
        bl /= bl.sum()

        if self.mode == "blosum":
            probs = bl
        else:
            # blosum_elite
            prior = self._elite_prior.copy()
            prior[_AA2IDX[old]] = 0.0
            prior /= prior.sum() if prior.sum() > 0 else 1.0
            probs = (1 - self.lam) * bl + self.lam * prior
            probs /= probs.sum()

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