import random
from typing import List, Set
import numpy as np

_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


class PeptideMutator:
    def __init__(
        self,
        min_sequence_length: int = 6,
        max_sequence_length: int = 40,
        mutation_rate: float = 0.01,
    ):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.mutation_rate = mutation_rate


    def generate_random_sequence(self) -> str:
        return ''.join(random.choice(_AMINO_ACIDS) for _ in range(random.randint(self.min_sequence_length, self.max_sequence_length)))

    def mutate_sequence(self, seq: str, evaluated_sequences: Set[str]) -> str:
        """
        - n_edits ~ Poisson(len(seq) * mutation_rate), min 1
        - ops: substitution, deletion, insertion
        - respects min/max length at every step
        - guarantees a novel child not seen in evaluated_sequences
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
            and result != seq and result not in evaluated_sequences:
                return result

        return seq

    def mutate_pool(self, parents: List[str], k_pool: int, evaluated_sequences: Set[str]) -> List[str]:
        """
        - m_select: how many top-performing *parents* are selected from the evaluated set.
        - k_pool:   how many *offspring candidates* to generate in this mutation step.
        """
        # Generate a unique pool and avoid sequences we've already evaluated
        pool: set[str] = set()
        attempts = 0
        max_attempts = max(k_pool * 20, 10_000)

        while len(pool) < k_pool and attempts < max_attempts:
            parent = random.choice(parents)
            child = self.mutate_sequence(parent, evaluated_sequences)
            if child not in evaluated_sequences:
                pool.add(child)
            attempts += 1

        return list(pool)
