import random
from typing import List, Set, Dict
import numpy as np

_AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


class Mutator:
    """
    Simple sequence mutation and crossover for genetic algorithms.
    
    Supports:
      - Uniform random substitution mutations
      - Insertions and deletions
      - Single-point and two-point crossover
    """
    def __init__(
        self,
        min_sequence_length: int = 6,
        max_sequence_length: int = 40,
        mutation_rate: float = 0.01,
        p_ins: float = 0.10,
        p_del: float = 0.10,
    ):
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.mutation_rate = mutation_rate
        self.p_ins = float(p_ins)
        self.p_del = float(p_del)

    def generate_random_sequence(self) -> str:
        """Generate a random sequence sequence within length constraints."""
        L = random.randint(self.min_sequence_length, self.max_sequence_length)
        return ''.join(random.choice(_AMINO_ACIDS) for _ in range(L))

    def mutate_sequence(self, seq: str, evaluated_sequences: Set[str], objectives: Dict[str, float] = None) -> str:
        """
        Mutate a sequence using uniform random substitutions, insertions, and deletions.
        """
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
                    # Uniform substitution (excluding current amino acid)
                    choices = [aa for aa in _AMINO_ACIDS if aa != old]
                    child[i] = random.choice(choices)

                elif op == "del":
                    i = np.random.randint(len(child))
                    del child[i]

                else:  # "ins"
                    i = np.random.randint(len(child) + 1)
                    aa = random.choice(_AMINO_ACIDS)
                    child.insert(i, aa)

            result = "".join(child)
            if (self.min_sequence_length <= len(result) <= self.max_sequence_length
                and result != seq and result not in evaluated_sequences):
                return result

        return seq

    def mutate_pool_with_parents(
        self,
        parents: List[str],
        k_pool: int,
        evaluated_sequences: Set[str],
        objectives: Dict[str, float] = None
    ) -> Dict[str, str]:
        """
        Generate a pool of mutated offspring and return child -> parent lineage.
        """
        child_to_parent = {}
        attempts = 0
        max_attempts = max(k_pool * 20, 10_000)

        while len(child_to_parent) < k_pool and attempts < max_attempts:
            parent = random.choice(parents)
            avoid_sequences = set(evaluated_sequences) | set(child_to_parent)
            child = self.mutate_sequence(parent, avoid_sequences, objectives)
            if child != parent and child not in evaluated_sequences and child not in child_to_parent:
                child_to_parent[child] = parent
            attempts += 1

        return child_to_parent

    def mutate_pool(self, parents: List[str], k_pool: int, evaluated_sequences: Set[str], objectives: Dict[str, float] = None) -> List[str]:
        """
        Generate a pool of mutated offspring from parent sequences.
        """
        return list(self.mutate_pool_with_parents(parents, k_pool, evaluated_sequences, objectives))

if __name__ == "__main__":
    mutator = Mutator()
    parent1 = "ACDEFGHIKLMNPQRSTVWY"
    parent2 = "WYVSRTQPNMLKIHGFEDCA"
    evaluated = {parent1, parent2}
    
    print(f"Parent 1: {parent1}")
    print(f"Parent 2: {parent2}\n")
    
    print("=== Mutation Examples ===")
    for i in range(5):
        child = mutator.mutate_sequence(parent1, evaluated)
        evaluated.add(child)
        print(f"Mutant {i+1}: {child}")
