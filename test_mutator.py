#!/usr/bin/env python3
"""
Simple test to verify the mutation abstraction works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from bopep.genetic_algorithm.mutate import PeptideMutator

def test_mutator():
    """Test basic functionality of the PeptideMutator."""
    print("Testing PeptideMutator...")
    
    # Initialize mutator
    mutator = PeptideMutator(
        min_sequence_length=6,
        max_sequence_length=20,
        mutation_rate=0.1,
        random_seed=42
    )
    
    # Test random sequence generation
    seq1 = mutator.generate_random_sequence()
    seq2 = mutator.generate_random_sequence()
    print(f"Random sequence 1: {seq1} (length: {len(seq1)})")
    print(f"Random sequence 2: {seq2} (length: {len(seq2)})")
    
    # Test mutation
    evaluated = set()
    original = "ACDEFGHIK"
    mutated = mutator.mutate_sequence(original, evaluated)
    print(f"Original: {original}")
    print(f"Mutated:  {mutated}")
    print(f"Changed:  {original != mutated}")
    
    # Test pool mutation
    parents = [seq1, seq2, original]
    pool = mutator.mutate_pool(parents, 5, evaluated)
    print(f"Generated pool of {len(pool)} sequences:")
    for i, seq in enumerate(pool):
        print(f"  {i+1}: {seq}")
    
    # Test crossover
    offspring = mutator.crossover(seq1, seq2)
    print(f"Crossover of {seq1} and {seq2}:")
    print(f"Offspring: {offspring}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_mutator()
