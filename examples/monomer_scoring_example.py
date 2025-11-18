"""
Example of using MonomerScorer for unconditional protein generation scoring.

This demonstrates how to score single-chain proteins without binding targets,
useful for unconditional generation tasks.
"""

from bopep import MonomerScorer

# Initialize the monomer scorer
scorer = MonomerScorer()

print("=== MonomerScorer Example ===\n")
print(f"Available scores: {len(scorer.available_scores)} total")
print(f"  - {', '.join(scorer.confidence_scores[:3])}...")
print()

# Example 1: Score sequence-based properties only
print("Example 1: Sequence-only scoring")
print("-" * 50)
test_sequence = "MKTIIALSYIFCLVFADYKDDDDK"

scores = scorer.score(
    scores_to_include=[
        'molecular_weight',
        'aromaticity',
        'gravy',
        'instability_index',
        'helix_fraction',
    ],
    sequence=test_sequence
)

print(f"Sequence: {test_sequence}")
for seq, score_dict in scores.items():
    print("\nScores:")
    for score_name, score_value in score_dict.items():
        print(f"  {score_name}: {score_value}")

# Example 2: Score with AlphaFold structure prediction output
print("\n\nExample 2: Scoring with AlphaFold output")
print("-" * 50)
print("To score AlphaFold monomer predictions, use:")
print("""
# Assuming you have AlphaFold output in processed_dir/
scores = scorer.score(
    scores_to_include=[
        'plddt',           # Confidence score
        'ptm',             # Predicted TM-score
        'pae',             # Predicted aligned error
        'molecular_weight',
        'dssp_helix_fraction',
        'radius_of_gyration',
        'compactness',
    ],
    processed_dir='path/to/alphafold_output/'
)
""")

# Example 3: Available scores for different contexts
print("\n\nExample 3: Context-aware score availability")
print("-" * 50)

# With sequence only
seq_only_scores = scorer.get_available_scores()
print(f"Scores available with sequence only: {len(seq_only_scores)}")
print(f"  {seq_only_scores[:5]}...")

# With structure file (hypothetical)
# structure_scores = scorer.get_available_scores(
#     structure_file="path/to/protein.pdb"
# )
# print(f"\nScores available with structure: {len(structure_scores)}")

print("\n\n=== Key Differences from ComplexScorer ===")
print("-" * 50)
print("MonomerScorer:")
print("  ✓ Single-chain proteins")
print("  ✓ Intrinsic properties (pLDDT, compactness, etc.)")
print("  ✓ No binding site requirements")
print("  ✓ Uses AlphaFold monomer predictions")
print()
print("ComplexScorer:")
print("  ✓ Protein-peptide complexes")
print("  ✓ Interface metrics (iPTM, dG, binding site)")
print("  ✓ Requires target protein")
print("  ✓ Uses AlphaFold/Boltz docking predictions")
