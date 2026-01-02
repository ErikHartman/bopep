import re


def filter_sequences(
    sequences,
    max_single_aa_fraction: float = 0.75,
    max_repeat_length: int = 5,
    min_length: int = 5,
    max_length: int = 30,
):
    """
    Filters a list of sequences based on several criteria:
    - Length constraints: sequence length must be between min_length and max_length.
    - Fraction of the most common amino acid: must not exceed max_single_aa_fraction.
    - Consecutive repeats: no amino acid should repeat more than max_repeat_length times consecutively.
    - Valid amino acids: sequence must only contain valid amino acids (ACDEFGHIKLMNPQRSTVWY).
    """
    filtered_sequences = []
    for sequence in sequences:
        sequence = sequence.upper()
        length = len(sequence)

        # Length constraints
        if length <= min_length or length >= max_length:
            continue

        # Fraction of the most common amino acid
        aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
        max_aa_fraction = max(aa_counts.values()) / length
        if max_aa_fraction > max_single_aa_fraction:
            continue

        # Consecutive repeats
        if re.search(r"(.)\1{" + str(max_repeat_length) + r",}", sequence):
            continue

        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", sequence):
            continue

        filtered_sequences.append(sequence)
    return filtered_sequences
