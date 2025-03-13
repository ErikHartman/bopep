import re


def filter_peptides(
    peptides,
    max_single_aa_fraction: float = 0.75,
    max_repeat_length: int = 5,
    min_length: int = 5,
    max_length: int = 30,
):
    """
    Filters a list of peptides based on several criteria:
    - Length constraints: peptide length must be between min_length and max_length.
    - Fraction of the most common amino acid: must not exceed max_single_aa_fraction.
    - Consecutive repeats: no amino acid should repeat more than max_repeat_length times consecutively.
    - Valid amino acids: peptide must only contain valid amino acids (ACDEFGHIKLMNPQRSTVWY).

    Args:
        peptides (list of str): List of peptide sequences to filter.
        max_single_aa_fraction (float): Maximum allowed fraction of the most common amino acid in a peptide.
        max_repeat_length (int): Maximum allowed length of consecutive repeats of the same amino acid.
        min_length (int): Minimum length of a peptide.
        max_length (int): Maximum length of a peptide.

    Returns:
        list of str: List of peptides that meet all the filtering criteria.
    """
    filtered_peptides = []
    for peptide in peptides:
        peptide = peptide.upper()
        length = len(peptide)

        # Length constraints
        if length <= min_length or length >= max_length:
            continue

        # Fraction of the most common amino acid
        aa_counts = {aa: peptide.count(aa) for aa in set(peptide)}
        max_aa_fraction = max(aa_counts.values()) / length
        if max_aa_fraction > max_single_aa_fraction:
            continue

        # Consecutive repeats
        if re.search(r"(.)\1{" + str(max_repeat_length) + r",}", peptide):
            continue

        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", peptide):
            continue

        filtered_peptides.append(peptide)

    return filtered_peptides
