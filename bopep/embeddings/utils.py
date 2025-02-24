import re

def filter_peptides(peptides, max_single_aa_fraction=0.75, max_repeat_length=5, min_length=5, max_length=30):
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
        if re.search(r'(.)\1{' + str(max_repeat_length) + r',}', peptide):
            continue
        
        if not re.match('^[ACDEFGHIKLMNPQRSTVWY]+$', peptide):
            continue
        
        filtered_peptides.append(peptide)
    
    return filtered_peptides

