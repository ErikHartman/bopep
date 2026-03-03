"""
Example usage of ProteomeSearch for proteome-based peptide discovery.

This script demonstrates how to search for peptides from a proteome
using surrogate model-guided optimization.
"""

from bopep.search.proteome_search import ProteomeSearch
from pathlib import Path


def example_proteome_search():
    """
    Example of running a proteome search for peptide binders.
    """
    
    # Example proteome (small sample - replace with real proteome)
    proteome = {
        'PROT1': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL',
        'PROT2': 'MRILPISTIKGKLNEFVDAVSSTQDQTIIPAHDILVVSGLKVGDKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA',
        'PROT3': 'MSIIGATRLQNDKSDTYSAGPCYAGGCSAFTPRGTCGKDWDLGEQTCASGFCTSQPLCARIKKTQVCGLRYSSKGKDPLVSAEWDSRGAPYVRCTYDADLIDTQAQVDQFVSMFGESPSLAERYCMRGVKNTAGELVSRVSSDADPAGGWCRKWYSAHRGPDQDAALGSFCIKNPGAADCKCINRASDPVYQKVKTLHAYPDQCWYVPCAADVGELKMGTAAWQAALCAAEQDRAHRAHSSMVWRDGRCAVHVVTRRSTVAEEVAACLRAFQAMRDDSLEPFPKVSVVIADSGVGKSTLLP',
    }
    
    # Target structure path (replace with your target)
    target_structure = './data/target_receptor.pdb'
    
    # Define objective function (example: maximize iptm)
    def objective_fn(scores_dict):
        """Simple objective: maximize iptm."""
        objectives = {}
        for seq, scores in scores_dict.items():
            # Access iptm from the scores
            objectives[seq] = scores.get('iptm', 0.0)
        return objectives
    
    # Initialize ProteomeSearch
    searcher = ProteomeSearch(
        proteome=proteome,
        target_structure_path=target_structure,
        
        # Peptide sampling
        min_peptide_length=8,
        max_peptide_length=15,
        length_distribution='uniform',
        
        # Initial sampling
        n_init=20,
        
        # Selection parameters
        k_propose=1000,  # Sample 1000 peptides from proteome per iteration
        m_select=5,      # Dock top 5 per iteration
        
        # Surrogate model
        surrogate_model_kwargs={
            'model_type': 'deep_evidential',
            'network_type': 'bigru',
            'hpo_interval': 5,
        },
        
        # Embedding
        embed_method='esm',
        pca_n_components=20,
        
        # Scoring
        scoring_kwargs={
            'scores_to_include': ['iptm', 'pae', 'dG'],
            'n_jobs': 4,
        },
        
        # Docker
        docker_kwargs={
            'models': ['boltz'],
            'output_dir': './proteome_search_output',
            'num_models': 1,
        },
        
        # Objective function
        objective_function=objective_fn,
        
        # Logging
        log_dir='./proteome_search_logs',
    )
    
    # Define search schedule
    schedule = [
        {
            'acquisition': 'ucb',
            'generations': 5,
            'acquisition_kwargs': {'kappa': 2.0}
        },
        {
            'acquisition': 'ei',
            'generations': 5,
        },
        {
            'acquisition': 'greedy',
            'generations': 3,
        }
    ]
    
    # Run search
    final_objectives = searcher.run(schedule)
    
    # Print top results
    print("\n=== Top 10 Peptides ===")
    sorted_results = sorted(final_objectives.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (seq, obj) in enumerate(sorted_results, 1):
        print(f"{rank}. {seq}: {obj:.4f}")
    
    return final_objectives


if __name__ == '__main__':
    results = example_proteome_search()
