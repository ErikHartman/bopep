#!/usr/bin/env python3

from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).parent.parent))

from bopep.genetic_algorithm.mutate import Mutator
from bopep.embedding.embedder import Embedder
import umap


def hamming_distance(seq1: str, seq2: str) -> int:
    """Calculate Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        # For different lengths, align and count differences
        max_len = max(len(seq1), len(seq2))
        seq1_padded = seq1.ljust(max_len, 'X')  # Pad with 'X'
        seq2_padded = seq2.ljust(max_len, 'X')
        return sum(c1 != c2 for c1, c2 in zip(seq1_padded, seq2_padded))
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


class MutationPathExplorer:
    """Explore how different mutation modes navigate from start to target sequence."""
    
    def __init__(self, start_sequence: str, target_sequence: str):
        self.start_sequence = start_sequence
        self.target_sequence = target_sequence
        self.sequences_data = []  # Will store (sequence, phase, iteration, parent_seq, objective_score)
        self.evaluated_sequences = set()
        self.sequence_to_data = {}  # Map sequence -> (phase, iteration, parent, score)
        
        # Initialize embedder
        self.embedder = Embedder()
        
        print(f"Start sequence:  {start_sequence}")
        print(f"Target sequence: {target_sequence}")
        print(f"Initial distance: {hamming_distance(start_sequence, target_sequence)}")
        
                # Define mutation phases with parameters - simplified to 2 most contrasting approaches
        self.mutation_phases = [
            {
                'name': 'uniform_random',
                'mode': 'uniform',
                'tau': 1.0, # not used in uniform mode
                'lam': 0.3, # not used in uniform mode
                'max_iterations':1000,  # Maximum iterations before giving up
                'n_mutations_per_iter': 20,
                'selection_strategy': 'top_10',  # Random parent selection
            },
            {
                'name': 'elite_conservative',
                'mode': 'blosum_elite',
                'tau': 0.5, # more conservative BLOSUM
                'lam': 0.5, # stronger weight on elite prior
                'max_iterations': 1000,
                'n_mutations_per_iter': 20,
                'selection_strategy': 'top_10',  # Select from top 10%
            }
        ]
        
        # Track results for each phase
        self.phase_results = {}
        
    def objective_function(self, sequence: str) -> float:
        """
        Objective function: negative Hamming distance to target.
        Higher score = closer to target.
        """
        distance = hamming_distance(sequence, self.target_sequence)
        max_possible_distance = max(len(self.start_sequence), len(self.target_sequence))
        # Normalize to [0, 1] where 1 = perfect match
        return 1.0 - (distance / max_possible_distance)
    
    def select_parents(self, available_sequences: List[str], strategy: str, n_select: int) -> List[str]:
        """Select parent sequences based on strategy and objective scores."""
        if strategy == 'random':
            return random.choices(available_sequences, k=n_select)
        
        # Score all available sequences
        scored_sequences = [(seq, self.objective_function(seq)) for seq in available_sequences]
        scored_sequences.sort(key=lambda x: x[1], reverse=True)  # Best first
        
        if strategy == 'top_50':
            top_fraction = max(1, len(scored_sequences) // 2)
        elif strategy == 'top_25':
            top_fraction = max(1, len(scored_sequences) // 4)
        elif strategy == 'top_10':
            top_fraction = max(1, len(scored_sequences) // 10)
        else:
            top_fraction = len(scored_sequences)
        
        # Select from top fraction
        top_sequences = [seq for seq, _ in scored_sequences[:top_fraction]]
        return random.choices(top_sequences, k=n_select)
    
    def run_single_phase(self, phase: dict, mutator: 'Mutator') -> dict:
        """Run a single mutation phase until convergence or max iterations."""
        print(f"\n=== Phase: {phase['name']} ===")
        print(f"Mode: {phase['mode']}, Selection: {phase['selection_strategy']}")
        
        # Configure mutator for this phase
        mutator.set_mode(phase['mode'])
        mutator.tau = phase['tau']
        mutator.lam = phase['lam']
        
        # Phase-specific tracking
        phase_sequences = []
        phase_evaluated = set()
        
        # Start with only the start sequence for this phase
        current_sequences = [self.start_sequence]
        
        # Add start sequence to phase data
        start_score = self.objective_function(self.start_sequence)
        phase_sequences.append((self.start_sequence, phase['name'], 0, None, start_score))
        phase_evaluated.add(self.start_sequence)
        
        # Track convergence
        target_found = False
        best_score_this_phase = start_score
        iterations_to_convergence = None
        
        for iteration in range(phase['max_iterations']):
            new_sequences = []
            
            # Update elite prior for elite modes
            if phase['mode'] == 'blosum_elite' and len(current_sequences) > 1:
                good_sequences = [seq for seq in current_sequences 
                                if self.objective_function(seq) > start_score]
                if good_sequences:
                    mutator.set_elite_prior_from_sequences(good_sequences)
            
            # Select parents using strategy
            n_parents = min(phase['n_mutations_per_iter'], len(current_sequences))
            parents = self.select_parents(current_sequences, phase['selection_strategy'], n_parents)
            
            # Generate mutations
            attempts = 0
            max_attempts = phase['n_mutations_per_iter'] * 20
            
            while len(new_sequences) < phase['n_mutations_per_iter'] and attempts < max_attempts:
                parent = random.choice(parents)
                
                # Generate mutation
                mutated = mutator.mutate_sequence(parent, phase_evaluated)
                
                if mutated not in phase_evaluated:
                    score = self.objective_function(mutated)
                    new_sequences.append(mutated)
                    phase_evaluated.add(mutated)
                    phase_sequences.append((mutated, phase['name'], iteration + 1, parent, score))
                    
                    # Check for convergence (found target)
                    if mutated == self.target_sequence:
                        target_found = True
                        iterations_to_convergence = iteration + 1
                        print(f"  🎯 TARGET FOUND at iteration {iteration + 1}!")
                        break
                    
                    # Track best score
                    if score > best_score_this_phase:
                        best_score_this_phase = score
                
                attempts += 1
            
            # Add new sequences to current pool
            current_sequences.extend(new_sequences)
            
            # If target found, stop this phase
            if target_found:
                break
            
            # Optional: limit pool size to prevent explosion
            if len(current_sequences) > 150:
                scored = [(seq, self.objective_function(seq)) for seq in current_sequences]
                scored.sort(key=lambda x: x[1], reverse=True)
                current_sequences = [seq for seq, _ in scored[:120]] + [seq for seq, _ in scored[120:]][:30]
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: Generated {len(new_sequences)} new sequences, "
                      f"Best score: {best_score_this_phase:.3f}")
        
        # Return results for this phase
        return {
            'phase_data': phase_sequences,
            'target_found': target_found,
            'iterations_to_convergence': iterations_to_convergence,
            'total_iterations': iteration + 1,
            'best_score': best_score_this_phase,
            'total_sequences': len(phase_sequences)
        }

    def run_exploration(self):
        """Run mutation exploration through all phases independently until convergence."""
        print(f"\nStarting path exploration...")
        
        # Add target sequence for reference (shared across all phases)
        target_score = self.objective_function(self.target_sequence)
        self.sequences_data.append((self.target_sequence, 'target', 0, None, target_score))
        
        # Initialize mutator
        mutator = Mutator(
            min_sequence_length=max(6, min(len(self.start_sequence), len(self.target_sequence)) - 3),
            max_sequence_length=max(len(self.start_sequence), len(self.target_sequence)) + 3,
            mutation_rate=0.08,
        )
        
        # Run each phase independently until convergence
        for phase in self.mutation_phases:
            result = self.run_single_phase(phase, mutator)
            self.phase_results[phase['name']] = result
            
            # Add this phase's data to the main data structure
            self.sequences_data.extend(result['phase_data'])
            
            # Print convergence results
            if result['target_found']:
                print(f"✅ {phase['name']} converged in {result['iterations_to_convergence']} iterations")
            else:
                print(f"❌ {phase['name']} did not find target in {result['total_iterations']} iterations")
                print(f"   Best score: {result['best_score']:.3f}")
        
        print(f"\nExploration complete! Generated {len(self.sequences_data)} total sequences")
        
        # Print convergence summary
        print("\n=== Convergence Summary ===")
        for phase_name, result in self.phase_results.items():
            if result['target_found']:
                print(f"{phase_name}: ✅ Found target in {result['iterations_to_convergence']} iterations "
                      f"({result['total_sequences']} sequences)")
            else:
                best_distance = min([hamming_distance(seq, self.target_sequence) 
                                   for seq, _, _, _, _ in result['phase_data']])
                print(f"{phase_name}: ❌ Best distance: {best_distance} "
                      f"({result['total_sequences']} sequences)")
        
    def embed_sequences(self):
        """Embed all sequences using ESM."""
        print("\nEmbedding sequences using ESM...")
        
        sequences = [seq for seq, _, _, _, _ in self.sequences_data]
        
        # Embed using ESM with averaging
        embeddings = self.embedder.embed_esm(
            sequences,
            average=True,
            filter=False,
            batch_size=64
        )
        
        # Convert to numpy array for UMAP
        embedding_matrix = np.array([embeddings[seq] for seq in sequences])
        
        print(f"Embedded {len(sequences)} sequences with shape {embedding_matrix.shape}")
        return embedding_matrix
    
    def create_path_visualization(self, embeddings: np.ndarray, save_path: str = "mutation_paths_umap.png"):
        """Create UMAP visualization with separate subplot for each mutation mode."""
        print("Creating UMAP path visualization...")
        
        # Perform UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )
        
        umap_embedding = reducer.fit_transform(embeddings)
        
        # Prepare data for plotting
        df = pd.DataFrame({
            'sequence': [seq for seq, _, _, _, _ in self.sequences_data],
            'phase': [phase for _, phase, _, _, _ in self.sequences_data],
            'iteration': [iter for _, _, iter, _, _ in self.sequences_data],
            'parent': [parent for _, _, _, parent, _ in self.sequences_data],
            'score': [score for _, _, _, _, score in self.sequences_data],
            'umap_x': umap_embedding[:, 0],
            'umap_y': umap_embedding[:, 1]
        })
        
                # Get target and start positions
        target_data = df[df['phase'] == 'target'].iloc[0]
        target_pos = (target_data['umap_x'], target_data['umap_y'])
        
        # Calculate consistent axis limits for all subplots
        all_x = df['umap_x']
        all_y = df['umap_y']
        x_margin = (all_x.max() - all_x.min()) * 0.1
        y_margin = (all_y.max() - all_y.min()) * 0.1
        xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
        ylim = (all_y.min() - y_margin, all_y.max() + y_margin)
        
        # Prepare background data (all sequences except target/reference)
        background_data = df[~df['phase'].isin(['target'])]
        
        # Create 1x2 subplot for the 2 mutation modes
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, phase in enumerate(self.mutation_phases):
            ax = axes[idx]
            phase_name = phase['name']
            
            # Set consistent axis limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # Plot background points (all other sequences) very faintly
            other_phases_data = background_data[background_data['phase'] != phase_name]
            if len(other_phases_data) > 0:
                ax.scatter(other_phases_data['umap_x'], other_phases_data['umap_y'], 
                          c='lightgray', alpha=0.15, s=8, zorder=1)
            
            # Get data for this phase only
            phase_data = df[df['phase'] == phase_name]
            
            if len(phase_data) == 0:
                ax.text(0.5, 0.5, f'No data for {phase_name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(phase_name)
                continue
            
            # Color points by iteration (temporal progression)
            scatter = ax.scatter(phase_data['umap_x'], phase_data['umap_y'], 
                               c=phase_data['iteration'], cmap='coolwarm', 
                               alpha=0.8, s=50, edgecolors='white', linewidth=0.5, zorder=3, vmin=0, vmax=500)
            
            # Add start point (find it in phase data or use target reference)
            start_point = phase_data[phase_data['iteration'] == 0]
            if len(start_point) > 0:
                ax.scatter(start_point['umap_x'], start_point['umap_y'], 
                          c='red', marker='o', s=100, label='Start', 
                          zorder=5)
            
            # Add target point
            ax.scatter(target_pos[0], target_pos[1], 
                      c='gold', marker='o', s=100, label='Target', 
                      zorder=5)
            
            # Check if this phase found the target
            convergence_info = self.phase_results.get(phase_name, {})
            
            # Build title with mutation parameters and convergence info
            title_parts = [phase_name.replace('_', ' ').title()]
            
            # Add mutation parameters based on mode
            if phase['mode'] == 'uniform':
                title_parts.append("(Uniform)")
            elif phase['mode'] == 'blosum':
                title_parts.append(f"(BLOSUM, τ={phase['tau']:.1f})")
            elif phase['mode'] == 'blosum_elite':
                title_parts.append(f"(BLOSUM+Elite, τ={phase['tau']:.1f}, λ={phase['lam']:.1f})")
            
            # Add convergence information
            if convergence_info.get('target_found', False):
                title_parts.append(f"✅ {convergence_info['iterations_to_convergence']} iter")
                title_color = 'green'
            else:
                # Find best distance achieved
                if len(phase_data) > 0:
                    best_distance = min([hamming_distance(seq, self.target_sequence) 
                                       for seq in phase_data['sequence']])
                    title_parts.append(f"❌ Best dist: {best_distance}")
                    title_color = 'red'
                else:
                    title_color = 'black'
            
            # Combine title parts
            title = '\n'.join(title_parts)
            ax.set_title(title, color=title_color, fontsize=12)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.legend(loc='upper right')
            
            # Add colorbar for iteration
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Iteration', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Path visualization saved to {save_path}")
        
        return df
    

def main():
    """Run the mutation path exploration analysis."""
    output_dir = Path("examples/mutation_path_output")
    output_dir.mkdir(exist_ok=True)
    
    start_sequence = "FILFKKIEKVARNQRDFIIKHGPEVATVGTATQIAK"
    target_sequence = "KWKLFKKIEKVGRNVRDGIIKAGPAVAAVGQATQIAK"
    
    explorer = MutationPathExplorer(start_sequence, target_sequence)
    
    explorer.run_exploration()
    embeddings = explorer.embed_sequences()
    output_path = output_dir / "mutation_paths_umap.png"
    df = explorer.create_path_visualization(embeddings, str(output_path))

    

if __name__ == "__main__":
    main()