#!/usr/bin/env python
# filepath: /home/er8813ha/bopep/examples/generate_dummy_logs.py

"""
Script to generate dummy output files mimicking those produced by the BoPep.optimize() method.
These files can be used for experimenting with plotting functions without running the
full optimization process.
"""

import os
import csv
from datetime import datetime, timedelta
import random
import numpy as np
import string
import argparse

def generate_peptide_sequences(num_peptides, length_range=(5, 15)):
    """Generate random peptide sequences."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    peptides = []
    
    for _ in range(num_peptides):
        length = random.randint(length_range[0], length_range[1])
        peptide = ''.join(random.choice(amino_acids) for _ in range(length))
        peptides.append(peptide)
    
    return peptides

def create_directory(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def generate_mock_scores(peptides, num_initial, batch_size, schedule, scores_to_include=None):
    """
    Generate mock scores for peptides that would be produced during optimization.
    
    Args:
        peptides: List of peptide sequences
        num_initial: Number of initial peptides
        batch_size: Number of peptides per batch
        schedule: List of dicts with 'acquisition' and 'iterations' keys
        scores_to_include: List of score types to include
    
    Returns:
        Dictionary mapping iteration to scores for that iteration
    """
    if scores_to_include is None:
        scores_to_include = ["binding_energy", "shape_complementarity", "interface_area", "peptide_rmsd"]
    
    result = {}
    
    # Initial peptides (iteration 0)
    initial_peptides = peptides[:num_initial]
    initial_scores = {}
    for peptide in initial_peptides:
        peptide_scores = {}
        for score_type in scores_to_include:
            # Generate a reasonable range of scores for each type
            if score_type == "binding_energy":
                peptide_scores[score_type] = random.uniform(-12.0, -5.0)  # Lower is better
            elif score_type == "shape_complementarity":
                peptide_scores[score_type] = random.uniform(0.4, 0.9)  # Higher is better
            elif score_type == "interface_area":
                peptide_scores[score_type] = random.uniform(500, 2000)  # Higher is better
            elif score_type == "peptide_rmsd":
                peptide_scores[score_type] = random.uniform(0.5, 5.0)  # Lower is better
            else:
                peptide_scores[score_type] = random.uniform(0.0, 1.0)
        initial_scores[peptide] = peptide_scores
    
    result[0] = initial_scores
    
    # Scores for iterations in schedule
    iteration = 1
    peptide_index = num_initial
    
    for phase in schedule:
        iterations = phase["iterations"]
        
        for i in range(iterations):
            if peptide_index + batch_size > len(peptides):
                print("Warning: Not enough peptides for all iterations. Reusing peptides.")
                peptide_index = 0
                
            batch_peptides = peptides[peptide_index:peptide_index + batch_size]
            batch_scores = {}
            
            for peptide in batch_peptides:
                peptide_scores = {}
                for score_type in scores_to_include:
                    # As iterations increase, we expect scores to improve
                    improvement_factor = 1 + (iteration * 0.05)  # 5% improvement per iteration
                    
                    if score_type == "binding_energy":
                        # Lower is better, so subtract improvement
                        peptide_scores[score_type] = random.uniform(-12.0, -8.0) * improvement_factor
                    elif score_type == "shape_complementarity":
                        # Higher is better
                        base = random.uniform(0.4, 0.8)
                        peptide_scores[score_type] = min(0.95, base * improvement_factor)
                    elif score_type == "interface_area":
                        # Higher is better
                        base = random.uniform(600, 1800) 
                        peptide_scores[score_type] = base * improvement_factor
                    elif score_type == "peptide_rmsd":
                        # Lower is better, so divide by improvement
                        base = random.uniform(1.0, 4.0)
                        peptide_scores[score_type] = base / improvement_factor
                    else:
                        peptide_scores[score_type] = random.uniform(0.0, 1.0) * improvement_factor
                
                batch_scores[peptide] = peptide_scores
            
            result[iteration] = batch_scores
            iteration += 1
            peptide_index += batch_size
    
    return result

def generate_mock_model_losses(schedule):
    """Generate mock model training losses."""
    result = []
    
    for phase_idx, phase in enumerate(schedule):
        acquisition = phase["acquisition"]
        iterations = phase["iterations"]
        
        for iteration in range(iterations):
            # Loss should decrease over iterations
            base_loss = 1.0 / (1 + 0.2 * (phase_idx * iterations + iteration))
            loss = max(0.01, random.uniform(base_loss * 0.8, base_loss * 1.2))
            
            # R² should increase over iterations, starting around 0.3-0.5 and going up to 0.7-0.9
            base_r2 = 0.3 + 0.6 * (1 - 1.0 / (1 + 0.15 * (phase_idx * iterations + iteration)))
            r2 = min(0.95, random.uniform(base_r2 * 0.9, base_r2 * 1.1))
            
            result.append((iteration, loss, r2))
    
    return result

def generate_mock_predictions(peptides, num_initial, batch_size, schedule):
    """Generate mock model predictions."""
    result = {}
    
    # Set of peptides already "docked" (initially empty)
    docked_peptides = set()
    
    # Track current iteration
    current_iter = 0
    
    # Add initial peptides to docked set
    initial_peptides = set(peptides[:num_initial])
    docked_peptides.update(initial_peptides)
    
    # For each phase
    for phase in schedule:
        iterations = phase["iterations"]
        
        for i in range(iterations):
            # Generate predictions for undocked peptides
            undocked_peptides = set(peptides) - docked_peptides
            predictions = {}
            
            for peptide in undocked_peptides:
                # As iterations increase, mean predictions should improve
                improvement_factor = 1 + (current_iter * 0.08)
                
                # Mean value should be between -10 and -2, improving with iterations
                mean = random.uniform(-10, -5) * improvement_factor
                
                # Standard deviation should decrease with iterations
                std = random.uniform(0.5, 2.0) / improvement_factor
                
                predictions[(current_iter, peptide)] = (mean, std)
            
            # Select next batch of peptides to "dock"
            peptide_index = num_initial + ((current_iter - 0) * batch_size)
            if peptide_index + batch_size <= len(peptides):
                batch_peptides = set(peptides[peptide_index:peptide_index + batch_size])
                docked_peptides.update(batch_peptides)
            
            # Store predictions for this iteration
            result[current_iter] = predictions
            current_iter += 1
    
    return result

def generate_mock_acquisition_values(peptides, num_initial, batch_size, schedule):
    """Generate mock acquisition function values."""
    result = {}
    
    # Set of peptides already "docked" (initially empty)
    docked_peptides = set()
    
    # Add initial peptides to docked set
    initial_peptides = set(peptides[:num_initial])
    docked_peptides.update(initial_peptides)
    
    # Track current iteration and acquisition function
    current_iter = 0
    
    # For each phase
    for phase in schedule:
        acquisition = phase["acquisition"]
        iterations = phase["iterations"]
        
        for i in range(iterations):
            # Generate acquisition values for undocked peptides
            undocked_peptides = set(peptides) - docked_peptides
            acquisition_values = {}
            
            for peptide in undocked_peptides:
                # Generate acquisition values based on the type of acquisition function
                if acquisition == "expected_improvement":
                    # EI values are typically small positive numbers
                    value = random.uniform(0, 2.0) * (1 + 0.1 * current_iter)
                elif acquisition == "upper_confidence_bound":
                    # UCB values can be larger
                    value = random.uniform(-15, -5) - (0.2 * current_iter)
                elif acquisition == "standard_deviation":
                    # Standard deviation typically decreases over iterations
                    value = random.uniform(0.5, 3.0) / (1 + 0.1 * current_iter)
                elif acquisition == "probability_of_improvement":
                    # Probability values between 0 and 1
                    value = random.uniform(0.1, 0.9)
                else:  # "mean"
                    # Mean predicted values
                    value = random.uniform(-12, -5) - (0.2 * current_iter)
                    
                acquisition_values[(current_iter, acquisition, peptide)] = value
            
            # Select next batch of peptides to "dock"
            peptide_index = num_initial + (current_iter * batch_size)
            if peptide_index + batch_size <= len(peptides):
                batch_peptides = set(peptides[peptide_index:peptide_index + batch_size])
                docked_peptides.update(batch_peptides)
            
            # Store acquisition values for this iteration
            result[(current_iter, acquisition)] = acquisition_values
            current_iter += 1
    
    return result

def generate_mock_objectives(scores):
    """
    Generate mock objective values based on the scores.
    
    Args:
        scores: Dictionary mapping iteration to score dictionaries
        
    Returns:
        Dictionary mapping iteration to objective dictionaries
    """
    result = {}
    
    for iteration, iteration_scores in scores.items():
        objectives = {}
        
        for peptide, peptide_scores in iteration_scores.items():
            # Create a weighted combination of scores to generate an objective value
            # Lower binding energy, higher shape complementarity, higher interface area, lower RMSD
            binding_energy = peptide_scores.get("binding_energy", 0)
            shape_comp = peptide_scores.get("shape_complementarity", 0)
            interface_area = peptide_scores.get("interface_area", 0) / 2000  # Normalize to 0-1 range
            peptide_rmsd = peptide_scores.get("peptide_rmsd", 0)
            
            # Combine scores into an objective (lower is better)
            # Negative binding energy is good, so we'll keep it negative
            # Inverse shape complementarity and interface area since higher is better
            # RMSD is already correct (lower is better)
            objective = binding_energy * 0.5 - shape_comp * 5.0 - interface_area * 3.0 + peptide_rmsd * 2.0
            
            # Add some randomness
            objective = objective * random.uniform(0.9, 1.1)
            
            objectives[peptide] = objective
        
        result[iteration] = objectives
    
    return result

def write_scores_csv(log_dir, all_scores):
    """Write scores to CSV file."""
    filepath = os.path.join(log_dir, "scores.csv")
    
    # Get all score types
    score_types = set()
    for iteration_scores in all_scores.values():
        for peptide_scores in iteration_scores.values():
            score_types.update(peptide_scores.keys())
    score_types = sorted(score_types)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "iteration", "peptide"] + score_types)
        
        base_time = datetime.now()
        
        for iteration, scores in sorted(all_scores.items()):
            timestamp = (base_time + timedelta(minutes=5*iteration)).isoformat()
            
            for peptide, peptide_scores in scores.items():
                row = [timestamp, iteration, peptide]
                for score_type in score_types:
                    row.append(peptide_scores.get(score_type, None))
                writer.writerow(row)
    
    print(f"Scores saved to {filepath}")

def write_model_losses_csv(log_dir, model_losses):
    """Write model losses to CSV file."""
    filepath = os.path.join(log_dir, "model_losses.csv")
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "iteration", "epoch_in_fit", "loss", "r2"])
        
        base_time = datetime.now()
        
        for idx, (iteration, loss, r2) in enumerate(model_losses):
            timestamp = (base_time + timedelta(minutes=5*idx)).isoformat()
            epoch_in_fit = 100  # Assuming 100 epochs per fit
            writer.writerow([timestamp, iteration, epoch_in_fit, loss, r2])
    
    print(f"Model losses saved to {filepath}")

def write_predictions_csv(log_dir, predictions):
    """Write predictions to CSV file."""
    filepath = os.path.join(log_dir, "predictions.csv")
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "iteration", "peptide", "mean", "std"])
        
        base_time = datetime.now()
        
        for iteration, iter_predictions in sorted(predictions.items()):
            timestamp = (base_time + timedelta(minutes=5*iteration)).isoformat()
            
            for (_, peptide), (mean, std) in iter_predictions.items():
                writer.writerow([timestamp, iteration, peptide, mean, std])
    
    print(f"Predictions saved to {filepath}")

def write_acquisition_csv(log_dir, acquisition_values):
    """Write acquisition values to CSV file."""
    filepath = os.path.join(log_dir, "acquisition.csv")
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "iteration", "peptide", "acquisition_name", "acquisition_value"])
        
        base_time = datetime.now()
        
        for (iteration, acquisition), values in sorted(acquisition_values.items()):
            timestamp = (base_time + timedelta(minutes=5*iteration)).isoformat()
            
            for (_, _, peptide), value in values.items():
                writer.writerow([timestamp, iteration, peptide, acquisition, value])
    
    print(f"Acquisition values saved to {filepath}")

def write_objectives_csv(log_dir, objectives):
    """Write objective values to CSV file."""
    filepath = os.path.join(log_dir, "objectives.csv")
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "iteration", "peptide", "objective"])
        
        base_time = datetime.now()
        
        for iteration, iteration_objectives in sorted(objectives.items()):
            timestamp = (base_time + timedelta(minutes=5*iteration)).isoformat()
            
            for peptide, objective_value in iteration_objectives.items():
                writer.writerow([timestamp, iteration, peptide, objective_value])
    
    print(f"Objective values saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate dummy log files for BoPep optimization")
    parser.add_argument("--log-dir", type=str, default="dummy_logs", 
                        help="Directory to save log files (default: dummy_logs)")
    parser.add_argument("--num-peptides", type=int, default=100, 
                        help="Number of peptides to generate (default: 100)")
    parser.add_argument("--num-initial", type=int, default=10, 
                        help="Number of initial peptides (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Batch size for each iteration (default: 4)")
    
    args = parser.parse_args()
    
    # Create log directory
    create_directory(args.log_dir)
    
    # Generate peptides
    peptides = generate_peptide_sequences(args.num_peptides)
    
    # Define schedule (same as default in BoPep.optimize)
    schedule = [
        {"acquisition": "standard_deviation", "iterations": 10},
        {"acquisition": "expected_improvement", "iterations": 10},
    ]
    
    # Generate mock data
    scores = generate_mock_scores(peptides, args.num_initial, args.batch_size, schedule)
    objectives = generate_mock_objectives(scores)
    model_losses = generate_mock_model_losses(schedule)
    predictions = generate_mock_predictions(peptides, args.num_initial, args.batch_size, schedule)
    acquisition_values = generate_mock_acquisition_values(peptides, args.num_initial, args.batch_size, schedule)
    
    # Write CSV files
    write_scores_csv(args.log_dir, scores)
    write_objectives_csv(args.log_dir, objectives)
    write_model_losses_csv(args.log_dir, model_losses)
    write_predictions_csv(args.log_dir, predictions)
    write_acquisition_csv(args.log_dir, acquisition_values)
    
    print(f"All dummy log files have been generated in {args.log_dir}")
    print(f"Total peptides: {len(peptides)}")
    print(f"Initial peptides: {args.num_initial}")
    print(f"Batch size: {args.batch_size}")
    print(f"Schedule: {schedule}")

if __name__ == "__main__":
    main()
