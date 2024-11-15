import pandas as pd
import os

def save_results(scores, output_dir, iteration, mode, filename):
    data = []
    for peptide, peptide_scores in scores.items():

        result_entry = {
            "peptide_sequence": peptide,
            "iptm_score": peptide_scores.get("iptm_score"),
            "interface_sasa": peptide_scores.get("interface_sasa"),
            "interface_dG": peptide_scores.get("interface_dG"),
            "rosetta_score": peptide_scores.get("rosetta_score"),
            "interface_delta_hbond_unsat": peptide_scores.get("interface_delta_hbond_unsat"),
            "packstat": peptide_scores.get("packstat"),
            "is_proximate": peptide_scores.get("is_proximate"),
            "iteration": iteration,
            "mode": mode,
        }
        data.append(result_entry)

    df = pd.DataFrame(data)
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

def save_validation_metrics(validation_metrics_log, output_dir, filename):
    # Convert the list of validation metrics dictionaries to a DataFrame
    validation_metrics_df = pd.DataFrame(validation_metrics_log)
    
    # Expand the val_r2s list into separate columns
    val_r2s_df = validation_metrics_df['val_r2s'].apply(pd.Series)
    val_r2s_df.columns = [f'val_r2_model_{i+1}' for i in val_r2s_df.columns]
    
    # Expand the hyperparameters dictionary into separate columns
    hyperparams_df = validation_metrics_df['hyperparameters'].apply(pd.Series)
    
    # Concatenate the dataframes
    validation_metrics_df = pd.concat(
        [validation_metrics_df.drop(['val_r2s', 'hyperparameters'], axis=1),
         val_r2s_df, hyperparams_df],
        axis=1
    )
    
    # Save to CSV
    output_path = os.path.join(output_dir, filename)
    validation_metrics_df.to_csv(output_path, index=False)
    print(f"Validation metrics saved to {output_path}")