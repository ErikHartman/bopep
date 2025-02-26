import os
import json
import re

def get_ipTM_from_dir(colab_dir):
    """
    Extracts the ipTM score from an unzipped docking result directory.

    Parameters:
    - colab_dir: Directory containing the unzipped docking result.

    Returns:
    - ipTM score as a float, or None if not found.
    """
    if not os.path.isdir(colab_dir):
        print(f"Directory {colab_dir} does not exist.")
        return None

    json_pattern = re.compile(r".*_scores_rank_001_.*\.json")
    json_files = []
    
    # Walk through directory to find matching JSON files
    for root, _, files in os.walk(colab_dir):
        json_files.extend([os.path.join(root, f) for f in files if json_pattern.search(f)])

    if not json_files:
        print(f"No matching JSON file found in {colab_dir}")
        return None

    try:
        with open(json_files[0], 'r') as f:
            return json.load(f).get("iptm")
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return None
