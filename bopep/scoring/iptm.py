import os
import json
import re


def get_ipTM_from_dir(colab_dir: str, rank_num: int = 1):
    """
    Extracts the ipTM score from an unzipped docking result directory.
    """
    if not os.path.isdir(colab_dir):
        print(f"Directory {colab_dir} does not exist.")
        return None

    json_pattern = re.compile(rf".*_scores_rank_00{rank_num}_.*\.json")
    json_files = []

    for root, _, files in os.walk(colab_dir):
        json_files.extend(
            [os.path.join(root, f) for f in files if json_pattern.search(f)]
        )

    if not json_files:
        print(f"No matching JSON file found in {colab_dir}")
        return None

    try:
        with open(json_files[0], "r") as f:
            return json.load(f).get("iptm")
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return None
