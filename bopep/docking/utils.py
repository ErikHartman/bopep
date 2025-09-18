import os
import glob


def get_pdb_files_in_dir(directory_path: str) -> list:
    """
    Get all PDB files in a directory.

    Parameters:
    - directory_path: Path to directory to search.

    Returns:
    - List of PDB file paths.
    """
    return glob.glob(os.path.join(directory_path, "*.pdb"))