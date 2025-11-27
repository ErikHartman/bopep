import os
import glob


def get_pdb_files_in_dir(directory_path: str) -> list:
    """
    Get all PDB files in a directory.
    """
    return glob.glob(os.path.join(directory_path, "*.pdb"))