import torch


def check_starting_index_in_pdb(pdb_file: str) -> int:
    """
    Parses PDB file and checks what the starting residue index is. 
    Some PDB files are not 0-indexed.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        The starting residue index (usually 1 for standard PDB files), 
        or None if no valid residue is found
    """
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # Look for ATOM records which define coordinates for standard residues
                if line.startswith("ATOM  "):
                    # Extract residue number from columns 23-26 (0-indexed)
                    residue_number = line[22:26].strip()
                    
                    # Try to parse as integer
                    try:
                        return int(residue_number)
                    except ValueError:
                        continue         
        # If we get here, we didn't find any valid ATOM records with residue numbers
        return None
        
    except FileNotFoundError:
        print(f"Error: PDB file {pdb_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading PDB file: {e}")
        return None


def load_model(self, load_path: str):
    """
    Load a saved model with its configuration.
    
    Args:
        load_path: Path to the saved model file
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    
    # Restore configuration
    self.surrogate_model_kwargs = checkpoint['surrogate_model_kwargs']
    self.best_hyperparams = checkpoint['model_config']['hyperparameters']
    
    # Initialize the model with the saved configuration
    self._initialize_model(self.best_hyperparams)
    
    # Load the state dictionary
    self.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {load_path}")
    print(f"Model type: {checkpoint['model_config']['model_type']}")
    print(f"Network architecture: {checkpoint['model_config']['network_type']}")
    print(f"Model class: {checkpoint['model_class']}")
    
    return checkpoint['model_config']