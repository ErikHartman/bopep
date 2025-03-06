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
