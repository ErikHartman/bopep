from bopep.docking.utils import extract_sequence_from_pdb

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
                if line.startswith("ATOM  "):
                    residue_number = line[22:26].strip()
                    try:
                        return int(residue_number)
                    except ValueError:
                        continue         
        return None
        
    except FileNotFoundError:
        print(f"Error: PDB file {pdb_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading PDB file: {e}")
        return None
    

def _check_binding_site_residue_indices(
    binding_site_residue_indices,
    target_structure_path,
    assume_zero_indexed=None,
):
    """
    Checks if starting index is 0.

    If not, asks the user if the binding site residues are expected to be 0-indexed.
    Corrects for the starting index if wanted.

    Return a visualization of the residues that are selected as binding site residues.
    """
    starting_index = check_starting_index_in_pdb(target_structure_path)
    print("PDB chain starts at residue", starting_index)
    protein_sequence = extract_sequence_from_pdb(target_structure_path)
    
    if binding_site_residue_indices is None:
        return None
    
    for residue in binding_site_residue_indices:
        if residue < 0 or residue >= len(protein_sequence):
            raise ValueError(
                f"Binding site residue index {residue} is out of range for protein sequence of length {len(protein_sequence)}"
            )

    # Handle both list and dict formats
    is_dict_format = isinstance(binding_site_residue_indices, dict)
    
    if is_dict_format:
        print(f"Found peptide-specific binding sites for {len(binding_site_residue_indices)} peptides")
        # Get all unique residue indices for validation and visualization
        all_residues = set()
        for residue_list in binding_site_residue_indices.values():
            all_residues.update(residue_list)
        residues_to_check = sorted(list(all_residues))
    else:
        print("Using same binding site for all peptides")
        residues_to_check = binding_site_residue_indices

    if starting_index != 0:
        if assume_zero_indexed is None:
            print(
                f"\n\nStarting index is {starting_index}. Are the provided binding site residues 0-indexed?"
            )
            answer = input("y/n: ")
        else:
            if assume_zero_indexed is True:
                print(
                    f"\n\nStarting index is {starting_index}. Assuming binding site residues are 0-indexed."
                )
                answer = "y"
            else:
                print(
                    f"\n\nStarting index is {starting_index}. Assuming binding site residues are 1-indexed."
                )
                answer = "n"
        
        if answer == "y":
            if is_dict_format:
                # Adjust all residue lists in the dict
                binding_site_residue_indices = {
                    peptide: [residue - starting_index for residue in residue_list]
                    for peptide, residue_list in binding_site_residue_indices.items()
                }
                residues_to_check = [residue - starting_index for residue in residues_to_check]
            else:
                binding_site_residue_indices = [
                    residue - starting_index for residue in binding_site_residue_indices
                ]
                residues_to_check = binding_site_residue_indices

    print("\nBinding Site Residues Visualization:")
    print("=" * 60)
    print(f"Full sequence length: {len(protein_sequence)}")
    if isinstance(binding_site_residue_indices, dict):
        print("Peptide-specific binding sites:")
        for peptide, residues in binding_site_residue_indices.items():
            print(f"  {peptide}: {residues}")
    else:
        print(f"Selected binding site residues: {binding_site_residue_indices}")
    print("-" * 60)

    residues_to_check = sorted(residues_to_check)
    context_size = 5

    for residue_idx in residues_to_check:
        if residue_idx < 0 or residue_idx >= len(protein_sequence):
            print(f"Warning: Residue index {residue_idx} out of range")
            raise ValueError(
                f"Binding site residue index {residue_idx} is out of range for protein sequence of length {len(protein_sequence)}"
            )

        # Calculate start and end positions for context
        start = max(0, residue_idx - context_size)
        end = min(len(protein_sequence), residue_idx + context_size + 1)

        positions = list(range(start + starting_index, end + starting_index))
        vis_seq = list(protein_sequence[start:end])

        # Mark the selected residue
        rel_pos = residue_idx - start
        if 0 <= rel_pos < len(vis_seq):
            vis_seq[rel_pos] = f"[{vis_seq[rel_pos]}]"

        print(
            f"Residue {residue_idx + starting_index} ({protein_sequence[residue_idx]}):"
        )
        print("Position:" + " ".join(f"{pos:3d}" for pos in positions))
        print("Sequence: " + " ".join(f"{aa:3s}" for aa in vis_seq))
        print("-" * 60)

    print("=" * 60)
    

    # increment binding site residues by 1 since alphafold pdbs start at 1
    if isinstance(binding_site_residue_indices, dict):
        binding_site_residue_indices = {
            peptide: [residue + 1 for residue in residue_list]
            for peptide, residue_list in binding_site_residue_indices.items()
        }
        print("The internally stored peptide-specific binding site residues are (1-indexed):")
        for peptide, residues in binding_site_residue_indices.items():
            print(f"  {peptide}: {residues}")
    else:
        binding_site_residue_indices = [
            residue + 1 for residue in binding_site_residue_indices
        ]
        print(
            f"The internally stored binding site residues are: {binding_site_residue_indices} (1-indexed)"
        )
    return binding_site_residue_indices

if __name__ == "__main__":
    # Example usage
    pdb_path = "/home/er8813ha/bopep/data/4glf.pdb"
    binding_site_indices = [23, 42]
    _check_binding_site_residue_indices(binding_site_indices, pdb_path)
    # This will print the binding site residues and their context in the sequence