import math
from bopep.structure.parser import get_chain_coordinates

"""
Distance-based loss function for protein-peptide interactions.
"""


def distance_score_from_structure(structure_file_path : str, receptor_chain : str ="A", peptide_chain : str ="B", threshold : float = 8.0):
    """
    1) Parses the structure file (PDB/CIF) to get coordinates/bfactors
    2) Computes the distance-based score
    """
    receptor_coords = get_chain_coordinates(
        structure_file_path, receptor_chain
    )
    peptide_coords = get_chain_coordinates(
        structure_file_path, peptide_chain
    )
    if not receptor_coords or not peptide_coords:
        return 0.0
        
    # Identify receptor alpha carbons within threshold distance of any peptide alpha carbon
    interface_atoms = []
    for r_atom in receptor_coords:
        for p_atom in peptide_coords:
            dist = math.dist(r_atom, p_atom)
            if dist <= threshold:
                interface_atoms.append(r_atom)
                break
    
    if not interface_atoms:
        return float('inf')  # No interface atoms found
    
    # For each interface alpha carbon, find distance to nearest peptide alpha carbon
    nearest_distances = []
    for r_atom in interface_atoms:
        min_dist = float('inf')
        for p_atom in peptide_coords:
            dist = math.dist(r_atom, p_atom)
            if dist < min_dist:
                min_dist = dist
        nearest_distances.append(min_dist)
    
    # Calculate mean of nearest distances
    return sum(nearest_distances) / len(nearest_distances)



if __name__ == "__main__":
    structure_file_path = "./data/1ssc.pdb"
    score = distance_score_from_structure(structure_file_path)
    print(f"Distance score for {structure_file_path}: {score}")
