import math
from bopep.structure.parser import get_chain_coordinates

"""
Distance-based loss function for protein-sequence interactions.
"""


def distance_score_from_structure(structure_file_path : str, receptor_chain : str ="A", sequence_chain : str ="B", threshold : float = 8.0):
    """
    1) Parses the structure file (PDB/CIF) to get coordinates/bfactors
    2) Computes the distance-based score which is the mean distance from each receptor alpha carbon
    """
    receptor_coords = get_chain_coordinates(
        structure_file_path, receptor_chain
    )
    sequence_coords = get_chain_coordinates(
        structure_file_path, sequence_chain
    )
    if not receptor_coords or not sequence_coords:
        return 0.0
        
    # Identify receptor alpha carbons within threshold distance of any sequence alpha carbon
    interface_atoms = []
    for r_atom in receptor_coords:
        for p_atom in sequence_coords:
            dist = math.dist(r_atom, p_atom)
            if dist <= threshold:
                interface_atoms.append(r_atom)
                break
    
    if not interface_atoms:
        return float('inf')  # No interface atoms found
    
    # For each interface alpha carbon, find distance to nearest sequence alpha carbon
    nearest_distances = []
    for r_atom in interface_atoms:
        min_dist = float('inf')
        for p_atom in sequence_coords:
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
