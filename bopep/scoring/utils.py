from bopep.structure.parser import get_chain_coordinates

def match_and_truncate(ref_seq :  str, ref_coords : list, target_seq : str, target_coords : list):
    if ref_seq in target_seq:
        i = target_seq.index(ref_seq)
        return ref_coords, target_coords[i:i+len(ref_seq)]
    elif target_seq in ref_seq:
        i = ref_seq.index(target_seq)
        return ref_coords[i:i+len(target_seq)], target_coords
    else:
        raise ValueError(f"Could not match reference and target receptor sequences for alignment. Reference sequence: {ref_seq}, Target sequence: {target_seq}")