import numpy as np
from typing import Dict, Tuple, Optional, Callable, Any


def _ptm(x: np.ndarray, d0: float) -> np.ndarray:
    return 1.0 / (1.0 + (x / d0) ** 2.0)

def _calc_d0_array(L: np.ndarray, pair_type: str) -> np.ndarray:
    """
    Calculate d0 values from counts L according to pair_type rules.
    """
    L = np.asarray(L, dtype=float)
    L = np.maximum(27.0, L)
    min_value = 1.0
    return np.maximum(min_value, 1.24 * (L - 15.0) ** (1.0/3.0) - 1.8)

_PROT_RES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
}

_ALLOWED = _PROT_RES

def compute_ipsae(
    structure_or_model: Any,
    pae: np.ndarray,
    pae_cutoff: float = 10.0,
    residue_selector: Optional[Callable[[Any], bool]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Compute ipSAE for all chain pairs from a Biopython Structure or Model and a PAE array.

    Parameters
    ----------
    structure_or_model : Bio.PDB.Structure.Structure or Bio.PDB.Model.Model
        Parsed with Bio.PDB. The residue order used for scoring is the order
        encountered during iteration over chains then residues.
    pae : array-like
        NxN PAE matrix (Å). A flat vector of size N*N is also accepted and will be reshaped.
    pae_cutoff : float, default 10.0
        PAE threshold for counting inter-chain pairs.
    residue_selector : callable(residue) -> bool, optional
        If provided, receives each Biopython Residue and returns True to include it.
        Use this to enforce the exact residue set and order that matches your PAE.

    Returns
    -------
    results : dict
        Keys are unordered chain-pair tuples with alphabetical order, e.g. ('A','B').
        For each pair:
          {
            'asym': {
               'A->B': float,
               'A->B_res': (chain, resname, resseq, icode),
               'B->A': float,
               'B->A_res': (chain, resname, resseq, icode),
            },
            'max': float,                 # max of the two directions
            'max_dir': 'A->B' or 'B->A',  # direction of the max
          }

    Notes
    -----
    - ipSAE here is the d0_res variant from ipsae.py. For each residue i in chain1,
      d0 is computed from the count of chain2 residues with PAE < cutoff for i, then
      PTM(x, d0_i) is averaged over those valid pairs. The asymmetric chain1->chain2
      ipSAE is the maximum over residues i in chain1. The reported 'max' is the max
      over directions.
    """
    # 1) Gather residues in the intended token order
    # Accept Structure or Model; if Structure, use first model
    try:
        from Bio.PDB.Structure import Structure
        from Bio.PDB.Model import Model
        from Bio.PDB.Residue import Residue
    except Exception:
        pass  # Do not force BioPython import here, caller already has it

    if hasattr(structure_or_model, "get_chains"):
        # Likely a Model
        model = structure_or_model
    else:
        # Likely a Structure with .get_models()
        model = next(iter(structure_or_model.get_models()))

    tokens = []  # [(chain_id, resname, resseq, icode)]
    chain_to_resnames = {}  # for NA/protein classification
    for ch in model.get_chains():
        cid = ch.id
        chain_to_resnames.setdefault(cid, [])
        for res in ch.get_residues():
            # Biopython residue.id is (hetflag, resseq, icode)
            hetflag, resseq, icode = res.id
            resname = res.get_resname().strip()
            if residue_selector is not None:
                keep = residue_selector(res)
            else:
                # default: standard AAs only, no waters or ligands
                keep = (hetflag == " " and resname in _ALLOWED)
            if not keep:
                continue
            tokens.append((cid, resname, int(resseq), icode if icode != " " else None))
            chain_to_resnames[cid].append(resname)

    N = len(tokens)
    if N == 0:
        raise ValueError("No residues selected. Provide a residue_selector that matches your PAE tokens.")

    pae = np.asarray(pae, dtype=float)
    if pae.ndim == 1:
        if pae.size != N * N:
            raise ValueError(f"Flat PAE has size {pae.size}, expected {N*N}.")
        pae = pae.reshape(N, N)
    elif pae.ndim == 2:
        if pae.shape != (N, N):
            raise ValueError(f"PAE shape {pae.shape} does not match number of residues {N}.")
    else:
        raise ValueError("PAE must be a 2D matrix or a flat vector of size N*N.")

    # 2) Vectorized bookkeeping
    chains = np.array([t[0] for t in tokens])  # shape (N,)
    uniq = np.unique(chains)


    # 3) Compute ipSAE_d0res asym values in both directions for each chain pair
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i1, c1 in enumerate(uniq):
        for i2, c2 in enumerate(uniq):
            if c1 >= c2:
                continue  # handle each unordered pair once; compute both directions inside

            pair_key = (c1, c2)
            pair_type_12 = "protein"
            pair_type_21 = pair_type_12  # symmetric type

            # Precompute masks and valid-pair matrices for both directions
            mask_c1 = chains == c1
            mask_c2 = chains == c2

            # valid_pairs_matrix[i, j] means residue i in c1 to residue j in c2 under cutoff
            vpm_12 = (chains[None, :] == c2) & (pae < pae_cutoff)
            vpm_21 = (chains[None, :] == c1) & (pae < pae_cutoff)

            # Direction c1 -> c2
            # Count valid partners per residue i, then compute d0_i and mean PTM over valid pairs
            n0res_byres_12 = np.sum(vpm_12, axis=1)  # shape (N,)
            d0res_byres_12 = _calc_d0_array(n0res_byres_12, pair_type_12)

            ipsae_byres_12 = np.zeros(N, dtype=float)
            for i in np.where(mask_c1)[0]:
                if n0res_byres_12[i] == 0:
                    ipsae_byres_12[i] = 0.0
                    continue
                cols = vpm_12[i]  # boolean mask over j in c2 with PAE<cutoff
                d0_i = float(d0res_byres_12[i])
                ipsae_byres_12[i] = _ptm(pae[i, cols], d0_i).mean()

            if np.any(mask_c1):
                idx12 = np.argmax(ipsae_byres_12 * mask_c1)  # zeros outside mask; safe
                ipsae_12 = float(ipsae_byres_12[idx12])
                best_res_12 = tokens[idx12]  # (chain, resname, resseq, icode)
            else:
                ipsae_12 = 0.0
                best_res_12 = None

            # Direction c2 -> c1
            n0res_byres_21 = np.sum(vpm_21, axis=1)
            d0res_byres_21 = _calc_d0_array(n0res_byres_21, pair_type_21)

            ipsae_byres_21 = np.zeros(N, dtype=float)
            for i in np.where(mask_c2)[0]:
                if n0res_byres_21[i] == 0:
                    ipsae_byres_21[i] = 0.0
                    continue
                cols = vpm_21[i]  # valid j in c1
                d0_i = float(d0res_byres_21[i])
                ipsae_byres_21[i] = _ptm(pae[i, cols], d0_i).mean()

            if np.any(mask_c2):
                idx21 = np.argmax(ipsae_byres_21 * mask_c2)
                ipsae_21 = float(ipsae_byres_21[idx21])
                best_res_21 = tokens[idx21]
            else:
                ipsae_21 = 0.0
                best_res_21 = None

            # Max over directions
            if ipsae_12 >= ipsae_21:
                ip_max = ipsae_12
                max_dir = f"{c1}->{c2}"
            else:
                ip_max = ipsae_21
                max_dir = f"{c2}->{c1}"

            results[pair_key] = {
                "asym": {
                    f"{c1}->{c2}": ipsae_12,
                    f"{c1}->{c2}_res": best_res_12,
                    f"{c2}->{c1}": ipsae_21,
                    f"{c2}->{c1}_res": best_res_21,
                },
                "max": ip_max,
                "max_dir": max_dir,
            }

    return results


def get_ipsae_scores_from_structure_and_pae(
    structure_file: str, 
    pae_data: np.ndarray,
    receptor_chain: str = "A",
    sequence_chain: str = "B",
    pae_cutoff: float = 10.0,
    residue_selector: Optional[Callable[[Any], bool]] = None
) -> Dict[str, float]:
    """
    Compute IPSAE scores from structure file (PDB/CIF) and PAE data.
    """
    from bopep.structure.parser import parse_structure
    
    structure = parse_structure(structure_file, structure_id="model")
    
    results = compute_ipsae(
        structure, 
        pae_data, 
        pae_cutoff=pae_cutoff,
        residue_selector=residue_selector
    )
    
    # Extract values specifically for the receptor-sequence pair
    # Create a normalized pair key (alphabetically sorted)
    pair_key = tuple(sorted([receptor_chain, sequence_chain]))
    
    if pair_key in results:
        pair_data = results[pair_key]
        asym_data = pair_data.get('asym', {})
        
        # Get both asymmetric directions
        direction_values = []
        for key, value in asym_data.items():
            if not key.endswith('_res') and isinstance(value, (int, float)):
                direction_values.append(value)
        
        if direction_values:
            ipsae_max = max(direction_values)
            ipsae_min = min(direction_values)
        else:
            ipsae_max = pair_data.get('max', 0.0)
            ipsae_min = pair_data.get('max', 0.0)
    else:
        # Fallback to old behavior if specific pair not found
        max_values = []
        min_values = []
        
        for pair_key, pair_data in results.items():
            max_val = pair_data.get('max', 0.0)
            asym_data = pair_data.get('asym', {})
            
            # Get both asymmetric directions
            direction_values = []
            for key, value in asym_data.items():
                if not key.endswith('_res') and isinstance(value, (int, float)):
                    direction_values.append(value)
            
            if direction_values:
                max_values.append(max(direction_values))
                min_values.append(min(direction_values))
            else:
                max_values.append(max_val)
                min_values.append(max_val)
        
        # Return overall max and min across all chain pairs
        ipsae_max = max(max_values) if max_values else 0.0
        ipsae_min = min(min_values) if min_values else 0.0
    
    return {
        'ipsae_max': ipsae_max,
        'ipsae_min': ipsae_min
    }


if __name__ == "__main__":
    # Example usage
    from bopep.structure.parser import parse_structure
    import json

    pdb_file = "/srv/data1/er8813ha/bopep/docked/cd14/4glf_NYLSELSEHV/4glf_NYLSELSEHV_relaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb"
    pae_file = "/srv/data1/er8813ha/bopep/docked/cd14/4glf_NYLSELSEHV/4glf_NYLSELSEHV_predicted_aligned_error_v1.json"

    structure = parse_structure(pdb_file, structure_id="model")

    pae_data = json.load(open(pae_file))["predicted_aligned_error"]

    results = compute_ipsae(structure, pae_data, pae_cutoff=10.0)

    print(results)

    # boltz

    root_path = "/srv/data1/er8813ha/bopep/docked/cd14_processed/raw/boltz/4glf_NYLSELSEHV/boltz_results_4glf_NYLSELSEHV/predictions/4glf_NYLSELSEHV/"
    
    import numpy as np

    pae_data = np.load(f"{root_path}/pae_4glf_NYLSELSEHV_model_0.npz")["pae"]
    pdb_file = f"{root_path}/4glf_NYLSELSEHV_model_0.pdb"
    structure2 = parse_structure(pdb_file, structure_id="model")
    print("Boltz")

    results = compute_ipsae(structure2, pae_data, pae_cutoff=10.0)

    print(results)