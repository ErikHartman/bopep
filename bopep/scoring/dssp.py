import numpy as np
import pydssp
import torch
from bopep.structure.parser import extract_sequence_from_structure, get_all_atom_coordinates


class DSSPAnalyzer:
    """DSSP secondary structure analysis for sequences using pydssp."""
    
    def __init__(self, structure_file: str, chain_id: str = "B"):
        """
        Initialize DSSP analyzer with a structure file.
        
        Parameters
        ----------
        structure_file : str
            Path to structure file (.pdb/.cif)
        chain_id : str, default "B"
            Chain ID to analyze (typically "B" for sequence)
        """
        self.structure_file = structure_file
        self.chain_id = chain_id
        self.sequence = extract_sequence_from_structure(structure_file, chain_id=chain_id)
        
    def get_dssp_helix_fraction(self):
        """Get helix fraction using DSSP."""
        coord = self._extract_coordinates()
        if coord is None:
            raise ValueError(f"Could not extract coordinates from structure file: {self.structure_file}")
        
        dssp_assignment = pydssp.assign(coord, out_type='index')
        # 0: loop, 1: alpha-helix, 2: beta-sheet
        helix_count = torch.sum(dssp_assignment == 1).item()
        return float(helix_count / len(dssp_assignment)) if len(dssp_assignment) > 0 else 0.0

    def get_dssp_sheet_fraction(self):
        """Get sheet/beta-sheet fraction using DSSP."""
        coord = self._extract_coordinates()
        if coord is None:
            raise ValueError(f"Could not extract coordinates from structure file: {self.structure_file}")
        
        dssp_assignment = pydssp.assign(coord, out_type='index')
        # 0: loop, 1: alpha-helix, 2: beta-sheet
        sheet_count = torch.sum(dssp_assignment == 2).item()
        return float(sheet_count / len(dssp_assignment)) if len(dssp_assignment) > 0 else 0.0
        
    def get_dssp_loop_fraction(self):
        """Get loop fraction using DSSP."""
        coord = self._extract_coordinates()
        if coord is None:
            raise ValueError(f"Could not extract coordinates from structure file: {self.structure_file}")
        
        dssp_assignment = pydssp.assign(coord, out_type='index')
        # 0: loop, 1: alpha-helix, 2: beta-sheet
        loop_count = torch.sum(dssp_assignment == 0).item()
        return float(loop_count / len(dssp_assignment)) if len(dssp_assignment) > 0 else 0.0
        
    def get_all_dssp_fractions(self):
        """Get all DSSP secondary structure fractions."""
        return {
            'dssp_helix_fraction': self.get_dssp_helix_fraction(),
            'dssp_sheet_fraction': self.get_dssp_sheet_fraction(),
            'dssp_loop_fraction': self.get_dssp_loop_fraction(),
        }
        
    def _extract_coordinates(self):
        """Extract backbone coordinates (N, CA, C, O) for DSSP calculation using existing parser."""
        try:
            # Get all atom coordinates for the chain using the existing parser
            residue_coords = get_all_atom_coordinates(self.structure_file, self.chain_id)
            
            if not residue_coords:
                raise ValueError(f"No residues found for chain '{self.chain_id}' in structure file: {self.structure_file}")
            
            coords = []
            for residue_num in sorted(residue_coords.keys()):
                atom_coords = residue_coords[residue_num]
                
                # Check if all required backbone atoms are present
                required_atoms = ['N', 'CA', 'C', 'O']
                missing_atoms = [atom for atom in required_atoms if atom not in atom_coords]
                if missing_atoms:
                    # Skip residues with missing backbone atoms but warn
                    continue
                    
                # Extract N, CA, C, O coordinates in the required order
                res_coords = np.array([
                    atom_coords['N'],
                    atom_coords['CA'], 
                    atom_coords['C'],
                    atom_coords['O']
                ])
                coords.append(res_coords)
            
            if len(coords) == 0:
                raise ValueError(f"No residues with complete backbone atoms (N, CA, C, O) found for chain '{self.chain_id}' in structure file: {self.structure_file}")
                
            # Convert to torch tensor with shape [length, 4, 3]
            coord_array = np.array(coords)
            coord_tensor = torch.from_numpy(coord_array).float()
            
            return coord_tensor
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Structure file not found: {self.structure_file}")
        except Exception as e:
            raise ValueError(f"Error extracting coordinates from structure file '{self.structure_file}': {str(e)}")


def get_dssp_scores_from_structure(structure_file: str, chain_id: str = "B"):
    """
    Convenience function to get all DSSP scores from a structure file.
    
    Parameters
    ----------
    structure_file : str
        Path to structure file (.pdb/.cif)
    chain_id : str, default "B"
        Chain ID to analyze
        
    Returns
    -------
    dict
        Dictionary with DSSP fraction scores
    """
    analyzer = DSSPAnalyzer(structure_file, chain_id)
    return analyzer.get_all_dssp_fractions()


if __name__ == "__main__":
    # Test the DSSP analyzer
    structure_file = "/Users/erikhartman/dev/bopep/data/1ssc.pdb"

    analyzer = DSSPAnalyzer(structure_file)
    print(f"Peptide sequence: {analyzer.sequence}")
    print("DSSP analysis:")
    
    fractions = analyzer.get_all_dssp_fractions()
    for key, value in fractions.items():
        print(f"{key}: {value}")
