from bopep.docking.utils import extract_sequence_from_pdb
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np


class PeptideProperties:

    def __init__(self, pdb_file: str):
        self.peptide_sequence = extract_sequence_from_pdb(pdb_file, chain_id="B")
        self.pa = ProteinAnalysis(self.peptide_sequence)

    def get_molecular_weight(self):
        return self.pa.molecular_weight()

    def get_aromaticity(self):
        return self.pa.aromaticity()

    def get_instability_index(self):
        return self.pa.instability_index()

    def get_isoelectric_point(self):
        return self.pa.isoelectric_point()

    def get_gravy(self):
        return self.pa.gravy()

    def get_helix_fraction(self):
        return self.pa.secondary_structure_fraction()[0]

    def get_turn_fraction(self):
        return self.pa.secondary_structure_fraction()[1]

    def get_sheet_fraction(self):
        return self.pa.secondary_structure_fraction()[2]

    def get_hydrophobic_aa_percent(self):
        aa_composition = self.pa.amino_acids_percent
        hydrophobic_aa_percent = sum([aa_composition[aa] for aa in ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y']])
        return hydrophobic_aa_percent

    def get_polar_aa_percent(self):
        aa_composition = self.pa.amino_acids_percent
        polar_aa_percent = sum([aa_composition[aa] for aa in ['N', 'C', 'Q', 'S', 'T']])
        return polar_aa_percent

    def get_positively_charged_aa_percent(self):
        aa_composition = self.pa.amino_acids_percent
        positively_charged = sum([aa_composition[aa] for aa in ['K', 'R', 'H']])
        return positively_charged

    def get_negatively_charged_aa_percent(self):
        aa_composition = self.pa.amino_acids_percent
        negatively_charged = sum([aa_composition[aa] for aa in ['D', 'E']])
        return negatively_charged

    def get_delta_net_charge_frac(self):
        aa_composition = self.pa.amino_acids_percent
        positively_charged = sum([aa_composition[aa] for aa in ['K', 'R', 'H']])
        negatively_charged = sum([aa_composition[aa] for aa in ['D', 'E']])
        net_charge = positively_charged - negatively_charged
        return net_charge

    def get_uHrel(self):
        return float(self._compute_uHrel())

    def get_all_properties(self):
        return {
            'length': len(self.peptide_sequence),
            'molecular_weight': self.get_molecular_weight(),
            'aromaticity': self.get_aromaticity(),
            'instability_index': self.get_instability_index(),
            'isoelectric_point': self.get_isoelectric_point(),
            'gravy': self.get_gravy(),
            'helix_fraction': self.get_helix_fraction(),
            'turn_fraction': self.get_turn_fraction(),
            'sheet_fraction': self.get_sheet_fraction(),
            'hydrophobic_aa_percent': self.get_hydrophobic_aa_percent(),
            'polar_aa_percent': self.get_polar_aa_percent(),
            'positively_charged_aa_percent': self.get_positively_charged_aa_percent(),
            'negatively_charged_aa_percent': self.get_negatively_charged_aa_percent(),
            'delta_net_charge_frac': self.get_delta_net_charge_frac(),
            'uHrel': self.get_uHrel(),
        }

    def _compute_uHrel(self):
        """
        Computes the relative hydrophobic moment (uHrel) of the peptide sequence.
        The hydrophobic moment is a measure of the amphipathicity of a peptide sequence.
        It is calculated as the vector sum of the hydrophobicity values of the amino acids
        in the sequence, with the hydrophobicity values weighted by the angle of the amino
        acid in the sequence. The relative hydrophobic moment is then calculated as the
        hydrophobic moment divided by the sum of the absolute hydrophobicity values.
        """
        # Default hydrophobicity scale (Kyte-Doolittle)
        hydrophobicity_scale = {
            "A": 1.8,
            "R": -4.5,
            "N": -3.5,
            "D": -3.5,
            "C": 2.5,
            "Q": -3.5,
            "E": -3.5,
            "G": -0.4,
            "H": -3.2,
            "I": 4.5,
            "L": 3.8,
            "K": -3.9,
            "M": 1.9,
            "F": 2.8,
            "P": -1.6,
            "S": -0.8,
            "T": -0.7,
            "W": -0.9,
            "Y": -1.3,
            "V": 4.2,
        }

        # Helix angle (100 degrees per residue)
        helix_angle = 100 * (np.pi / 180)  # Convert to radians

        # Initialize variables
        hydrophobic_moment_x = 0
        hydrophobic_moment_y = 0
        sum_abs_hydrophobicity = 0

        # Calculate hydrophobic moment
        for i, aa in enumerate(self.peptide_sequence):
            hydrophobicity = hydrophobicity_scale.get(aa, None)
            if hydrophobicity is None:
                raise ValueError(f"Unknown amino acid '{aa}' in peptide sequence.")

            angle = i * helix_angle
            hydrophobic_moment_x += hydrophobicity * np.cos(angle)
            hydrophobic_moment_y += hydrophobicity * np.sin(angle)
            sum_abs_hydrophobicity += abs(hydrophobicity)

        hydrophobic_moment = np.sqrt(hydrophobic_moment_x**2 + hydrophobic_moment_y**2)

        # Normalize to get uHrel (relative hydrophobic moment)
        uHrel = (
            hydrophobic_moment / sum_abs_hydrophobicity
            if sum_abs_hydrophobicity > 0
            else 0
        )

        return uHrel

if __name__ == "__main__":
    pdb_file = "data/1ssc.pdb"
    pp = PeptideProperties(pdb_file)
    print(pp.get_all_properties())
