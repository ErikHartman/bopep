from abc import ABC, abstractmethod

class Scorer(ABC):
    """
    Abstract base class for scoring functions.
    All scoring classes should inherit from this class.
    """

    @abstractmethod
    def score(self, pdb_path: str):
        """
        Calculate a score for the given protein-peptide interaction.
        :param pdb_path: Path to the PDB file containing the protein-peptide complex
        :return: A float indicating the predicted binding affinity or score
        """
        pass