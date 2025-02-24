from bopep.scoring.loss_evobind import evobind_loss_from_pdb
from bopep.scoring.loss_rosetta import rosetta_scores_from_pdb

class Scorer:
    def __init__(self):
        pass

    def calculate_evobind_score(self, pdb_file):
        return evobind_loss_from_pdb(pdb_file)

    def calculate_rosetta_scores(self, pdb_file):
        return rosetta_scores_from_pdb(pdb_file)
   