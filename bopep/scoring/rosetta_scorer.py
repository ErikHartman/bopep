import pyrosetta
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

class RosettaScorer:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.initialized = False
        self.scorefxn = None
        self.pose = None
        self.ia = None
        self.rosetta_score = None

    def _initialize(self):
        if not self.initialized:
            try:
                pyrosetta.get_fa_scorefxn()
            except Exception:
                pyrosetta.init("-mute all")
            self.scorefxn = pyrosetta.get_fa_scorefxn()
            self.pose = pose_from_pdb(self.pdb_file)
            self.rosetta_score = self.scorefxn(self.pose)
            self.ia = InterfaceAnalyzerMover()
            self.ia.set_compute_packstat(True)
            self.ia.apply(self.pose)
            self.initialized = True

    def get_rosetta_score(self):
        if not self.initialized:
            self._initialize()
        return self.rosetta_score

    def get_interface_sasa(self):
        if not self.initialized:
            self._initialize()
        return self.ia.get_interface_delta_sasa()

    def get_interface_dG(self):
        if not self.initialized:
            self._initialize()
        return self.ia.get_interface_dG()

    def get_interface_delta_hbond_unsat(self):
        if not self.initialized:
            self._initialize()
        return self.ia.get_interface_delta_hbond_unsat()

    def get_packstat(self):
        if not self.initialized:
            self._initialize()
        return self.ia.get_interface_packstat()

    def get_all_metrics(self):
        if not self.initialized:
            self._initialize()
        return {
            'rosetta_score': self.get_rosetta_score(),
            'interface_sasa': self.get_interface_sasa(),
            'interface_dG': self.get_interface_dG(),
            'interface_delta_hbond_unsat': self.get_interface_delta_hbond_unsat(),
            'packstat': self.get_packstat()
        }

if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    analyzer = RosettaScorer(pdb_file_path)
    metrics = analyzer.get_all_metrics()
    print(f"Rosetta metrics for {pdb_file_path}: {metrics}")
