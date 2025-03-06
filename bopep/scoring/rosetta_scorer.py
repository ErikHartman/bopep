import pyrosetta
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

class RosettaScorer:
    def __init__(self, pdb_file):
        try:
            self.scorefxn = pyrosetta.get_fa_scorefxn()
        except Exception:
            pyrosetta.init("-mute all")
            self.scorefxn = pyrosetta.get_fa_scorefxn()
        
        self.pose = pose_from_pdb(pdb_file)
        self.rosetta_score = self.scorefxn(self.pose)
        
        self.ia = InterfaceAnalyzerMover()
        self.ia.set_compute_packstat(True)
        self.ia.apply(self.pose)
    
    def get_rosetta_score(self):
        return self.rosetta_score

    def get_interface_sasa(self):
        return self.ia.get_interface_delta_sasa()
    
    def get_interface_dG(self):
        return self.ia.get_interface_dG()
    
    def get_interface_delta_hbond_unsat(self):
        return self.ia.get_interface_delta_hbond_unsat()
    
    def get_packstat(self):
        return self.ia.get_interface_packstat()
    
    def get_all_metrics(self):
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
