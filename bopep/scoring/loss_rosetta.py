import pyrosetta

from pyrosetta import Pose
from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

def rosetta_scores_from_pdb(pdb_file):
    try:
        Pose()
    except:
        pyrosetta.init("-mute all")

    pose = pose_from_pdb(pdb_file)

    scorefxn = pyrosetta.get_fa_scorefxn()  
    rosetta_score = scorefxn(pose)

    ia = InterfaceAnalyzerMover()
    ia.set_compute_packstat(True)
    ia.apply(pose)
    interface_sasa = ia.get_interface_delta_sasa()
    interface_dG = ia.get_interface_dG()
    interface_delta_hbond_unsat = ia.get_interface_delta_hbond_unsat()
    packstat = ia.get_interface_packstat()

    return {
        'interface_sasa': interface_sasa,
        'interface_dG': interface_dG,
        'rosetta_score': rosetta_score,
        'interface_delta_hbond_unsat': interface_delta_hbond_unsat,
        'packstat': packstat
    }

if __name__ == "__main__":
    pdb_file_path = "./data/1ssc.pdb"
    score = rosetta_scores_from_pdb(pdb_file_path)
    print(f"Rosetta scores {pdb_file_path}: {score}")