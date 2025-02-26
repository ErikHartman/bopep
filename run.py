from bopep import BoPep
import pandas as pd

if __name__ == "__main__":

    surrogate_model_kwargs = {"model_type": "nn_ensemble"}
    objective_weights = {}
    embedding_function = "esm"
    docker_kwargs = {}
    log_dir = ""

    bopep = BoPep()

    peptides = []
    target_structure_path = ""
    batch_size = 4
    schedule = {}
    
    bopep.optimize()