# demo_boga.py

import logging
import random
import numpy as np
import tempfile
import os
from typing import List, Dict, Any
from bopep import BoGA  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def demo_boga_genetic_algorithm():
    """
    Demonstrates how to instantiate and run BoGA with all heavy steps mocked out.
    """

    # 1) Subclass BoGA to override embedding, docking, scoring and HPO/model init
    class MockBoGA(BoGA):
        def __init__(self, *args, **kwargs):
            # make the demo tiny
            kwargs.setdefault("n_init", 5)
            kwargs.setdefault("m_select", 2)
            kwargs.setdefault("k_pool", 10)
            kwargs.setdefault("generations", 3)
            kwargs.setdefault("hpo_interval", 2)
            
            # Required docker_kwargs with output_dir
            kwargs.setdefault("docker_kwargs", {
                "output_dir": tempfile.mkdtemp(),
                "num_models": 1,
                "num_recycles": 1,
                "amber": False,
                "num_relax": 0
            })
            
            # Required scoring_kwargs
            kwargs.setdefault("scoring_kwargs", {
                "scores_to_include": ["distance_score"],
                "n_jobs": 1
            })
            
            # minimal surrogate kwargs
            kwargs.setdefault("surrogate_model_kwargs", {
                "network_type": "mlp",
                "model_type": "nn_ensemble",
                # HPO parameters (won't actually run)
                "n_trials": 1,
                "n_splits": 1,
                "random_state": 42,
            })
            super().__init__(*args, **kwargs)

        def _embed(self, peptides: List[str]) -> Dict[str, Any]:
            # return a fixed‐length random vector per peptide
            return {pep: np.random.rand(16).astype(np.float32) for pep in peptides}

        def _dock_and_score(self, sequences: List[str]) -> Dict[str, Dict[str, float]]:
            # Return the expected format: {peptide: {score_name: value, ...}}
            return {
                seq: {
                    "distance_score": random.random(),
                    "in_binding_site": random.choice([True, False])
                } 
                for seq in sequences
            }

        def _optimize_hyperparameters(self, embeddings, objectives):
            # skip real tuning: just pick trivial hyperparams
            self.best_hyperparams = {
                "input_dim": next(iter(embeddings.values())).shape[-1],
                "hidden_dims": [8],
                "hidden_dim": 8,
                "num_layers": 1,
                "uncertainty_param": 2,  # For nn_ensemble this is n_networks
            }

        def _initialize_model(self, hyperparams: Dict[str, Any]):
            # dummy surrogate: fit does nothing, predict returns random scores
            class DummyModel:
                def fit_dict(self, embs, objs, device=None): 
                    pass
                def predict_dict(self, embs, device=None):
                    return {pep: random.random() for pep in embs}
                def to(self, device):
                    pass  # Mock the to method for device placement
            self.model = DummyModel()

    # Create a temporary directory for outputs
    temp_output_dir = tempfile.mkdtemp()
    print(f"Using temporary output directory: {temp_output_dir}")

    # 2) Instantiate MockBoGA
    boga = MockBoGA(
        target_structure_path="./data/1ssc.pdb",  # will not actually be used
        sequence_length=8,
        embed_method="aaindex",   # doesn't matter, _embed is overridden
        embed_average=True,
    )

    # 3) Run the GA
    logging.info("Starting mock BoGA run…")
    final_scores = boga.run()

    # 4) Summarize results
    top3 = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print("\nTop 3 peptides after mock evolution:")
    for seq, score in top3:
        print(f"  {seq} → score {score:.3f}")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_output_dir, ignore_errors=True)
    print(f"\nCleaned up temporary directory: {temp_output_dir}")


if __name__ == "__main__":
    demo_boga_genetic_algorithm()
