import json
import os
import shutil

from bopep.docking.docker import Docker


out_dir = "/tmp/bopep_openfold_smoke"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

mock_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "mock_run_openfold.py"))


docker = Docker(
    {
        "models": ["openfold3"],
        "output_dir": out_dir,
        "gpu_ids": ["0"],
        "save_raw": True,
        "num_models": 2,
        "num_model_seeds": 1,
        "openfold3_binary": mock_bin,
        "openfold3_subcommand": "predict",
        "openfold3_use_msa_server": False,
        "openfold3_use_templates": False,
    }
)

docker.set_target_structure("data/5CR6.cif")
result_dirs = docker.dock_sequences(["ACDEFGHIK"])

metrics_path = os.path.join(result_dirs[0], "openfold3_metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)

result = {
    "result_dirs": result_dirs,
    "metrics_path": metrics_path,
    "best_score": metrics.get("sample_ranking_score"),
    "best_iptm": metrics.get("iptm"),
    "model_count": metrics.get("model_count"),
    "best_model_index": metrics.get("best_model_index"),
    "all_models_len": len(metrics.get("all_models", [])),
}

result_file = "/tmp/bopep_openfold_smoke_result.json"
with open(result_file, "w") as f:
    json.dump(result, f, indent=2)

print(result_file)
