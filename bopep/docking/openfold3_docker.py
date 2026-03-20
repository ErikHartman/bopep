import glob
import json
import logging
import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Tuple

from bopep.docking.base_docking_model import BaseDockingModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OpenFold3Docker(BaseDockingModel):
    """
    OpenFold3-based docking backend.

    By default this backend uses the documented OpenFold3 CLI pattern:
    `run_openfold predict --query-json ... --output-dir ...`

    If your OpenFold3 fork has a custom entrypoint, you can provide
    `openfold3_command_template` to fully override command construction.
    """

    def __init__(self, **kwargs):
        self.method_name = "openfold3"
        super().__init__(**kwargs)

        self.num_models = kwargs.get("num_models", 5)
        self.num_model_seeds = kwargs.get("num_model_seeds", 1)

        # Native run_openfold settings
        self.openfold3_binary = kwargs.get("openfold3_binary", "run_openfold")
        self.openfold3_subcommand = kwargs.get("openfold3_subcommand", "predict")
        self.openfold3_use_msa_server = kwargs.get("openfold3_use_msa_server", True)
        self.openfold3_use_templates = kwargs.get("openfold3_use_templates", True)
        self.openfold3_runner_yaml = kwargs.get("openfold3_runner_yaml")
        self.openfold3_inference_ckpt_path = kwargs.get("openfold3_inference_ckpt_path")
        self.openfold3_inference_ckpt_name = kwargs.get("openfold3_inference_ckpt_name")

        # Optional custom JSON for advanced users
        self.openfold3_query_builder = kwargs.get("openfold3_query_builder", "default")

        # Full command override for custom forks
        self.openfold3_command_template = kwargs.get("openfold3_command_template")
        self.openfold3_extra_args = kwargs.get("openfold3_extra_args", [])

    def dock(
        self,
        sequences: List[str],
        target_structure: str,
        target_sequence: str,
        target_name: str,
    ) -> List[str]:
        return self._dock_with_common_logic(
            sequences, target_structure, target_sequence, target_name
        )

    def _dock_single_sequence(
        self,
        sequence_sequence: str,
        target_structure: str,
        target_sequence: str,
        target_name: str,
        gpu_id: str = "0",
    ) -> str:
        logging.info(f"Docking sequence '{sequence_sequence}' on GPU {gpu_id}...")

        raw_sequence_dir = self._create_raw_sequence_dir(target_name, sequence_sequence)
        query_json_path = self._create_query_json(
            raw_sequence_dir=raw_sequence_dir,
            target_name=target_name,
            sequence_sequence=sequence_sequence,
            target_sequence=target_sequence,
        )

        command = self._build_openfold3_command(
            query_json_path=query_json_path,
            output_dir=raw_sequence_dir,
            target_structure=target_structure,
            target_sequence=target_sequence,
            sequence_sequence=sequence_sequence,
            target_name=target_name,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.pop("MPLBACKEND", None)

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            output, _ = process.communicate()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=process.returncode, cmd=command, output=output
                )

            with open(os.path.join(raw_sequence_dir, "finished.txt"), "w") as finished_file:
                finished_file.write("Docking finished successfully.")
        except subprocess.CalledProcessError as error:
            logging.error(f"OpenFold3 docking failed for {sequence_sequence}: {error}")
            if getattr(error, "output", None):
                logging.error(f"OpenFold3 output:\n{error.output}")
            return None

        return raw_sequence_dir

    def _create_query_json(
        self,
        raw_sequence_dir: str,
        target_name: str,
        sequence_sequence: str,
        target_sequence: str,
    ) -> str:
        query_name = f"{target_name}_{sequence_sequence}"

        if self.openfold3_query_builder == "default":
            # Minimal protein-protein complex query compatible with
            # openfold3.projects.of3_all_atom.config.inference_query_format.InferenceQuerySet
            # Chain A = target, Chain B = designed sequence.
            query_payload: Any = {
                "queries": {
                    query_name: {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": "A",
                                "sequence": target_sequence,
                            },
                            {
                                "molecule_type": "protein",
                                "chain_ids": "B",
                                "sequence": sequence_sequence,
                            },
                        ]
                    }
                }
            }
        else:
            raise ValueError(
                f"Unsupported openfold3_query_builder='{self.openfold3_query_builder}'. "
                "Currently supported: ['default']"
            )

        query_json_path = os.path.join(raw_sequence_dir, "query.json")
        with open(query_json_path, "w") as query_file:
            json.dump(query_payload, query_file, indent=2)

        return query_json_path

    def _build_openfold3_command(
        self,
        query_json_path: str,
        output_dir: str,
        target_structure: str,
        target_sequence: str,
        sequence_sequence: str,
        target_name: str,
    ) -> List[str]:
        """
        Build OpenFold3 command from `openfold3_command_template`.

        Supported placeholders:
        - {query_json}
        - {output_dir}
        - {target_structure}
        - {target_sequence}
        - {sequence}
        - {target_name}
        - {num_models}
        - {num_model_seeds}
        """
        if self.openfold3_command_template:
            formatted_command = self.openfold3_command_template.format(
                query_json=query_json_path,
                output_dir=output_dir,
                target_structure=target_structure,
                target_sequence=target_sequence,
                sequence=sequence_sequence,
                target_name=target_name,
                num_models=self.num_models,
                num_model_seeds=self.num_model_seeds,
            )
            command = shlex.split(formatted_command)
        else:
            command = [
                self.openfold3_binary,
                self.openfold3_subcommand,
                "--query-json",
                query_json_path,
                "--output-dir",
                output_dir,
                "--num-diffusion-samples",
                str(self.num_models),
                "--num-model-seeds",
                str(self.num_model_seeds),
            ]

            if self.openfold3_use_msa_server:
                command.append("--use-msa-server")
            else:
                command.append("--use-msa-server=False")

            if self.openfold3_use_templates:
                command.append("--use-templates")
            else:
                command.append("--use-templates=False")

            if self.openfold3_runner_yaml:
                command.extend(["--runner-yaml", str(self.openfold3_runner_yaml)])
            if self.openfold3_inference_ckpt_path:
                command.extend(
                    ["--inference-ckpt-path", str(self.openfold3_inference_ckpt_path)]
                )
            if self.openfold3_inference_ckpt_name:
                command.extend(
                    ["--inference-ckpt-name", str(self.openfold3_inference_ckpt_name)]
                )

        if self.openfold3_extra_args:
            if isinstance(self.openfold3_extra_args, list):
                command.extend([str(item) for item in self.openfold3_extra_args])
            else:
                command.extend(shlex.split(str(self.openfold3_extra_args)))

        return command

    def process_raw_output(
        self, raw_sequence_dir: str, sequence_sequence: str, target_name: str
    ) -> str:
        processed_dir = self._create_processed_sequence_dir(target_name, sequence_sequence)

        sample_entries = self._collect_sample_entries(raw_sequence_dir)
        if not sample_entries:
            raise ValueError(
                f"No OpenFold3 structure files found under raw output: {raw_sequence_dir}"
            )

        sorted_entries = sorted(sample_entries, key=self._sample_sort_key, reverse=True)

        all_models_data = []
        for model_index, entry in enumerate(sorted_entries, 1):
            structure_path = entry["structure_path"]
            standardized_filename = self._standardize_model_filename(structure_path, model_index)
            destination = os.path.join(processed_dir, standardized_filename)
            shutil.copy2(structure_path, destination)

            model_metrics = {
                "pdb_file": standardized_filename,
                "model_index": model_index,
                "source_structure": os.path.relpath(structure_path, raw_sequence_dir),
            }
            model_metrics.update(entry.get("metrics", {}))
            all_models_data.append(model_metrics)

        best_model = all_models_data[0]

        metrics = {
            "sequence": sequence_sequence,
            "target_name": target_name,
            "docking_method": "openfold3",
            "model_count": len(all_models_data),
            "best_model_index": best_model.get("model_index", 1),
            "all_models": all_models_data,
            **{k: v for k, v in best_model.items() if k != "pdb_file"},
        }

        self._save_metrics_json(metrics, processed_dir, prefix="openfold3_metrics")
        return processed_dir

    def _collect_sample_entries(self, raw_sequence_dir: str) -> List[Dict[str, Any]]:
        structure_files = self._find_structure_files(raw_sequence_dir)
        sample_entries: List[Dict[str, Any]] = []

        for structure_path in structure_files:
            metrics = self._read_confidence_metrics_for_structure(structure_path)
            sample_entries.append(
                {
                    "structure_path": structure_path,
                    "metrics": metrics,
                }
            )

        return sample_entries

    @staticmethod
    def _sample_sort_key(entry: Dict[str, Any]) -> Tuple[float, float, float, float]:
        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        score = OpenFold3Docker._as_float(metrics.get("sample_ranking_score"))
        iptm = OpenFold3Docker._as_float(metrics.get("iptm"))
        ptm = OpenFold3Docker._as_float(metrics.get("ptm"))
        plddt = OpenFold3Docker._as_float(metrics.get("avg_plddt"))
        return (score, iptm, ptm, plddt)

    @staticmethod
    def _as_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float("-inf")

    @staticmethod
    def _best_model_key(model_metrics: Dict[str, Any]) -> float:
        for key in ["iptm", "ptm", "confidence", "mean_plddt", "plddt"]:
            value = model_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list) and value:
                numeric_values = [v for v in value if isinstance(v, (int, float))]
                if numeric_values:
                    return float(sum(numeric_values) / len(numeric_values))
        return 0.0

    def _find_structure_files(self, raw_sequence_dir: str) -> List[str]:
        candidates = []
        for extension in ("*.pdb", "*.cif"):
            candidates.extend(
                glob.glob(os.path.join(raw_sequence_dir, "**", extension), recursive=True)
            )

        filtered = [
            file_path
            for file_path in candidates
            if "template" not in os.path.basename(file_path).lower()
            and os.path.basename(file_path).lower().endswith(("_model.pdb", "_model.cif"))
        ]

        return sorted(filtered)

    def _read_confidence_metrics_for_structure(self, structure_path: str) -> Dict[str, Any]:
        directory = os.path.dirname(structure_path)
        file_name = os.path.basename(structure_path)
        if file_name.endswith("_model.cif"):
            stem = file_name[: -len("_model.cif")]
        elif file_name.endswith("_model.pdb"):
            stem = file_name[: -len("_model.pdb")]
        else:
            stem = os.path.splitext(file_name)[0]

        aggregated_path = os.path.join(directory, f"{stem}_confidences_aggregated.json")
        confidence_path = os.path.join(directory, f"{stem}_confidences.json")

        chosen_json = None
        if os.path.exists(aggregated_path):
            chosen_json = aggregated_path
        elif os.path.exists(confidence_path):
            chosen_json = confidence_path
        else:
            json_candidates = sorted(glob.glob(os.path.join(directory, "*.json")))
            for candidate in json_candidates:
                candidate_name = os.path.basename(candidate).lower()
                if "confidences_aggregated" in candidate_name:
                    chosen_json = candidate
                    break
            if chosen_json is None:
                for candidate in json_candidates:
                    candidate_name = os.path.basename(candidate).lower()
                    if "confidence" in candidate_name:
                        chosen_json = candidate
                        break

        if not chosen_json:
            return {}

        try:
            with open(chosen_json, "r") as json_handle:
                data = json.load(json_handle)
        except Exception as error:
            logging.warning(f"Failed reading OpenFold3 metrics JSON {chosen_json}: {error}")
            return {}

        if isinstance(data, dict):
            return data
        return {"raw_metrics": data}

    def _get_method_parameters(self) -> dict:
        return {
            "num_models": self.num_models,
            "num_model_seeds": self.num_model_seeds,
            "openfold3_binary": self.openfold3_binary,
            "openfold3_subcommand": self.openfold3_subcommand,
            "openfold3_use_msa_server": self.openfold3_use_msa_server,
            "openfold3_use_templates": self.openfold3_use_templates,
            "openfold3_runner_yaml": self.openfold3_runner_yaml,
            "openfold3_inference_ckpt_path": self.openfold3_inference_ckpt_path,
            "openfold3_inference_ckpt_name": self.openfold3_inference_ckpt_name,
            "openfold3_query_builder": self.openfold3_query_builder,
            "openfold3_command_template": self.openfold3_command_template,
            "openfold3_extra_args": self.openfold3_extra_args,
            "save_raw": self.save_raw,
            "overwrite_results": self.overwrite_results,
        }

    @staticmethod
    def _dock_sequences_for_gpu(
        sequences: List[str],
        gpu_id: str,
        target_structure: str,
        target_sequence: str,
        target_name: str,
        raw_output_dir: str,
        method_params: dict,
    ) -> List[Tuple[str, str]]:
        output_dir = os.path.dirname(os.path.dirname(raw_output_dir))

        temp_docker = OpenFold3Docker(
            output_dir=output_dir,
            gpu_ids=[gpu_id],
            **method_params,
        )

        docked_results = []
        for i, sequence in enumerate(sequences, 1):
            print(f"GPU {gpu_id} progress: {i}/{len(sequences)} - docking {sequence}")
            raw_dir = temp_docker._dock_single_sequence(
                sequence, target_structure, target_sequence, target_name, gpu_id
            )
            if raw_dir:
                docked_results.append((sequence, raw_dir))

        return docked_results
