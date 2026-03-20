import glob
import json
import logging
import os
import re
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

    This backend is intentionally CLI-template driven to support different OpenFold3
    repository entrypoints. Provide `openfold3_command_template` with placeholders
    documented in `_build_openfold3_command`.
    """

    def __init__(self, **kwargs):
        self.method_name = "openfold3"
        super().__init__(**kwargs)

        self.num_models = kwargs.get("num_models", 5)
        self.num_recycles = kwargs.get("num_recycles", 10)
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

        combined_fasta_path = os.path.join(raw_sequence_dir, f"input_{sequence_sequence}.fasta")
        with open(combined_fasta_path, "w") as fasta_handle:
            fasta_handle.write(
                f">{target_name}_{sequence_sequence}\n{target_sequence}:{sequence_sequence}\n"
            )

        command = self._build_openfold3_command(
            input_fasta=combined_fasta_path,
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

    def _build_openfold3_command(
        self,
        input_fasta: str,
        output_dir: str,
        target_structure: str,
        target_sequence: str,
        sequence_sequence: str,
        target_name: str,
    ) -> List[str]:
        """
        Build OpenFold3 command from `openfold3_command_template`.

        Supported placeholders:
        - {input_fasta}
        - {output_dir}
        - {target_structure}
        - {target_sequence}
        - {sequence}
        - {target_name}
        - {num_models}
        - {num_recycles}
        """
        if not self.openfold3_command_template:
            raise ValueError(
                "'openfold3_command_template' is required for model='openfold3'. "
                "Example: \"python /path/to/openfold3/infer.py --fasta {input_fasta} "
                "--template {target_structure} --out {output_dir}\""
            )

        formatted_command = self.openfold3_command_template.format(
            input_fasta=input_fasta,
            output_dir=output_dir,
            target_structure=target_structure,
            target_sequence=target_sequence,
            sequence=sequence_sequence,
            target_name=target_name,
            num_models=self.num_models,
            num_recycles=self.num_recycles,
        )
        command = shlex.split(formatted_command)

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

        structure_files = self._find_structure_files(raw_sequence_dir)
        if not structure_files:
            raise ValueError(
                f"No OpenFold3 structure files found under raw output: {raw_sequence_dir}"
            )

        all_models_data = []
        for model_index, structure_path in enumerate(structure_files, 1):
            standardized_filename = self._standardize_model_filename(structure_path, model_index)
            destination = os.path.join(processed_dir, standardized_filename)
            shutil.copy2(structure_path, destination)

            model_metrics = {
                "pdb_file": standardized_filename,
                "model_index": model_index,
            }
            model_metrics.update(self._extract_neighbor_metrics(structure_path))
            all_models_data.append(model_metrics)

        best_model = max(all_models_data, key=self._best_model_key)

        metrics = {
            "sequence": sequence_sequence,
            "target_name": target_name,
            "docking_method": "openfold3",
            "model_count": len(all_models_data),
            "best_model_index": best_model.get("model_index", 1),
            **{k: v for k, v in best_model.items() if k != "pdb_file"},
        }

        self._save_metrics_json(metrics, processed_dir, prefix="openfold3_metrics")
        return processed_dir

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
        ]

        return sorted(filtered)

    def _extract_neighbor_metrics(self, structure_path: str) -> Dict[str, Any]:
        directory = os.path.dirname(structure_path)
        base_name = os.path.splitext(os.path.basename(structure_path))[0]
        rank_match = re.search(r"(?:rank|model)[_\-]?(\d+)", base_name.lower())
        rank_token = rank_match.group(1) if rank_match else None

        json_candidates = glob.glob(os.path.join(directory, "*.json"))
        chosen_json = None
        if rank_token:
            for candidate in json_candidates:
                candidate_name = os.path.basename(candidate).lower()
                if rank_token in candidate_name and (
                    "score" in candidate_name
                    or "metric" in candidate_name
                    or "rank" in candidate_name
                    or "result" in candidate_name
                ):
                    chosen_json = candidate
                    break

        if chosen_json is None:
            for candidate in json_candidates:
                candidate_name = os.path.basename(candidate).lower()
                if any(token in candidate_name for token in ["score", "metric", "rank", "result"]):
                    chosen_json = candidate
                    break

        if chosen_json is None:
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
            "num_recycles": self.num_recycles,
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
