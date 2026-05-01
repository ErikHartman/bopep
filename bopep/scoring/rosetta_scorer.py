import logging
import os
from pathlib import Path
from typing import Optional


class RosettaScorer:
    _METRICS_CACHE = {}
    _FASTRELAX_MOVER_CACHE = {}
    _PYROSETTA_READY = False

    def __init__(
        self,
        structure_file: str,
        fastrelax_before_scoring: bool | None = None,
        fastrelax_xml_path: str | None = None,
        use_metrics_cache: bool = True,
    ):
        self.structure_file = structure_file
        self.initialized = False
        self.scorefxn = None
        self.pose = None
        self.ia = None
        self.rosetta_score = None
        self.interface_sasa = None
        self.interface_dg = None
        self.interface_delta_hbond_unsat = None
        self.packstat = None

        self.fastrelax_before_scoring = self._resolve_fastrelax_enabled(fastrelax_before_scoring)
        self.fastrelax_xml_path = self._resolve_fastrelax_xml_path(fastrelax_xml_path)
        self.use_metrics_cache = bool(use_metrics_cache)
        self._fastrelax_mover = None

    @staticmethod
    def _env_true(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _resolve_fastrelax_enabled(self, requested: bool | None) -> bool:
        if requested is not None:
            return bool(requested)
        return self._env_true("SCORE_ROSETTA_FASTRELAX", default=False)

    def _resolve_fastrelax_xml_path(self, requested: str | None) -> str:
        if requested:
            return requested

        env_xml = os.environ.get("SCORE_ROSETTA_FASTRELAX_XML", "").strip()
        if env_xml:
            return env_xml

        # Default to BoPep's bundled RosettaScripts XML.
        return str(
            Path(__file__).resolve().parents[1] / "diffusion" / "rosetta" / "RosettaFastRelaxUtil.xml"
        )

    @staticmethod
    def get_relaxed_structure_path(structure_file: str, suffix: str = "_relaxed") -> str:
        """Return sibling path used for persisted relaxed structures."""
        p = Path(structure_file)
        out_suffix = p.suffix
        # PyRosetta persist path uses dump_pdb, so non-PDB inputs should emit .pdb outputs.
        if out_suffix.lower() != ".pdb":
            out_suffix = ".pdb"
        return str(p.with_name(f"{p.stem}{suffix}{out_suffix}"))

    @classmethod
    def choose_structure_for_rosetta(
        cls,
        structure_file: str,
        prefer_relaxed: bool = False,
        relaxed_suffix: str = "_relaxed",
    ) -> str:
        """
        Return relaxed sibling structure when requested and present, else original path.
        """
        if not prefer_relaxed:
            return structure_file
        primary = cls.get_relaxed_structure_path(structure_file, suffix=relaxed_suffix)
        candidates = [primary]
        # Back-compat: if an earlier run wrote *_relaxed with original suffix, still accept it.
        p = Path(structure_file)
        legacy = str(p.with_name(f"{p.stem}{relaxed_suffix}{p.suffix}"))
        if legacy not in candidates:
            candidates.append(legacy)
        for relaxed_path in candidates:
            if os.path.exists(relaxed_path):
                return relaxed_path
        return structure_file

    def _default_init_flags(self) -> str:
        if self.fastrelax_before_scoring:
            # Required by RosettaFastRelaxUtil.xml (uses beta_nov16 scorefunctions).
            return (
                "-beta_nov16 "
                "-corrections::beta_nov16 true "
                "-in:file:silent_struct_type binary "
                "-use_terminal_residues true "
                "-mute all"
            )
        return "-mute all"

    def _ensure_pyrosetta_initialized(self, pyrosetta):
        if RosettaScorer._PYROSETTA_READY:
            return
        # Shared process marker across modules (e.g., scoring script worker init)
        # to avoid repeated expensive init calls in the same worker process.
        if bool(getattr(pyrosetta, "_stat0540_pyrosetta_inited", False)):
            RosettaScorer._PYROSETTA_READY = True
            return

        # Optional strict mode for scoring pipelines that pre-initialize PyRosetta
        # once in worker startup. If this trips, initialization ordering regressed.
        if self._env_true("SCORE_REQUIRE_PREINIT", default=False):
            raise RuntimeError(
                "SCORE_REQUIRE_PREINIT=1 but PyRosetta was not pre-initialized in this worker. "
                "Expected worker initializer to call _ensure_pyrosetta_initialized() exactly once."
            )

        already_init = False
        try:
            already_init = bool(pyrosetta.rosetta.basic.was_init())
        except Exception:
            already_init = False

        if not already_init:
            init_flags = os.environ.get("SCORE_PYROSETTA_INIT_FLAGS", "").strip()
            if not init_flags:
                init_flags = self._default_init_flags()
            logging.info(
                "pyrosetta_init pid=%s fastrelax=%s flags=%s",
                os.getpid(),
                self.fastrelax_before_scoring,
                init_flags,
            )
            pyrosetta.init(init_flags)
        setattr(pyrosetta, "_stat0540_pyrosetta_inited", True)
        RosettaScorer._PYROSETTA_READY = True

    def _build_fastrelax_mover(self):
        if not self.fastrelax_before_scoring:
            return None

        xml_path = Path(self.fastrelax_xml_path).resolve()
        if not xml_path.exists():
            raise FileNotFoundError(
                f"FastRelax requested but XML not found: {xml_path}. "
                "Set SCORE_ROSETTA_FASTRELAX_XML to a valid RosettaScripts XML."
            )

        cached_mover = self._FASTRELAX_MOVER_CACHE.get(str(xml_path))
        if cached_mover is not None:
            return cached_mover

        objs = self.pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file(str(xml_path))
        mover = objs.get_mover("FastRelax")
        if mover is None:
            raise RuntimeError(f"FastRelax mover named 'FastRelax' not found in XML: {xml_path}")
        self._FASTRELAX_MOVER_CACHE[str(xml_path)] = mover
        return mover

    def _initialize(self):
        if not self.initialized:
            try:
                import pyrosetta
                from pyrosetta.io import pose_from_pdb
                from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

                self.pyrosetta = pyrosetta
                self.pose_from_pdb = pose_from_pdb
                self.InterfaceAnalyzerMover = InterfaceAnalyzerMover

                cache_key = (
                    str(Path(self.structure_file).resolve()),
                    bool(self.fastrelax_before_scoring),
                    str(Path(self.fastrelax_xml_path).resolve()) if self.fastrelax_before_scoring else "",
                )
                if self.use_metrics_cache:
                    cached = self._METRICS_CACHE.get(cache_key)
                    if cached is not None:
                        self.rosetta_score = cached["rosetta_score"]
                        self.interface_sasa = cached["interface_sasa"]
                        self.interface_dg = cached["interface_dg"]
                        self.interface_delta_hbond_unsat = cached["interface_delta_hbond_unsat"]
                        self.packstat = cached["packstat"]
                        self.initialized = True
                        return

                self._ensure_pyrosetta_initialized(pyrosetta)
                self.scorefxn = pyrosetta.get_fa_scorefxn()
                self.pose = pose_from_pdb(self.structure_file)

                if self.fastrelax_before_scoring:
                    self._fastrelax_mover = self._build_fastrelax_mover()
                    self._fastrelax_mover.apply(self.pose)

                self.rosetta_score = self.scorefxn(self.pose)
                self.ia = InterfaceAnalyzerMover()
                self.ia.set_compute_packstat(True)
                self.ia.apply(self.pose)
                self.interface_sasa = self.ia.get_interface_delta_sasa()
                self.interface_dg = self.ia.get_interface_dG()
                self.interface_delta_hbond_unsat = self.ia.get_interface_delta_hbond_unsat()
                self.packstat = self.ia.get_interface_packstat()
                if self.use_metrics_cache:
                    self._METRICS_CACHE[cache_key] = {
                        "rosetta_score": self.rosetta_score,
                        "interface_sasa": self.interface_sasa,
                        "interface_dg": self.interface_dg,
                        "interface_delta_hbond_unsat": self.interface_delta_hbond_unsat,
                        "packstat": self.packstat,
                    }
                self.initialized = True
            except ImportError as e:
                raise ImportError(f"PyRosetta is required for RosettaScorer but not installed: {e}")
            except Exception as e:
                logging.exception("RosettaScorer failed for structure_file=%s", self.structure_file)
                raise RuntimeError(f"Rosetta scoring failed for {self.structure_file}: {e}") from e

    def get_rosetta_score(self):
        if not self.initialized:
            self._initialize()
        return self.rosetta_score

    def get_interface_sasa(self):
        if not self.initialized:
            self._initialize()
        return self.interface_sasa

    def get_interface_dG(self):
        if not self.initialized:
            self._initialize()
        return self.interface_dg

    def get_interface_delta_hbond_unsat(self):
        if not self.initialized:
            self._initialize()
        return self.interface_delta_hbond_unsat

    def get_packstat(self):
        if not self.initialized:
            self._initialize()
        return self.packstat

    def get_all_metrics(self):
        if not self.initialized:
            self._initialize()
        return {
            "rosetta_score": self.get_rosetta_score(),
            "interface_sasa": self.get_interface_sasa(),
            "interface_dG": self.get_interface_dG(),
            "interface_delta_hbond_unsat": self.get_interface_delta_hbond_unsat(),
            "packstat": self.get_packstat(),
        }


def _collect_structure_files(input_dir: str, recursive: bool = True) -> list[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    patterns = ("*.pdb", "*.cif", "*.mmcif", "*.pdbx")
    out = []
    if recursive:
        for patt in patterns:
            out.extend(root.rglob(patt))
    else:
        for patt in patterns:
            out.extend(root.glob(patt))
    return sorted(str(p) for p in out)


def relax_structure_file(
    input_structure_file: str,
    output_structure_file: Optional[str] = None,
    fastrelax_xml_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Run Rosetta FastRelax once and persist a relaxed structure file.
    """
    out_file = output_structure_file or RosettaScorer.get_relaxed_structure_path(input_structure_file)
    if os.path.exists(out_file) and not overwrite:
        return out_file
    scorer = RosettaScorer(
        input_structure_file,
        fastrelax_before_scoring=True,
        fastrelax_xml_path=fastrelax_xml_path,
        use_metrics_cache=False,
    )
    scorer._initialize()
    scorer.pose.dump_pdb(out_file)
    return out_file


def relax_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = True,
    fastrelax_xml_path: Optional[str] = None,
    overwrite: bool = False,
    skip_relaxed_inputs: bool = True,
) -> dict:
    """
    Relax all structures in a directory and write relaxed outputs.
    """
    structure_files = _collect_structure_files(input_dir=input_dir, recursive=recursive)
    processed = 0
    skipped = 0
    failed = 0
    outputs = []
    for structure_file in structure_files:
        if skip_relaxed_inputs and Path(structure_file).stem.endswith("_relaxed"):
            skipped += 1
            continue
        try:
            if output_dir:
                in_path = Path(structure_file).resolve()
                rel = in_path.relative_to(Path(input_dir).resolve())
                out_base = Path(output_dir).resolve() / rel
                out_base.parent.mkdir(parents=True, exist_ok=True)
                output_path = RosettaScorer.get_relaxed_structure_path(str(out_base))
            else:
                output_path = None
            relaxed = relax_structure_file(
                input_structure_file=structure_file,
                output_structure_file=output_path,
                fastrelax_xml_path=fastrelax_xml_path,
                overwrite=overwrite,
            )
            outputs.append(relaxed)
            processed += 1
        except Exception:
            logging.exception("Failed to relax structure: %s", structure_file)
            failed += 1
    return {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "outputs": outputs,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Rosetta scoring and FastRelax utilities.")
    parser.add_argument("--structure-file", type=str, default=None, help="Single input structure file (.pdb/.cif).")
    parser.add_argument("--relax-dir", type=str, default=None, help="Relax all structures in a directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for --relax-dir.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover structures in --relax-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing relaxed outputs.")
    parser.add_argument("--fastrelax-xml-path", type=str, default=None, help="Optional RosettaScripts XML path.")
    args = parser.parse_args()

    if args.relax_dir:
        summary = relax_directory(
            input_dir=args.relax_dir,
            output_dir=args.output_dir,
            recursive=args.recursive,
            fastrelax_xml_path=args.fastrelax_xml_path,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
    else:
        pdb_file_path = args.structure_file or "../../data/1ssc.pdb"
        analyzer = RosettaScorer(pdb_file_path)
        metrics = analyzer.get_all_metrics()
        print(f"Rosetta metrics for {pdb_file_path}: {metrics}")
