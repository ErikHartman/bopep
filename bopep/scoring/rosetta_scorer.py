import logging
import os
from pathlib import Path


class RosettaScorer:
    _METRICS_CACHE = {}
    _FASTRELAX_MOVER_CACHE = {}
    _PYROSETTA_READY = False

    def __init__(
        self,
        structure_file: str,
        fastrelax_before_scoring: bool | None = None,
        fastrelax_xml_path: str | None = None,
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


if __name__ == "__main__":
    pdb_file_path = "../../data/1ssc.pdb"
    analyzer = RosettaScorer(pdb_file_path)
    metrics = analyzer.get_all_metrics()
    print(f"Rosetta metrics for {pdb_file_path}: {metrics}")
