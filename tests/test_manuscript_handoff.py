from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.confirmatory_baselines import load_manuscript_config
from src.manuscript_analysis import build_writer_handoff


class ManuscriptHandoffTests(unittest.TestCase):
    def test_handoff_excludes_raw_data_and_contains_writer_map(self) -> None:
        config = load_manuscript_config("configs/manuscript_readiness.yaml", profile_name="smoke")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            (analysis / "publication_tables").mkdir(parents=True)
            (analysis / "latex_tables").mkdir()
            (analysis / "publication_figures").mkdir()
            pd.DataFrame([{"passed": True}]).to_csv(
                analysis / "publication_tables" / "final_readiness_audit.csv", index=False
            )
            (analysis / "analysis_manifest.json").write_text(
                json.dumps({"audit_passed": True}), encoding="utf-8"
            )

            confirmatory = root / "confirmatory"
            baseline = root / "baseline"
            for dataset in config["baseline"]["datasets"]:
                manifests = confirmatory / dataset / "manifests"
                manifests.mkdir(parents=True)
                for name in ["execution_manifest.json", "configuration_lock.json", "execution_state.json"]:
                    (manifests / name).write_text("{}", encoding="utf-8")
                base_manifest = baseline / "confirmatory_baselines" / dataset / "manifests"
                base_manifest.mkdir(parents=True)
                (base_manifest / "execution_manifest.json").write_text("{}", encoding="utf-8")

            output = root / "handoff"
            built = build_writer_handoff(
                analysis_root=analysis,
                confirmatory_root=confirmatory,
                baseline_root=baseline,
                output_dir=output,
                repository_root=Path.cwd(),
                manuscript_config=config,
            )
            self.assertTrue((built / "00_START_HERE.md").exists())
            self.assertTrue((built / "03_CONTRIBUTIONS_AND_NOVELTY.md").exists())
            self.assertTrue((built / "PACKAGE_MANIFEST.json").exists())
            self.assertFalse((built / "data").exists())


if __name__ == "__main__":
    unittest.main()
