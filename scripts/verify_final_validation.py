#!/usr/bin/env python3
"""Verify exported results independently; require full empirical gates to promote."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.final_validation.io import write_json  # noqa: E402

REQUIRED = [
    "evidence",
    "nsl_kdd_data",
    "nsl_kdd_stability",
    "nsl_kdd_development",
    "nsl_kdd_replay",
    "cicids2017_data",
    "cicids2017_stability",
    "cicids2017_development",
    "cicids2017_replay",
    "benchmark",
    "sample120k",
    "report",
]


def verify_export(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe ZIP member")
        manifest = json.loads(archive.read("EXPORT_MANIFEST.json"))
        expected = set(manifest["files"]) | {"EXPORT_MANIFEST.json"}
        if set(names) != expected:
            raise ValueError("Export inventory mismatch")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Output hash mismatch: " + name)
        state = json.loads(archive.read("status.json"))
        statuses = {r["job"]: r["status"] for r in state["jobs"]}
        full = (
            json.loads(archive.read("execution_identity.json")).get("profile") == "full"
        )
        missing = [key for key in REQUIRED if statuses.get(key) != "VERIFIED_REAL_DATA"]
        if "scientific_qa.json" in names:
            qa = json.loads(archive.read("scientific_qa.json"))
            if qa.get("counts", {}).get("FAIL", 0):
                raise ValueError("Scientific numeric QA contains failures")
            if full and (
                qa.get("profile") != "full" or qa.get("counts", {}).get("PENDING", 0)
            ):
                missing.append("scientific_qa_full_completion")
        else:
            missing.append("scientific_qa_missing")
        excluded = set(manifest["excluded_private_files"])
        # Independent reconciliation of every nested successful-job manifest.
        for name in names:
            if not name.endswith("/manifest.json"):
                continue
            job = json.loads(archive.read(name))
            if not job["status"].startswith("VERIFIED_"):
                continue
            for rel, digest in job["outputs"].items():
                item = (Path(name).parent / rel).as_posix()
                if item in excluded:
                    continue
                if (
                    item not in names
                    or hashlib.sha256(archive.read(item)).hexdigest() != digest
                ):
                    raise ValueError("Job output missing/hash mismatch: " + item)
        return {
            "export_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "zip_hashes_pass": True,
            "full_profile": full,
            "required_real_data_jobs_missing": missing,
            "eligible_for_independent_final_review": full and not missing,
            "source_identity": json.loads(archive.read("execution_identity.json")),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--promote-after-independent-review",
        action="store_true",
        help="Use only in subsequent independent review after Sarosh uploads results",
    )
    args = parser.parse_args()
    review = verify_export(args.export)
    review["handoff_status"] = "PRELIMINARY"
    if args.promote_after_independent_review:
        if not review["eligible_for_independent_final_review"]:
            raise SystemExit("Cannot promote: missing full real-data gates")
        review["handoff_status"] = "VERIFIED_FINAL_AFTER_INDEPENDENT_REVIEW"
    write_json(args.output, review)
    print(
        json.dumps(
            {k: v for k, v in review.items() if k != "source_identity"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
