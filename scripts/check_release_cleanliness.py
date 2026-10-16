#!/usr/bin/env python3
"""Check publishable files without inspecting ignored private experiment folders."""

from pathlib import Path
import argparse
import json
import re
import subprocess

BLOCKED_NAMES = {
    "train_row_signatures.csv",
    "validation_row_signatures.csv",
    "test_row_signatures.csv",
    "sample_row_provenance.csv",
}
BLOCKED_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".pcap",
    ".pcapng",
    ".parquet",
}


def inspect_text(text, relative):
    issues = []
    if chr(0x2014) in text or ("\\" + "u" + format(0x2014, "04x")) in text.lower():
        issues.append("forbidden punctuation")
    if re.search(
        r"/workspace/"
        + r"scratch/|/Users/[A-Za-z]|C:[\\\\/]Users[\\\\/]|/home/(?!runner(?:/|$))[A-Za-z][^ /]*/",
        text,
    ):
        issues.append("machine-specific user path")
    if re.search(
        r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16})\b",
        text,
    ):
        issues.append("credential pattern")
    if relative.endswith(".ipynb"):
        obj = json.loads(text)
        for cell in obj.get("cells", []):
            if cell.get("outputs") or cell.get("execution_count") is not None:
                issues.append("executed notebook output")
            decoded = "".join(cell.get("source", []))
            if chr(0x2014) in decoded:
                issues.append("forbidden punctuation in notebook")
    return sorted(set(issues))


def inspect_files(root, paths):
    root = Path(root)
    failures = []
    total = 0
    for rel in sorted(set(paths)):
        path = root / rel
        if not path.is_file():
            continue
        total += 1
        p = Path(rel)
        issues = []
        if p.name in BLOCKED_NAMES or p.suffix.lower() in BLOCKED_SUFFIXES:
            issues.append("private artifact type")
        if p.parts[0] in {"results_final_revision", "private", ".venv", "venv"} or any(
            c in p.parts for c in ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]
        ):
            issues.append("private or temporary directory")
        if path.stat().st_size > 10 * 1024**2:
            issues.append("file exceeds 10 MiB release budget")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            text = None
        if text is not None:
            issues.extend(inspect_text(text, rel))
        if issues:
            failures.append({"path": rel, "issues": sorted(set(issues))})
    return {
        "status": "PASS" if not failures else "FAIL",
        "files_checked": total,
        "failures": failures,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=args.root,
        text=True,
    ).splitlines()
    result = inspect_files(args.root, paths)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
