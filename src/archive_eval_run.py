#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description="Archive evaluation outputs with a timestamped run prefix.")
    parser.add_argument("--run-name", default="part2", help="Logical run name prefix, e.g. part2 or scene3")
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Timestamp suffix to use in archived filenames",
    )
    parser.add_argument(
        "--source-dir",
        default="result/eval",
        help="Directory containing summary.json, frame_metrics.csv, performance_summary.json",
    )
    parser.add_argument(
        "--archive-dir",
        default="result/eval/archive",
        help="Directory where timestamped copies should be stored",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    archive_dir = Path(args.archive_dir)
    prefix = f"{args.run_name}_{args.timestamp}"

    mapping = {
        source_dir / "summary.json": archive_dir / f"{prefix}_summary.json",
        source_dir / "frame_metrics.csv": archive_dir / f"{prefix}_frame_metrics.csv",
        source_dir / "performance_summary.json": archive_dir / f"{prefix}_performance_summary.json",
    }

    copied = {}
    for src, dst in mapping.items():
        copied[str(dst)] = copy_if_exists(src, dst)

    manifest_path = archive_dir / f"{prefix}_archive_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "run_name": args.run_name,
        "timestamp": args.timestamp,
        "copied_files": copied,
    }, indent=2))

    print(json.dumps({
        "archive_manifest": str(manifest_path),
        "copied_files": copied,
    }, indent=2))


if __name__ == "__main__":
    main()
