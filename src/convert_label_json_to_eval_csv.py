#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_timestamp(stem: str):
    if "." not in stem:
        raise ValueError(f"filename stem must be '<stamp_sec>.<stamp_nsec>', got: {stem}")
    stamp_sec, stamp_nsec = stem.split(".", 1)
    return int(stamp_sec), int(stamp_nsec)


def convert_label_dir(label_dir: Path, output_csv: Path):
    fieldnames = [
        "stamp_sec",
        "stamp_nsec",
        "frame",
        "gt_id",
        "class_name",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "heading",
        "visible_ratio",
        "source_file",
    ]
    rows = []

    for json_path in sorted(label_dir.glob("*.json")):
        stamp_sec, stamp_nsec = parse_timestamp(json_path.stem)
        with json_path.open("r") as f:
            objects = json.load(f)

        for obj in objects:
            psr = obj.get("psr", {})
            position = psr.get("position", {})
            rotation = psr.get("rotation", {})
            scale = psr.get("scale", {})
            rows.append({
                "stamp_sec": stamp_sec,
                "stamp_nsec": stamp_nsec,
                "frame": "",
                "gt_id": obj.get("obj_id", ""),
                "class_name": obj.get("obj_type", ""),
                "x": position.get("x", ""),
                "y": position.get("y", ""),
                "z": position.get("z", ""),
                "dx": scale.get("x", ""),
                "dy": scale.get("y", ""),
                "dz": scale.get("z", ""),
                "heading": rotation.get("z", ""),
                "visible_ratio": "",
                "source_file": json_path.name,
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert per-frame JSON labels to evaluator CSV.")
    parser.add_argument("label_dir", type=Path, help="Directory containing timestamp-named JSON label files")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    args = parser.parse_args()

    row_count = convert_label_dir(args.label_dir, args.output_csv)
    print(f"wrote {row_count} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
