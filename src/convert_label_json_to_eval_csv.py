#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def parse_timestamp(stem: str):
    if "." not in stem:
        raise ValueError(f"filename stem must be '<stamp_sec>.<stamp_nsec>', got: {stem}")
    stamp_sec, stamp_nsec = stem.split(".", 1)
    return int(stamp_sec), int(stamp_nsec)


def parse_distance_group(scene_name: str):
    match = re.search(r"(\d+)m", scene_name)
    if not match:
        return scene_name
    return f"{match.group(1)}m"


def build_row(obj, stamp_sec, stamp_nsec, source_file, sequence_id="", distance_group=""):
    psr = obj.get("psr", {})
    position = psr.get("position", {})
    rotation = psr.get("rotation", {})
    scale = psr.get("scale", {})
    return {
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
        "sequence_id": sequence_id,
        "distance_group": distance_group,
        "source_file": source_file,
    }


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
        "sequence_id",
        "distance_group",
        "source_file",
    ]
    rows = []

    for json_path in sorted(label_dir.glob("*.json")):
        stamp_sec, stamp_nsec = parse_timestamp(json_path.stem)
        with json_path.open("r") as f:
            objects = json.load(f)

        for obj in objects:
            rows.append(build_row(obj, stamp_sec, stamp_nsec, json_path.name))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def convert_scene_tree(root_dir: Path, output_csv: Path, gap_seconds: float):
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
        "sequence_id",
        "distance_group",
        "source_file",
    ]
    rows = []

    for scene_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        label_dir = scene_dir / "label"
        if not label_dir.is_dir():
            continue

        distance_group = parse_distance_group(scene_dir.name)
        sequence_index = 1
        prev_time = None
        sequence_id = f"{scene_dir.name}_{sequence_index:02d}"

        for json_path in sorted(label_dir.glob("*.json")):
            stamp_sec, stamp_nsec = parse_timestamp(json_path.stem)
            curr_time = stamp_sec + stamp_nsec / 1e9
            if prev_time is not None and curr_time - prev_time > gap_seconds:
                sequence_index += 1
                sequence_id = f"{scene_dir.name}_{sequence_index:02d}"
            prev_time = curr_time

            with json_path.open("r") as f:
                objects = json.load(f)

            for obj in objects:
                rows.append(
                    build_row(
                        obj,
                        stamp_sec,
                        stamp_nsec,
                        json_path.name,
                        sequence_id=sequence_id,
                        distance_group=distance_group,
                    )
                )

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
    parser.add_argument(
        "--scene-root",
        action="store_true",
        help="Treat label_dir as a root containing scene folders like part3_5m/label and auto-generate sequence_id.",
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0,
        help="Gap threshold used to split sequence_id when --scene-root is enabled.",
    )
    args = parser.parse_args()

    if args.scene_root:
        row_count = convert_scene_tree(args.label_dir, args.output_csv, args.gap_seconds)
    else:
        row_count = convert_label_dir(args.label_dir, args.output_csv)
    print(f"wrote {row_count} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
