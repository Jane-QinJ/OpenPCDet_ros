#! /usr/bin/env python3.8
"""
Created on Thu Aug  6 11:27:43 2020

@author: Javier del Egido Sierra and Carlos Gómez-Huélamo

===

Modified on 23 Dec 2022
@author: Kin ZHANG (https://kin-zhang.github.io/)

Part of codes also refers: https://github.com/kwea123/ROS_notes
"""

# General use imports
import os
import time
import glob
import csv
import json
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ROS imports
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker

# Math and geometry imports
import math
import numpy as np
import torch

# OpenPCDet imports
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

# Kin's utils
from utils.draw_3d import Draw3DBox
from utils.global_def import *
from utils import *

import yaml
import os
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
with open(f"{BASE_DIR}/launch/config.yaml", 'r') as f:
    try:
        para_cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
    except:
        para_cfg = yaml.safe_load(f)

cfg_root = para_cfg["cfg_root"]
model_path = para_cfg["model_path"]
threshold = para_cfg["threshold"]
pointcloud_topic = para_cfg["pointcloud_topic"]
RATE_VIZ = para_cfg["viz_rate"]
output_file = para_cfg.get("output_file")
inference_time_list = []
skipped_empty_input_frames = []
skipped_sparse_forward_frames = []
PERF_CFG = para_cfg.get("performance", {})
ARCHIVE_CFG = para_cfg.get("archive", {})
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


EVAL_CFG = para_cfg.get("evaluation", {})


def _timestamped_output_path(path_value, enabled=True):
    if not path_value:
        return path_value
    if not enabled:
        return path_value
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}_{RUN_TIMESTAMP}{path.suffix}"))


class PerformanceLogger:
    def __init__(self, perf_cfg):
        self.enabled = _to_bool(perf_cfg.get("enabled"), True)
        self.sensor_hz = float(perf_cfg.get("sensor_hz", 10.0))
        self.summary_path = _timestamped_output_path(perf_cfg.get(
            "summary_file",
            os.path.join(BASE_DIR, "result", "eval", "performance_summary.json"),
        ), _to_bool(perf_cfg.get("use_timestamped_filename"), True))

    def flush(self, frame_evaluator=None):
        if not self.enabled:
            return

        times = np.asarray(inference_time_list, dtype=np.float64)
        if times.size == 0:
            summary = {
                "sensor_hz": self.sensor_hz,
                "frame_period_sec": 1.0 / self.sensor_hz if self.sensor_hz > 0 else None,
                "num_inference_frames": 0,
                "skipped_empty_input_frames": skipped_empty_input_frames,
                "skipped_sparse_forward_frames": skipped_sparse_forward_frames,
            }
        else:
            summary = {
                "sensor_hz": self.sensor_hz,
                "frame_period_sec": 1.0 / self.sensor_hz if self.sensor_hz > 0 else None,
                "num_inference_frames": int(times.size),
                "mean_inference_time_sec": float(times.mean()),
                "median_inference_time_sec": float(np.median(times)),
                "p95_inference_time_sec": float(np.percentile(times, 95)),
                "max_inference_time_sec": float(times.max()),
                "model_fps_mean": float(1.0 / times.mean()) if times.mean() > 0 else None,
                "model_fps_median": float(1.0 / np.median(times)) if np.median(times) > 0 else None,
                "skipped_empty_input_frames": skipped_empty_input_frames,
                "skipped_sparse_forward_frames": skipped_sparse_forward_frames,
            }

        if frame_evaluator is not None and frame_evaluator.gt_latency:
            latency_seconds = {
                gt_id: float(frames / self.sensor_hz) if self.sensor_hz > 0 else None
                for gt_id, frames in frame_evaluator.gt_latency.items()
            }
            summary["latency_frames"] = frame_evaluator.gt_latency
            summary["latency_seconds"] = latency_seconds
            summary["mean_latency_seconds"] = (
                float(np.mean(list(latency_seconds.values()))) if latency_seconds else None
            )

        summary_dir = os.path.dirname(self.summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        rospy.loginfo(f"Performance summary: {json.dumps(summary, ensure_ascii=True)}")


class ArchiveLogger:
    def __init__(self, archive_cfg):
        self.enabled = _to_bool(archive_cfg.get("enabled"), True)
        self.run_name = str(archive_cfg.get("run_name", "part2")).strip()
        self.archive_dir = archive_cfg.get(
            "archive_dir",
            os.path.join(BASE_DIR, "result", "eval", "archive"),
        )

    def flush(self, frame_evaluator=None, performance_logger=None):
        if not self.enabled:
            return

        archive_dir = Path(self.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.run_name}_{RUN_TIMESTAMP}"

        copied_files = {}

        if frame_evaluator is not None:
            for src_path in [frame_evaluator.summary_path, frame_evaluator.frame_log_path]:
                if not src_path:
                    continue
                src = Path(src_path)
                if not src.exists():
                    continue
                suffix = "_summary.json" if src.suffix == ".json" else "_frame_metrics.csv"
                dst = archive_dir / f"{prefix}{suffix}"
                shutil.copy2(src, dst)
                copied_files[str(dst)] = str(src)

        if performance_logger is not None and performance_logger.summary_path:
            src = Path(performance_logger.summary_path)
            if src.exists():
                dst = archive_dir / f"{prefix}_performance_summary.json"
                shutil.copy2(src, dst)
                copied_files[str(dst)] = str(src)

        manifest_path = archive_dir / f"{prefix}_archive_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "run_name": self.run_name,
                "timestamp": RUN_TIMESTAMP,
                "copied_files": copied_files,
            }, f, indent=2)

        rospy.loginfo(f"Archive manifest saved: {manifest_path}")


class FrameEvaluator:
    """
    Optional frame-wise evaluator driven by per-frame ground-truth boxes.

    Expected ground-truth CSV columns:
        Either:
            frame, gt_id, class_name, x, y, z, dx, dy, dz, heading
        Or:
            stamp_sec, stamp_nsec, gt_id, class_name, x, y, z, dx, dy, dz, heading
    Optional columns:
        visible_ratio, frame
    """

    def __init__(self, eval_cfg):
        self.enabled = _to_bool(eval_cfg.get("enabled"), False)
        self.class_name = eval_cfg.get("class_name", "Pedestrian")
        self.gt_path = eval_cfg.get("ground_truth_file")
        timestamp_enabled = _to_bool(eval_cfg.get("use_timestamped_filename"), True)
        self.frame_log_path = _timestamped_output_path(eval_cfg.get("frame_log_file"), timestamp_enabled)
        self.summary_path = _timestamped_output_path(eval_cfg.get("summary_file"), timestamp_enabled)
        self.iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
        self.track_iou_threshold = float(eval_cfg.get("track_iou_threshold", 0.1))
        self.stable_frames = max(int(eval_cfg.get("stable_frames", 3)), 1)
        self.crowd_distance_threshold = float(eval_cfg.get("crowd_distance_threshold", 1.5))
        self.match_key = str(eval_cfg.get("match_key", "stamp")).strip().lower()
        self.distance_bins = eval_cfg.get("distance_bins", [0.0, 5.0, 15.0, 20.0, float("inf")])
        self.distance_bin_names = eval_cfg.get(
            "distance_bin_names",
            ["near_0_5m", "mid_5_15m", "far_15_20m", "beyond_20m"],
        )
        self.gt_by_key = defaultdict(list)
        self.prev_tracks = {}
        self.next_track_id = 1
        self.prev_gt_to_track = {}
        self.frame_rows = []
        self.metrics = {
            "total_gt": 0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "id_switches": 0,
            "matched_iou_sum": 0.0,
            "matched_confidence_sum": 0.0,
            "matched_count": 0,
            "distance_bins": defaultdict(
                lambda: {
                    "gt": 0,
                    "tp": 0,
                    "center_error_sum": 0.0,
                    "center_error_count": 0,
                    "gt_point_count_sum": 0,
                    "gt_point_count_count": 0,
                    "trigger_point_counts": [],
                }
            ),
            "visibility_bins": defaultdict(lambda: {"gt": 0, "tp": 0}),
            "crowd_gt": 0,
            "crowd_missed": 0,
        }
        self.gt_first_seen = {}
        self.gt_streak = defaultdict(int)
        self.gt_latency = {}
        self.gt_best_visibility = {}
        self.visibility_available = False

        if self.enabled:
            self._validate_bins()
            self._load_ground_truth()

    def _validate_bins(self):
        if len(self.distance_bin_names) != len(self.distance_bins) - 1:
            raise ValueError("evaluation.distance_bin_names must have len(distance_bins) - 1 entries")

    def _load_ground_truth(self):
        if not self.gt_path:
            raise ValueError("evaluation.enabled=true requires evaluation.ground_truth_file")

        required = {"gt_id", "class_name", "x", "y", "z", "dx", "dy", "dz", "heading"}
        with open(self.gt_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"ground truth CSV missing columns: {sorted(missing)}")
            has_stamp = {"stamp_sec", "stamp_nsec"}.issubset(set(reader.fieldnames or []))
            has_frame = "frame" in (reader.fieldnames or [])
            if not has_stamp and not has_frame:
                raise ValueError("ground truth CSV must include either frame or stamp_sec/stamp_nsec")
            if self.match_key == "stamp" and not has_stamp:
                raise ValueError("evaluation.match_key=stamp requires stamp_sec and stamp_nsec in ground truth CSV")
            if self.match_key == "frame" and not has_frame:
                raise ValueError("evaluation.match_key=frame requires frame in ground truth CSV")

            self.visibility_available = "visible_ratio" in (reader.fieldnames or [])
            for row in reader:
                if row["class_name"] != self.class_name:
                    continue
                frame_value = int(row["frame"]) if has_frame and row.get("frame", "") != "" else None
                stamp_sec = int(row["stamp_sec"]) if has_stamp and row.get("stamp_sec", "") != "" else None
                stamp_nsec = int(row["stamp_nsec"]) if has_stamp and row.get("stamp_nsec", "") != "" else None
                gt_item = {
                    "frame": frame_value,
                    "stamp_sec": stamp_sec,
                    "stamp_nsec": stamp_nsec,
                    "gt_id": str(row["gt_id"]),
                    "class_name": row["class_name"],
                    "box": np.array([
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                        float(row["dx"]),
                        float(row["dy"]),
                        float(row["dz"]),
                        float(row["heading"]),
                    ], dtype=np.float32),
                }
                if self.visibility_available and row.get("visible_ratio", "") != "":
                    gt_item["visible_ratio"] = float(row["visible_ratio"])
                self.gt_by_key[self._make_key(frame_value, stamp_sec, stamp_nsec)].append(gt_item)

    def _make_key(self, frame, stamp_sec, stamp_nsec):
        if self.match_key == "stamp":
            return ("stamp", int(stamp_sec), int(stamp_nsec))
        return ("frame", int(frame))

    def _distance_bin_name(self, distance):
        for idx in range(len(self.distance_bins) - 1):
            low = self.distance_bins[idx]
            high = self.distance_bins[idx + 1]
            if low <= distance < high:
                return self.distance_bin_names[idx]
        return self.distance_bin_names[-1]

    def _visibility_bin_name(self, visible_ratio):
        if visible_ratio < 0.3:
            return "vis_0_30"
        if visible_ratio < 0.5:
            return "vis_30_50"
        if visible_ratio < 0.7:
            return "vis_50_70"
        return "vis_70_100"

    def _compute_iou_matrix(self, boxes_a, boxes_b):
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

        tensor_a = torch.from_numpy(np.asarray(boxes_a, dtype=np.float32))
        tensor_b = torch.from_numpy(np.asarray(boxes_b, dtype=np.float32))

        if torch.cuda.is_available():
            tensor_a = tensor_a.cuda()
            tensor_b = tensor_b.cuda()
            ious = iou3d_nms_utils.boxes_iou3d_gpu(tensor_a, tensor_b)
            return ious.detach().cpu().numpy()

        # Fallback to BEV IoU if CUDA IoU is unavailable at runtime.
        ious = iou3d_nms_utils.boxes_bev_iou_cpu(tensor_a, tensor_b)
        return ious.detach().cpu().numpy() if torch.is_tensor(ious) else np.asarray(ious, dtype=np.float32)

    def _count_points_in_boxes(self, points_xyz, boxes):
        if len(boxes) == 0 or points_xyz is None or len(points_xyz) == 0:
            return np.zeros((len(boxes),), dtype=np.int32)

        points_tensor = torch.from_numpy(np.asarray(points_xyz[:, :3], dtype=np.float32))
        boxes_tensor = torch.from_numpy(np.asarray(boxes, dtype=np.float32))
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor, boxes_tensor)
        if torch.is_tensor(point_masks):
            point_masks = point_masks.numpy()
        return np.asarray(point_masks.sum(axis=1), dtype=np.int32)

    def _assign_tracks(self, pred_boxes):
        assigned = [-1] * len(pred_boxes)
        if len(pred_boxes) == 0:
            self.prev_tracks = {}
            return assigned

        prev_ids = list(self.prev_tracks.keys())
        prev_boxes = [self.prev_tracks[track_id] for track_id in prev_ids]
        iou_matrix = self._compute_iou_matrix(np.asarray(prev_boxes), np.asarray(pred_boxes))

        used_prev = set()
        used_curr = set()
        if iou_matrix.size > 0:
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for flat_idx in flat_indices:
                prev_idx, curr_idx = np.unravel_index(flat_idx, iou_matrix.shape)
                if iou_matrix[prev_idx, curr_idx] < self.track_iou_threshold:
                    break
                if prev_idx in used_prev or curr_idx in used_curr:
                    continue
                track_id = prev_ids[prev_idx]
                assigned[curr_idx] = track_id
                used_prev.add(prev_idx)
                used_curr.add(curr_idx)

        for curr_idx in range(len(pred_boxes)):
            if assigned[curr_idx] == -1:
                assigned[curr_idx] = self.next_track_id
                self.next_track_id += 1

        self.prev_tracks = {assigned[idx]: pred_boxes[idx] for idx in range(len(pred_boxes))}
        return assigned

    def evaluate_frame(self, frame, stamp_sec, stamp_nsec, pred_boxes, pred_scores, pred_names, points_xyz=None):
        person_indices = [idx for idx, name in enumerate(pred_names) if name == self.class_name]
        person_boxes = np.asarray([pred_boxes[idx] for idx in person_indices], dtype=np.float32)
        person_scores = np.asarray([pred_scores[idx] for idx in person_indices], dtype=np.float32)
        person_track_ids = self._assign_tracks(person_boxes)

        gt_items = self.gt_by_key.get(self._make_key(frame, stamp_sec, stamp_nsec), [])
        gt_boxes = np.asarray([item["box"] for item in gt_items], dtype=np.float32)
        gt_distances = [calculate_distance(item["box"]) for item in gt_items]
        gt_point_counts = self._count_points_in_boxes(points_xyz, gt_boxes)

        iou_matrix = self._compute_iou_matrix(gt_boxes, person_boxes)
        gt_matches = {}
        pred_matches = {}
        if iou_matrix.size > 0:
            flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
            for flat_idx in flat_indices:
                gt_idx, pred_idx = np.unravel_index(flat_idx, iou_matrix.shape)
                iou_value = float(iou_matrix[gt_idx, pred_idx])
                if iou_value < self.iou_threshold:
                    break
                if gt_idx in gt_matches or pred_idx in pred_matches:
                    continue
                gt_matches[gt_idx] = (pred_idx, iou_value)
                pred_matches[pred_idx] = (gt_idx, iou_value)

        crowded_gt = set()
        for i in range(len(gt_items)):
            for j in range(i + 1, len(gt_items)):
                center_i = gt_items[i]["box"][:3]
                center_j = gt_items[j]["box"][:3]
                if np.linalg.norm(center_i - center_j) < self.crowd_distance_threshold:
                    crowded_gt.add(i)
                    crowded_gt.add(j)

        self.metrics["total_gt"] += len(gt_items)
        self.metrics["total_tp"] += len(gt_matches)
        self.metrics["total_fp"] += len(person_boxes) - len(pred_matches)
        self.metrics["total_fn"] += len(gt_items) - len(gt_matches)
        self.metrics["crowd_gt"] += len(crowded_gt)

        current_gt_to_track = {}

        for gt_idx, gt_item in enumerate(gt_items):
            gt_id = gt_item["gt_id"]
            gt_distance = gt_distances[gt_idx]
            distance_bin = self._distance_bin_name(gt_distance)
            self.metrics["distance_bins"][distance_bin]["gt"] += 1
            gt_point_count = int(gt_point_counts[gt_idx]) if gt_idx < len(gt_point_counts) else 0
            self.metrics["distance_bins"][distance_bin]["gt_point_count_sum"] += gt_point_count
            self.metrics["distance_bins"][distance_bin]["gt_point_count_count"] += 1
            if gt_id not in self.gt_first_seen:
                self.gt_first_seen[gt_id] = frame

            visible_ratio = gt_item.get("visible_ratio")
            if visible_ratio is not None:
                visibility_bin = self._visibility_bin_name(visible_ratio)
                self.metrics["visibility_bins"][visibility_bin]["gt"] += 1
                self.gt_best_visibility[gt_id] = max(self.gt_best_visibility.get(gt_id, 0.0), visible_ratio)

            matched = gt_idx in gt_matches
            matched_pred_idx = None
            matched_track_id = None
            matched_score = None
            matched_iou = 0.0
            center_error_m = None
            best_pred_idx = None
            best_iou_any = 0.0
            best_confidence_any = None

            if person_boxes.shape[0] > 0:
                best_pred_idx = int(np.argmax(iou_matrix[gt_idx]))
                best_iou_any = float(iou_matrix[gt_idx, best_pred_idx])
                best_confidence_any = float(person_scores[best_pred_idx])

            if matched:
                matched_pred_idx, matched_iou = gt_matches[gt_idx]
                matched_track_id = person_track_ids[matched_pred_idx]
                matched_score = float(person_scores[matched_pred_idx])
                self.metrics["matched_iou_sum"] += matched_iou
                self.metrics["matched_confidence_sum"] += matched_score
                self.metrics["matched_count"] += 1
                self.metrics["distance_bins"][distance_bin]["tp"] += 1
                center_error_m = float(np.linalg.norm(person_boxes[matched_pred_idx][:3] - gt_item["box"][:3]))
                self.metrics["distance_bins"][distance_bin]["center_error_sum"] += center_error_m
                self.metrics["distance_bins"][distance_bin]["center_error_count"] += 1
                self.metrics["distance_bins"][distance_bin]["trigger_point_counts"].append(gt_point_count)
                if visible_ratio is not None:
                    self.metrics["visibility_bins"][visibility_bin]["tp"] += 1

                prev_track = self.prev_gt_to_track.get(gt_id)
                if prev_track is not None and prev_track != matched_track_id:
                    self.metrics["id_switches"] += 1
                current_gt_to_track[gt_id] = matched_track_id

                self.gt_streak[gt_id] += 1
                if gt_id not in self.gt_latency and self.gt_streak[gt_id] >= self.stable_frames:
                    self.gt_latency[gt_id] = frame - self.gt_first_seen[gt_id] - self.stable_frames + 1
            else:
                self.gt_streak[gt_id] = 0
                if gt_idx in crowded_gt:
                    self.metrics["crowd_missed"] += 1

            self.frame_rows.append({
                "frame": frame,
                "stamp_sec": stamp_sec,
                "stamp_nsec": stamp_nsec,
                "gt_id": gt_id,
                "gt_distance": round(gt_distance, 4),
                "distance_bin": distance_bin,
                "visible_ratio": "" if visible_ratio is None else round(float(visible_ratio), 4),
                "matched": int(matched),
                "matched_track_id": "" if matched_track_id is None else matched_track_id,
                "matched_pred_local_index": "" if matched_pred_idx is None else int(person_indices[matched_pred_idx]),
                "iou": round(float(matched_iou), 4),
                "confidence": "" if matched_score is None else round(matched_score, 4),
                "center_error_m": "" if center_error_m is None else round(center_error_m, 4),
                "gt_points_in_box": gt_point_count,
                "best_pred_local_index": "" if best_pred_idx is None else int(person_indices[best_pred_idx]),
                "best_iou_any": round(float(best_iou_any), 4),
                "best_confidence_any": "" if best_confidence_any is None else round(best_confidence_any, 4),
                "crowded": int(gt_idx in crowded_gt),
            })

        self.prev_gt_to_track = current_gt_to_track

        for pred_idx in range(len(person_boxes)):
            if pred_idx in pred_matches:
                continue
            best_gt_iou = 0.0
            if gt_boxes.shape[0] > 0:
                best_gt_iou = float(np.max(iou_matrix[:, pred_idx]))
            self.frame_rows.append({
                "frame": frame,
                "stamp_sec": stamp_sec,
                "stamp_nsec": stamp_nsec,
                "gt_id": "",
                "gt_distance": "",
                "distance_bin": "",
                "visible_ratio": "",
                "matched": 0,
                "matched_track_id": person_track_ids[pred_idx],
                "matched_pred_local_index": int(person_indices[pred_idx]),
                "iou": 0.0,
                "confidence": round(float(person_scores[pred_idx]), 4),
                "center_error_m": "",
                "gt_points_in_box": "",
                "best_pred_local_index": int(person_indices[pred_idx]),
                "best_iou_any": round(float(best_gt_iou), 4),
                "best_confidence_any": round(float(person_scores[pred_idx]), 4),
                "crowded": 0,
            })

    def flush(self):
        if not self.enabled:
            return

        if self.frame_log_path:
            frame_log_dir = os.path.dirname(self.frame_log_path)
            if frame_log_dir:
                os.makedirs(frame_log_dir, exist_ok=True)
            with open(self.frame_log_path, "w", newline="") as f:
                if self.frame_rows:
                    writer = csv.DictWriter(f, fieldnames=list(self.frame_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(self.frame_rows)

        summary = {
            "class_name": self.class_name,
            "match_key": self.match_key,
            "iou_threshold": self.iou_threshold,
            "stable_frames": self.stable_frames,
            "total_gt": self.metrics["total_gt"],
            "total_tp": self.metrics["total_tp"],
            "total_fp": self.metrics["total_fp"],
            "total_fn": self.metrics["total_fn"],
            "mean_iou": self.metrics["matched_iou_sum"] / self.metrics["matched_count"] if self.metrics["matched_count"] else 0.0,
            "mean_confidence": self.metrics["matched_confidence_sum"] / self.metrics["matched_count"] if self.metrics["matched_count"] else 0.0,
            "mota": 1.0 - (
                (self.metrics["total_fn"] + self.metrics["total_fp"] + self.metrics["id_switches"]) / self.metrics["total_gt"]
            ) if self.metrics["total_gt"] else 0.0,
            "id_switches": self.metrics["id_switches"],
            "distance_wise_recall": {},
            "visibility_wise_recall": {},
            "latency_frames": self.gt_latency,
            "mean_latency_frames": float(np.mean(list(self.gt_latency.values()))) if self.gt_latency else None,
            "crowd_miss_rate": (
                self.metrics["crowd_missed"] / self.metrics["crowd_gt"]
            ) if self.metrics["crowd_gt"] else None,
        }

        for bin_name, values in self.metrics["distance_bins"].items():
            summary["distance_wise_recall"][bin_name] = (
                values["tp"] / values["gt"] if values["gt"] else 0.0
            )
        summary["distance_wise_details"] = {}
        for bin_name, values in self.metrics["distance_bins"].items():
            summary["distance_wise_details"][bin_name] = {
                "gt": int(values["gt"]),
                "tp": int(values["tp"]),
                "fn": int(values["gt"] - values["tp"]),
                "recall": values["tp"] / values["gt"] if values["gt"] else 0.0,
                "mean_center_error_m": (
                    values["center_error_sum"] / values["center_error_count"]
                    if values["center_error_count"] else None
                ),
                "mean_gt_points_in_box": (
                    values["gt_point_count_sum"] / values["gt_point_count_count"]
                    if values["gt_point_count_count"] else None
                ),
                "min_trigger_points": (
                    int(min(values["trigger_point_counts"]))
                    if values["trigger_point_counts"] else None
                ),
            }

        for bin_name, values in self.metrics["visibility_bins"].items():
            summary["visibility_wise_recall"][bin_name] = (
                values["tp"] / values["gt"] if values["gt"] else 0.0
            )

        if self.summary_path:
            summary_dir = os.path.dirname(self.summary_path)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        rospy.loginfo(f"Frame evaluator summary: {json.dumps(summary, ensure_ascii=True)}")


frame_evaluator = FrameEvaluator(EVAL_CFG)
performance_logger = PerformanceLogger(PERF_CFG)
archive_logger = ArchiveLogger(ARCHIVE_CFG)
output_file = _timestamped_output_path(output_file, _to_bool(para_cfg.get("use_timestamped_output_file"), True))


def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def calculate_distance(box):
    """
    Calculate the Euclidean distance from the origin to the object's center.
    Args:
        box: A bounding box array [x, y, z, ...].
    Returns:
        distance: The Euclidean distance.
    """
    x, y, z = box[:3]  # Assuming box contains [x, y, z, ...]
    distance = np.sqrt(x**2 + y**2 + z**2)
    return distance


def publish_distance_text(distance, box, frame_id,rate=10):
    """
    Publish the distance of the detected object as text in RViz.
    Args:
        distance: The distance to the object.
        box: The bounding box of the object [x, y, z, ...].
        frame_id: The frame ID to associate with the marker.
    """
    lifetime = 1.0/rate
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "distance_text"
    #  Ensure the ID is within the range of int32
    marker.id = abs(hash(str(box))) % 2147483647  # Modulo to fit within int32 bounds
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(lifetime)

    # Set the position of the text slightly above the object
    marker.pose.position.x = box[0]
    marker.pose.position.y = box[1]
    marker.pose.position.z = box[2] + 1.0  # Offset above the object
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set the text to display the distance
    marker.text = f"{distance:.2f} m"

    # Set the scale and color of the text
    marker.scale.z = 0.5  # Text size
    marker.color.a = 1.0  # Alpha
    marker.color.r = 1.0  # Red
    marker.color.g = 1.0  # Green
    marker.color.b = 1.0  # Blue

    # Publish the marker directly using the ROS publisher
    distance_text_publisher.publish(marker)

def rslidar_callback(msg):
    select_boxs, select_types = [],[]
    if proc_1.no_frame_id:
        proc_1.set_viz_frame_id(msg.header.frame_id)
        print(f"{bc.OKGREEN} setting marker frame id to lidar: {msg.header.frame_id} {bc.ENDC}")
        proc_1.no_frame_id = False

    frame = msg.header.seq
    stamp_sec = msg.header.stamp.secs
    stamp_nsec = msg.header.stamp.nsecs
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    # print(f"points data from msg_cloud: {msg_cloud} /t")
    
    np_p = get_xyz_points(msg_cloud, True)

    # Print points data received from the LiDAR
    # print(f"Points data received for frame {frame}:")
    # print(np_p)


    scores, dt_box_lidar, types, pred_dict = proc_1.run(np_p, frame)
    # === 新增：写入txt文件 ===
    print("Writing detection results to txt...")
    with open(output_file, "a") as f:
        for i, score in enumerate(scores):
            print(f"score: {score}")  # 调试用
            if score > threshold:
                distance = calculate_distance(dt_box_lidar[i])
                obj_type = pred_dict['name'][i]
                print(f"Writing: {frame},{stamp_sec},{stamp_nsec},{i},{distance:.3f},{score:.3f},{obj_type}")  # 调试用
                f.write(f"{frame},{stamp_sec},{stamp_nsec},{i},{distance:.3f},{score:.3f},{obj_type}\n")

    if frame_evaluator.enabled:
        frame_evaluator.evaluate_frame(frame, stamp_sec, stamp_nsec, dt_box_lidar, scores, pred_dict['name'], np_p)
    for i, score in enumerate(scores):
        if score>threshold:
            select_boxs.append(dt_box_lidar[i])
            select_types.append(pred_dict['name'][i])
            # Calculate the distance of the detected object
            distance = calculate_distance(dt_box_lidar[i])
            
            # Publish the distance as a text marker in RViz
            publish_distance_text(distance, dt_box_lidar[i], msg.header.frame_id)

            # Calculate the distance of the detected object
            distance = calculate_distance(dt_box_lidar[i])
            
            # Publish the distance as a text marker in RViz
            publish_distance_text(distance, dt_box_lidar[i], msg.header.frame_id)

    if(len(select_boxs)>0):
        # traker id is set into -1
        proc_1.pub_rviz.publish_3dbox(np.array(select_boxs), -1, pred_dict['name'])
        print_str = f"Frame id: {frame}. Prediction results: \n"
        for i in range(len(pred_dict['name'])):
            print_str += f"Type: {pred_dict['name'][i]:.3s} Prob: {scores[i]:.2f}\n"
        print(print_str)
    else:
        print(f"\n{bc.FAIL} No confident prediction in this time stamp {bc.ENDC}\n")
    print(f" -------------------------------------------------------------- ")

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
# This class initializes the detection model, processes the point cloud data, 
# and handles the publishing of detection results to RViz.
class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.pub_rviz = None
        self.no_frame_id = True
        self.rate = RATE_VIZ

    def set_pub_rviz(self, box3d_pub, marker_frame_id = 'velodyne'):
        self.pub_rviz = Draw3DBox(box3d_pub, marker_frame_id, self.rate)
    
    def set_viz_frame_id(self, marker_frame_id):
        self.pub_rviz.set_frame_id(marker_frame_id)

    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/kin/workspace/OpenPCDet/tools/000002.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        print("Model path: ", self.model_path)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def get_template_prediction(self, num_samples):
        """
        Generates a template dictionary for predictions with pre-allocated numpy arrays.

        Args:
            num_samples (int): The number of samples for which the prediction template is created.

        Returns:
            dict: A dictionary containing the following keys with pre-allocated numpy arrays:
                - 'name': Array of zeros with shape (num_samples,).
                - 'truncated': Array of zeros with shape (num_samples,).
                - 'occluded': Array of zeros with shape (num_samples,).
                - 'alpha': Array of zeros with shape (num_samples,).
                - 'bbox': Array of zeros with shape (num_samples, 4).
                - 'dimensions': Array of zeros with shape (num_samples, 3).
                - 'location': Array of zeros with shape (num_samples, 3).
                - 'rotation_y': Array of zeros with shape (num_samples,).
                - 'score': Array of zeros with shape (num_samples,).
                - 'boxes_lidar': Array of zeros with shape (num_samples, 7).
        """
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def run(self, points, frame):
        num_features = 4 # X,Y,Z,intensity       
        self.points = points.reshape([-1, num_features])
        if self.points.shape[0] == 0:
            skipped_empty_input_frames.append(int(frame))
            rospy.logwarn("Frame %s has no lidar points after preprocessing; skipping inference.", frame)
            empty_scores = np.zeros((0,), dtype=np.float32)
            empty_boxes = np.zeros((0, 7), dtype=np.float32)
            empty_types = np.zeros((0,), dtype=np.int32)
            pred_dict = self.get_template_prediction(0)
            pred_dict['name'] = np.asarray([], dtype=str)
            return empty_scores, empty_boxes, empty_types, pred_dict
        print(f"Total points: {self.points.shape[0]}")

        input_dict = {
            'points': self.points,
            'frame_id': frame,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        try:
            pred_dicts, _ = self.net.forward(data_dict)
        except RuntimeError as exc:
            err_msg = str(exc)
            if "cannot reshape tensor of 0 elements" not in err_msg:
                raise
            skipped_sparse_forward_frames.append(int(frame))
            rospy.logwarn(
                "Skipping sparse frame %s due to empty intermediate tensor in model forward: %s",
                frame,
                err_msg,
            )
            empty_scores = np.zeros((0,), dtype=np.float32)
            empty_boxes = np.zeros((0, 7), dtype=np.float32)
            empty_types = np.zeros((0,), dtype=np.int32)
            pred_dict = self.get_template_prediction(0)
            pred_dict['name'] = np.asarray([], dtype=str)
            return empty_scores, empty_boxes, empty_types, pred_dict
        
        torch.cuda.synchronize()
        inference_time = time.time() - t
        inference_time_list.append(inference_time)
        mean_inference_time = sum(inference_time_list)/len(inference_time_list)

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        pred_boxes = np.copy(boxes_lidar)
        pred_dict = self.get_template_prediction(scores.shape[0])

        pred_dict['name'] = np.array(cfg.CLASS_NAMES)[types - 1]
        pred_dict['score'] = scores
        pred_dict['boxes_lidar'] = pred_boxes

        return scores, boxes_lidar, types, pred_dict
    
 # The script initializes the ROS node,
# subscribes to the LiDAR point cloud topic, and sets up the publisher for visualization.
# The node starts spinning to process incoming messages and perform 3D object detection.
if __name__ == "__main__":
    no_frame_id = False
    proc_1 = Processor_ROS(cfg_root, model_path)
    print(f"\n{bc.OKCYAN}Config path: {bc.BOLD}{cfg_root}{bc.ENDC}")
    print(f"{bc.OKCYAN}Model path: {bc.BOLD}{model_path}{bc.ENDC}")
    print(f"{bc.OKCYAN}Run timestamp: {bc.BOLD}{RUN_TIMESTAMP}{bc.ENDC}")
    print(f"{bc.OKCYAN}Detection log: {bc.BOLD}{output_file}{bc.ENDC}")
    # print(f"If it's not correct please change in the config file... \n")

    proc_1.initialize()
    rospy.init_node('object_3d_detector_node')
    # pointcloud_topic: "/velodyne_points" from config file
    sub_lidar_topic = [pointcloud_topic]

    cfg_from_yaml_file(cfg_root, cfg)
    
    sub_ = rospy.Subscriber(sub_lidar_topic[0], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    pub_rviz = rospy.Publisher('detect_3dbox',MarkerArray, queue_size=10)
    distance_text_publisher = rospy.Publisher('distance_text_marker', Marker, queue_size=10)
    proc_1.set_pub_rviz(pub_rviz)

    if frame_evaluator.enabled:
        rospy.on_shutdown(frame_evaluator.flush)
        rospy.loginfo(
            "Frame evaluator enabled. gt=%s frame_log=%s summary=%s",
            frame_evaluator.gt_path,
            frame_evaluator.frame_log_path,
            frame_evaluator.summary_path,
        )
    if performance_logger.enabled:
        rospy.loginfo("Performance logger enabled. summary=%s", performance_logger.summary_path)
    rospy.on_shutdown(lambda: performance_logger.flush(frame_evaluator))
    if archive_logger.enabled:
        rospy.loginfo("Archive logger enabled. dir=%s run_name=%s", archive_logger.archive_dir, archive_logger.run_name)
    rospy.on_shutdown(lambda: archive_logger.flush(frame_evaluator, performance_logger))
    print(f"{bc.HEADER} ====================== {bc.ENDC}")
    print(" ===> [+] PCDet ros_node has started. Try to Run the rosbag file")
    print(f"{bc.HEADER} ====================== {bc.ENDC}")

    rospy.spin()
