"""Analysis of the real masonry masks that supports the generator parameters.

This module merges the individual analysis scripts into one command-line tool.
Every analysis keeps its own arguments, so a subcommand behaves exactly as the
separate script did.

Usage:
    python real_mask_analysis.py <command> [options]
    python real_mask_analysis.py <command> --help

Commands:
    calibrate               Calibrate the geometric crack-walk coefficients against real path shapes.
    crack-features          Extract crack-location and crack-path statistics from the colour masks.
    branch-traversal        Measure crack branching and complete mortar-brick-mortar traversals.
    high-recall-traversals  Extract high-recall crack traversals through reconstructed broken bricks.
    calibrate-with-maps     Calibrate gamma and lambda on fixed learned layout-probability maps.
    compare-paths           Compare smoothed generated crack coordinates with real crack paths.
    far-layouts             Generate clean 512 px procedural wall layouts for far-range calibration.
    preview-masks           Make deterministic mask-reconstruction previews and ordered crack polylines.
    verify-masks            Validate saved mask products, source integrity and accepted samples.
    draw-test-masks         Rasterise the 150 YOLO test annotations.
    paper-assets            Read the frozen evaluation outputs and create manuscript statistics and a figure.

Where two analyses defined the same helper differently, both definitions are
kept under distinct names rather than merged, so no result changes.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence
import argparse
import csv
import hashlib
import heapq
import html
import json
import math
import random
import re
import sys

from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from bcg_config import paths


# --------------------------------------------------------------------------
# Calibrate the geometric crack-walk coefficients against real path shapes.
# Originally calibrate_gamma_lambda.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


NEIGHBOURS_CALIB = np.asarray(
    [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ],
    dtype=np.int16,
)


def parse_args_calib() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-path-features",
        type=Path,
        default=paths.analysis('crack_mask_features')/'path_features.csv',
    )
    parser.add_argument(
        "--real-turn-angles",
        type=Path,
        default=paths.analysis('crack_mask_features')/'turn_angles.csv',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.analysis('gamma_lambda_calibration'),
    )
    parser.add_argument(
        "--gamma-values", nargs="+", type=float,
        default=(0.3, 0.5, 0.7, 0.9, 1.1),
    )
    parser.add_argument(
        "--lambda-values", nargs="+", type=float,
        default=(0.4, 0.6, 0.8, 0.9),
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--resample-step", type=float, default=5.0)
    parser.add_argument("--maximum-step-factor", type=float, default=8.0)
    parser.add_argument("--minimum-maximum-steps", type=int, default=500)
    parser.add_argument("--revisit-factor", type=float, default=0.01)
    parser.add_argument(
        "--smooth", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the documented 5-point median and 11-point cubic Savitzky-Golay refinement.",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def read_float_column(path: Path, column: str) -> np.ndarray:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
    if not values:
        raise ValueError(f"No finite values in {path}: {column}")
    return np.asarray(values, dtype=float)


def resample_path_calib(path_yx: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    if len(path_yx) < 2:
        return path_yx.copy(), 0.0
    lengths = np.linalg.norm(np.diff(path_yx, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if total <= 0.0:
        return path_yx[:1].copy(), 0.0
    targets = np.arange(0.0, total, step)
    if not len(targets) or not math.isclose(float(targets[-1]), total):
        targets = np.r_[targets, total]
    return np.column_stack(
        (
            np.interp(targets, cumulative, path_yx[:, 0]),
            np.interp(targets, cumulative, path_yx[:, 1]),
        )
    ), total


def turning_angles_calib(path_yx: np.ndarray) -> np.ndarray:
    if len(path_yx) < 3:
        return np.empty(0, dtype=float)
    vectors = np.diff(path_yx, axis=0)
    left, right = vectors[:-1], vectors[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominator > 0.0
    cosine = np.sum(left[valid] * right[valid], axis=1) / denominator[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def smooth_path(
    path_yx: np.ndarray,
    median_window: int = 5,
    savgol_window: int = 11,
    polynomial_order: int = 3,
) -> np.ndarray:
    """Reproduce the coordinate refinement used before 3-D curve creation."""
    points = np.asarray(path_yx, dtype=float)
    if len(points) < 3:
        return points.copy()

    median_window = min(median_window, len(points) if len(points) % 2 else len(points) - 1)
    median_window = max(1, median_window)
    radius = median_window // 2
    median_filtered = np.empty_like(points)
    for dimension in range(points.shape[1]):
        padded = np.pad(points[:, dimension], radius, mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, median_window)
        median_filtered[:, dimension] = np.median(windows, axis=1)

    savgol_window = min(
        savgol_window,
        len(points) if len(points) % 2 else len(points) - 1,
    )
    if savgol_window <= polynomial_order:
        return median_filtered
    half = savgol_window // 2
    result = np.empty_like(median_filtered)
    centre_offsets = np.arange(-half, half + 1, dtype=float)
    design = np.vander(centre_offsets, polynomial_order + 1, increasing=True)
    centre_coefficients = np.linalg.pinv(design)[0]
    for dimension in range(points.shape[1]):
        values = median_filtered[:, dimension]
        for index in range(half, len(values) - half):
            result[index, dimension] = float(
                centre_coefficients @ values[index - half:index + half + 1]
            )
        left_x = np.arange(savgol_window, dtype=float)
        left_coefficients = np.polyfit(left_x, values[:savgol_window], polynomial_order)
        result[:half, dimension] = np.polyval(
            left_coefficients, np.arange(half, dtype=float)
        )
        right_start = len(values) - savgol_window
        right_x = np.arange(right_start, len(values), dtype=float)
        right_coefficients = np.polyfit(
            right_x, values[-savgol_window:], polynomial_order
        )
        result[-half:, dimension] = np.polyval(
            right_coefficients,
            np.arange(len(values) - half, len(values), dtype=float),
        )
    return result


def path_features(path_yx: np.ndarray, resample_step: float) -> dict[str, object]:
    resampled, arc_length = resample_path_calib(path_yx, resample_step)
    endpoint_distance = float(np.linalg.norm(resampled[-1] - resampled[0]))
    tortuosity = arc_length / endpoint_distance if endpoint_distance > 0.0 else float("nan")
    angles = turning_angles_calib(resampled)
    goal = resampled[-1]
    distances = np.linalg.norm(resampled - goal, axis=1)
    progress = distances[:-1] - distances[1:]
    return {
        "tortuosity": tortuosity,
        "turn_angles": angles,
        "nonpositive_goal_progress_fraction": float(np.mean(progress <= 0.0)),
    }


def sample_path(
    endpoint_distance: float,
    endpoint_angle: float,
    gamma: float,
    inertia: float,
    seed: int,
    revisit_factor: float,
    max_steps: int,
) -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng(seed)
    start = np.asarray((0, 0), dtype=np.int32)
    goal = np.rint(
        endpoint_distance
        * np.asarray((math.sin(endpoint_angle), math.cos(endpoint_angle)))
    ).astype(np.int32)
    if np.all(goal == start):
        goal[1] = 1

    current = start.copy()
    initial = np.sign(goal - start).astype(np.int16)
    previous_direction = initial if np.any(initial) else np.asarray((0, 1), dtype=np.int16)
    path = [tuple(map(int, current))]
    visited = {path[0]}

    for _ in range(max_steps):
        if int(np.abs(goal - current).sum()) <= 2:
            return np.asarray(path, dtype=float), True

        candidates = current[None, :] + NEIGHBOURS_CALIB
        current_distance = float(np.linalg.norm(current - goal))
        candidate_distance = np.linalg.norm(candidates - goal[None, :], axis=1)
        progress = current_distance - candidate_distance
        goal_weight = np.exp(np.clip(gamma * progress, -50.0, 50.0))

        previous_norm = float(np.linalg.norm(previous_direction))
        candidate_norm = np.linalg.norm(NEIGHBOURS_CALIB, axis=1)
        cosine = (
            NEIGHBOURS_CALIB @ previous_direction.astype(float)
            / np.maximum(candidate_norm * previous_norm, 1e-12)
        )
        direction_weight = 1.0 + inertia * cosine
        revisit_weight = np.asarray(
            [revisit_factor if tuple(map(int, point)) in visited else 1.0 for point in candidates],
            dtype=float,
        )
        weights = goal_weight * direction_weight * revisit_weight
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        choice = int(rng.choice(len(candidates), p=weights / total))
        next_point = candidates[choice].astype(np.int32)
        previous_direction = (next_point - current).astype(np.int16)
        current = next_point
        item = tuple(map(int, current))
        path.append(item)
        visited.add(item)

    return np.asarray(path, dtype=float), False


def robust_scale(values: np.ndarray) -> float:
    scale = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    return max(scale, 1e-12)


def wasserstein_distance_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact first Wasserstein distance for two unweighted 1-D samples."""
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if not len(left) or not len(right):
        return float("inf")
    support = np.sort(np.concatenate((left, right)))
    if len(support) < 2:
        return 0.0
    deltas = np.diff(support)
    left_cdf = np.searchsorted(left, support[:-1], side="right") / len(left)
    right_cdf = np.searchsorted(right, support[:-1], side="right") / len(right)
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def summarise(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def write_csv_calib(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_calib() -> None:
    args = parse_args_calib()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_tortuosity = read_float_column(args.real_path_features, "tortuosity")
    real_progress = read_float_column(
        args.real_path_features, "nonpositive_goal_progress_fraction"
    )
    real_endpoint_distance = read_float_column(args.real_path_features, "endpoint_distance_px")
    real_angles = read_float_column(args.real_turn_angles, "turn_angle_deg")
    real_distributions = {
        "tortuosity": real_tortuosity,
        "turn_angle_deg": real_angles,
        "nonpositive_goal_progress_fraction": real_progress,
    }
    scales = {name: robust_scale(values) for name, values in real_distributions.items()}

    rng = np.random.default_rng(args.seed)
    sampled_distances = rng.choice(real_endpoint_distance, size=args.samples, replace=True)
    sampled_directions = rng.uniform(0.0, 2.0 * math.pi, size=args.samples)
    sampled_seeds = rng.integers(0, np.iinfo(np.int32).max, size=args.samples)

    detail_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for gamma in args.gamma_values:
        for inertia in args.lambda_values:
            tortuosities: list[float] = []
            all_angles: list[float] = []
            progress_fractions: list[float] = []
            completed = 0
            for sample_index, (distance, direction, seed) in enumerate(
                zip(sampled_distances, sampled_directions, sampled_seeds), start=1
            ):
                max_steps = max(
                    args.minimum_maximum_steps,
                    int(math.ceil(args.maximum_step_factor * float(distance))),
                )
                path, reached = sample_path(
                    endpoint_distance=float(distance),
                    endpoint_angle=float(direction),
                    gamma=float(gamma),
                    inertia=float(inertia),
                    seed=int(seed),
                    revisit_factor=float(args.revisit_factor),
                    max_steps=max_steps,
                )
                if not reached:
                    detail_rows.append(
                        {
                            "gamma": gamma,
                            "lambda": inertia,
                            "sample": sample_index,
                            "endpoint_distance_px": distance,
                            "completed": 0,
                            "path_points": len(path),
                            "tortuosity": "",
                            "median_turn_angle_deg": "",
                            "nonpositive_goal_progress_fraction": "",
                        }
                    )
                    continue
                completed += 1
                measured_path = smooth_path(path) if args.smooth else path
                features = path_features(measured_path, args.resample_step)
                angles = np.asarray(features["turn_angles"], dtype=float)
                tortuosities.append(float(features["tortuosity"]))
                all_angles.extend(map(float, angles))
                progress_fractions.append(
                    float(features["nonpositive_goal_progress_fraction"])
                )
                detail_rows.append(
                    {
                        "gamma": gamma,
                        "lambda": inertia,
                        "sample": sample_index,
                        "endpoint_distance_px": distance,
                        "completed": 1,
                        "path_points": len(path),
                        "tortuosity": features["tortuosity"],
                        "median_turn_angle_deg": float(np.median(angles)) if len(angles) else "",
                        "nonpositive_goal_progress_fraction": features[
                            "nonpositive_goal_progress_fraction"
                        ],
                    }
                )

            synthetic_distributions = {
                "tortuosity": np.asarray(tortuosities, dtype=float),
                "turn_angle_deg": np.asarray(all_angles, dtype=float),
                "nonpositive_goal_progress_fraction": np.asarray(
                    progress_fractions, dtype=float
                ),
            }
            distances: dict[str, float] = {}
            for name, real_values in real_distributions.items():
                synthetic_values = synthetic_distributions[name]
                distances[name] = (
                    wasserstein_distance_1d(real_values, synthetic_values) / scales[name]
                    if len(synthetic_values) else float("inf")
                )
            failure_fraction = 1.0 - completed / args.samples
            combined = float(np.mean(list(distances.values())) + failure_fraction)
            tortuosity_summary = summarise(synthetic_distributions["tortuosity"])
            angle_summary = summarise(synthetic_distributions["turn_angle_deg"])
            progress_summary = summarise(
                synthetic_distributions["nonpositive_goal_progress_fraction"]
            )
            grid_rows.append(
                {
                    "gamma": gamma,
                    "lambda": inertia,
                    "requested_paths": args.samples,
                    "completed_paths": completed,
                    "completion_fraction": completed / args.samples,
                    "tortuosity_median": tortuosity_summary["median"],
                    "tortuosity_q95": tortuosity_summary["q95"],
                    "turn_angle_median_deg": angle_summary["median"],
                    "turn_angle_q95_deg": angle_summary["q95"],
                    "nonpositive_progress_mean": progress_summary["mean"],
                    "distance_tortuosity": distances["tortuosity"],
                    "distance_turn_angle": distances["turn_angle_deg"],
                    "distance_nonpositive_progress": distances[
                        "nonpositive_goal_progress_fraction"
                    ],
                    "failure_penalty": failure_fraction,
                    "combined_distance": combined,
                }
            )
            print(
                f"gamma={gamma:.3g}, lambda={inertia:.3g}: "
                f"completed={completed}/{args.samples}, score={combined:.6f}",
                flush=True,
            )

    grid_rows.sort(key=lambda row: float(row["combined_distance"]))
    for rank, row in enumerate(grid_rows, start=1):
        row["rank"] = rank

    target_rows: list[dict[str, object]] = []
    for metric, values in real_distributions.items():
        target_rows.append({"metric": metric, **summarise(values), "robust_scale": scales[metric]})

    write_csv_calib(output_dir / "calibration_grid.csv", grid_rows)
    write_csv_calib(output_dir / "synthetic_path_features.csv", detail_rows)
    write_csv_calib(output_dir / "real_targets.csv", target_rows)
    metadata = {
        "calibration_scope": "geometric factors only; P_crack held constant",
        "formula": "exp(gamma * Euclidean goal progress) * (1 + lambda * cos(theta)) * revisit_factor",
        "real_path_features": str(args.real_path_features.resolve()),
        "real_turn_angles": str(args.real_turn_angles.resolve()),
        "samples_per_pair": args.samples,
        "resample_step_px": args.resample_step,
        "revisit_factor": args.revisit_factor,
        "coordinate_refinement": (
            "5-point median plus 11-point cubic Savitzky-Golay"
            if args.smooth else "none"
        ),
        "seed": args.seed,
        "gamma_values": args.gamma_values,
        "lambda_values": args.lambda_values,
        "score": "mean of three robustly normalised Wasserstein distances plus failure fraction",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    best = grid_rows[0]
    print(
        "\nBest candidate: gamma={gamma}, lambda={lambda}, score={combined_distance:.6f}".format(
            **best
        )
    )


# --------------------------------------------------------------------------
# Scale-normalised morphology measures for ordered crack paths.
# Originally crack_path_morphology.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


def brick_edge_distance(brick: np.ndarray) -> np.ndarray:
    edge = cv2.morphologyEx(
        brick.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    if not np.any(edge):
        return np.full(brick.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)


def prominent_extrema_count(values: np.ndarray, prominence: float) -> int:
    """Count reversals whose peak-to-trough amplitude reaches ``prominence``."""
    series = np.asarray(values, dtype=float)
    if len(series) < 3 or prominence <= 0:
        return 0
    low = high = float(series[0])
    direction = 0
    extrema = 0
    for value in series[1:]:
        value = float(value)
        if direction == 0:
            low = min(low, value)
            high = max(high, value)
            if high - low >= prominence:
                direction = 1 if value >= high else -1
            continue
        if direction > 0:
            if value > high:
                high = value
            elif high - value >= prominence:
                extrema += 1
                direction = -1
                low = value
        else:
            if value < low:
                low = value
            elif value - low >= prominence:
                extrema += 1
                direction = 1
                high = value
    return extrema


def path_morphology(
    path_yx: np.ndarray,
    brick: np.ndarray,
    brick_height: float,
    edge_distance: np.ndarray | None = None,
    refine: bool = True,
) -> dict[str, float]:
    path = np.asarray(path_yx, dtype=float)
    if len(path) < 3 or not np.isfinite(brick_height) or brick_height <= 0:
        return {
            key: float("nan")
            for key in (
                "lateral_rms_over_brick_height",
                "lateral_p95_over_brick_height",
                "lateral_range_over_brick_height",
                "prominent_extrema_per_brick_row",
                "brick_edge_follow_fraction",
                "in_brick_path_fraction",
            )
        }

    refined = smooth_path(path) if refine else path.copy()
    morphology_step = max(1.0, 0.10 * brick_height)
    sampled, _ = resample_path_calib(refined, morphology_step)
    chord = sampled[-1] - sampled[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 0:
        raise ValueError("Crack path has coincident endpoints")
    tangent = chord / chord_length
    normal = np.asarray((-tangent[1], tangent[0]), dtype=float)
    lateral = (sampled - sampled[0]) @ normal
    lateral_abs = np.abs(lateral)

    extrema = prominent_extrema_count(lateral, 0.05 * brick_height)
    vertical_rows = abs(float(sampled[-1, 0] - sampled[0, 0])) / brick_height

    material_sample, _ = resample_path_calib(refined, 1.0)
    pixels = np.rint(material_sample).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, brick.shape[0] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, brick.shape[1] - 1)
    if edge_distance is None:
        edge_distance = brick_edge_distance(brick)
    sampled_edge_distance = edge_distance[pixels[:, 0], pixels[:, 1]]
    sampled_brick = brick[pixels[:, 0], pixels[:, 1]]

    return {
        "lateral_rms_over_brick_height": float(
            np.sqrt(np.mean(lateral**2)) / brick_height
        ),
        "lateral_p95_over_brick_height": float(
            np.quantile(lateral_abs, 0.95) / brick_height
        ),
        "lateral_range_over_brick_height": float(
            (np.quantile(lateral, 0.95) - np.quantile(lateral, 0.05))
            / brick_height
        ),
        "prominent_extrema_per_brick_row": float(
            extrema / max(vertical_rows, 1.0)
        ),
        "brick_edge_follow_fraction": float(
            np.mean(sampled_edge_distance <= 0.10 * brick_height)
        ),
        "in_brick_path_fraction": float(np.mean(sampled_brick)),
    }


# --------------------------------------------------------------------------
# Extract crack-location and crack-path statistics from the colour masks.
# Originally extract_real_crack_features.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


PALETTE_FEATURES = np.asarray(
    [
        (0, 0, 0),       # mortar/background
        (0, 255, 0),     # intact brick
        (255, 0, 0),     # broken brick
        (255, 255, 0),   # crack
    ],
    dtype=np.int32,
)
CLASS_NAMES = ("mortar_or_background", "brick", "broken_brick", "crack")
BRICK_PROXIMITY_BINS = np.asarray(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, np.inf],
    dtype=float,
)


def parse_args_features() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=paths.test_annotations / 'masks_image',
        help="Directory containing colour-coded PNG masks.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=paths.analysis('crack_mask_features'),
        help="Directory in which CSV outputs and optional previews are written.",
    )
    parser.add_argument(
        "--resample-step", type=float, default=5.0,
        help="Skeleton-path resampling interval in pixels for turning angles.",
    )
    parser.add_argument(
        "--min-component-pixels", type=int, default=20,
        help="Ignore smaller skeleton components as annotation noise.",
    )
    parser.add_argument(
        "--min-brick-area", type=int, default=100,
        help="Minimum connected brick area used to estimate brick height.",
    )
    parser.add_argument(
        "--preview-count", type=int, default=0,
        help="Write quality-control overlays for the first N masks.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N masks, useful for a quick check.",
    )
    return parser.parse_args()


def classify_rgb(rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Return nearest palette class index and the number of off-palette pixels."""
    # Use int32 because squared RGB differences can reach 65,025 per
    # channel, which overflows int16 and can reverse the nearest-colour class.
    flat = rgb.reshape(-1, 3).astype(np.int32)
    exact = np.any(np.all(flat[:, None, :] == PALETTE_FEATURES[None, :, :], axis=2), axis=1)
    squared = np.sum((flat[:, None, :] - PALETTE_FEATURES[None, :, :]) ** 2, axis=2)
    classes = np.argmin(squared, axis=1).reshape(rgb.shape[:2]).astype(np.uint8)
    return classes, int((~exact).sum())


def wall_roi(brick_or_crack: np.ndarray) -> np.ndarray:
    """Approximate the wall support while excluding external black background."""
    points = np.column_stack(np.nonzero(brick_or_crack))
    if len(points) < 3:
        return np.ones(brick_or_crack.shape, dtype=bool)
    xy = points[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(xy)
    roi = np.zeros(brick_or_crack.shape, dtype=np.uint8)
    cv2.fillConvexPoly(roi, hull, 1)
    return roi.astype(bool)


def distance_to_true(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True pixel."""
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def infer_crack_substrate(
    observed_brick: np.ndarray,
    observed_mortar: np.ndarray,
    crack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer whether overwritten crack pixels were locally brick or mortar."""
    d_brick = distance_to_true(observed_brick)
    d_mortar = distance_to_true(observed_mortar)
    brick_crack = crack & (d_brick < d_mortar)
    mortar_crack = crack & (d_mortar < d_brick)
    ambiguous = crack & ~(brick_crack | mortar_crack)
    # Assign exact ties to brick only for construction of a complete binary
    # substrate.  Their count is retained separately in the output.
    reconstructed_brick = observed_brick | brick_crack | ambiguous
    return reconstructed_brick, brick_crack, mortar_crack, ambiguous, d_brick - d_mortar


def brick_component_data(
    brick: np.ndarray, min_area: int
) -> tuple[float, int, np.ndarray, dict[int, float]]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        brick.astype(np.uint8), connectivity=8
    )
    heights: list[int] = []
    height_by_label: dict[int, float] = {}
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        if area >= min_area and height >= 3 and width >= 3:
            heights.append(height)
            height_by_label[label] = float(height)
    if not heights:
        return float("nan"), 0, labels, height_by_label
    return float(np.median(heights)), len(heights), labels, height_by_label


def nearest_brick_component_labels(
    observed_brick: np.ndarray, component_labels: np.ndarray
) -> np.ndarray:
    """Propagate the nearest observed brick component label to every pixel."""
    if not np.any(observed_brick):
        return np.zeros(observed_brick.shape, dtype=np.int32)
    _, nearest_pixel_labels = cv2.distanceTransformWithLabels(
        (~observed_brick).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    lookup = np.zeros(int(nearest_pixel_labels.max()) + 1, dtype=np.int32)
    lookup[nearest_pixel_labels[observed_brick]] = component_labels[observed_brick]
    return lookup[nearest_pixel_labels]


def zhang_suen_thinning(binary: np.ndarray, max_iterations: int = 512) -> np.ndarray:
    """Vectorised Zhang-Suen thinning without a scikit-image dependency."""
    image = binary.astype(bool).copy()
    image[[0, -1], :] = False
    image[:, [0, -1]] = False

    for _ in range(max_iterations):
        changed = False
        for phase in (0, 1):
            p2 = np.roll(image, 1, axis=0)
            p3 = np.roll(p2, -1, axis=1)
            p4 = np.roll(image, -1, axis=1)
            p5 = np.roll(np.roll(image, -1, axis=0), -1, axis=1)
            p6 = np.roll(image, -1, axis=0)
            p7 = np.roll(p6, 1, axis=1)
            p8 = np.roll(image, 1, axis=1)
            p9 = np.roll(p2, 1, axis=1)

            neighbours = (p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9)
            transitions = (
                (~p2 & p3).astype(np.uint8)
                + (~p3 & p4)
                + (~p4 & p5)
                + (~p5 & p6)
                + (~p6 & p7)
                + (~p7 & p8)
                + (~p8 & p9)
                + (~p9 & p2)
            )
            remove = image & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if phase == 0:
                remove &= ~(p2 & p4 & p6)
                remove &= ~(p4 & p6 & p8)
            else:
                remove &= ~(p2 & p4 & p8)
                remove &= ~(p2 & p6 & p8)
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def skeletonise(crack: np.ndarray) -> np.ndarray:
    if not np.any(crack):
        return np.zeros_like(crack, dtype=bool)
    ys, xs = np.nonzero(crack)
    y0, y1 = max(0, int(ys.min()) - 1), min(crack.shape[0], int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min()) - 1), min(crack.shape[1], int(xs.max()) + 2)
    cropped = crack[y0:y1, x0:x1]
    thinned = zhang_suen_thinning(cropped)
    output = np.zeros_like(crack, dtype=bool)
    output[y0:y1, x0:x1] = thinned
    return output


NEIGHBOURS_FEATURES = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def graph_from_pixels(coords: np.ndarray) -> tuple[list[list[tuple[int, float]]], dict[tuple[int, int], int]]:
    lookup = {tuple(map(int, point)): i for i, point in enumerate(coords)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
    for i, (y, x) in enumerate(coords):
        for dy, dx in NEIGHBOURS_FEATURES:
            j = lookup.get((int(y + dy), int(x + dx)))
            if j is not None:
                adjacency[i].append((j, math.sqrt(2.0) if dy and dx else 1.0))
    return adjacency, lookup


def dijkstra_farthest(
    adjacency: Sequence[Sequence[tuple[int, float]]], source: int
) -> tuple[int, np.ndarray, np.ndarray]:
    distances = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=np.int32)
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    farthest = int(np.nanargmax(np.where(np.isfinite(distances), distances, -1.0)))
    return farthest, distances, previous


def approximate_graph_diameter(
    coords: np.ndarray, adjacency: Sequence[Sequence[tuple[int, float]]]
) -> np.ndarray:
    degrees = np.asarray([len(items) for items in adjacency])
    endpoints = np.flatnonzero(degrees == 1)
    start = int(endpoints[0]) if len(endpoints) else 0
    first, _, _ = dijkstra_farthest(adjacency, start)
    second, _, previous = dijkstra_farthest(adjacency, first)
    indices = [second]
    while indices[-1] != first and previous[indices[-1]] >= 0:
        indices.append(int(previous[indices[-1]]))
    indices.reverse()
    return coords[np.asarray(indices, dtype=int)].astype(float)


def orient_path(path_yx: np.ndarray) -> np.ndarray:
    """Orient a path along its dominant image axis for goal-progress statistics."""
    delta = path_yx[-1] - path_yx[0]
    if abs(delta[0]) >= abs(delta[1]):
        reverse = path_yx[0, 0] > path_yx[-1, 0]
    else:
        reverse = path_yx[0, 1] > path_yx[-1, 1]
    return path_yx[::-1].copy() if reverse else path_yx


def resample_path_features(path_yx: np.ndarray, step: float) -> tuple[np.ndarray, float]:
    if len(path_yx) < 2:
        return path_yx.copy(), 0.0
    segment_lengths = np.linalg.norm(np.diff(path_yx, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total = float(cumulative[-1])
    if total == 0.0:
        return path_yx[:1].copy(), 0.0
    targets = np.arange(0.0, total, step)
    if len(targets) == 0 or not math.isclose(float(targets[-1]), total):
        targets = np.r_[targets, total]
    y = np.interp(targets, cumulative, path_yx[:, 0])
    x = np.interp(targets, cumulative, path_yx[:, 1])
    return np.column_stack((y, x)), total


def turning_angles_features(resampled: np.ndarray) -> np.ndarray:
    if len(resampled) < 3:
        return np.empty(0, dtype=float)
    vectors = np.diff(resampled, axis=0)
    left, right = vectors[:-1], vectors[1:]
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denominators > 0
    cosine = np.ones(len(denominators), dtype=float)
    cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / denominators[valid]
    return np.degrees(np.arccos(np.clip(cosine[valid], -1.0, 1.0)))


def progress_statistics(resampled: np.ndarray) -> tuple[float, float]:
    if len(resampled) < 2:
        return float("nan"), float("nan")
    goal = resampled[-1]
    distances = np.linalg.norm(resampled - goal, axis=1)
    progress = distances[:-1] - distances[1:]
    return float(np.mean(progress)), float(np.mean(progress <= 0.0))


def junction_cluster_count(coords: np.ndarray, degrees: np.ndarray, shape: tuple[int, int]) -> int:
    junctions = coords[degrees >= 3]
    if not len(junctions):
        return 0
    mask = np.zeros(shape, dtype=np.uint8)
    mask[junctions[:, 0], junctions[:, 1]] = 1
    return int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)


def brick_runs_along_path(
    path_yx: np.ndarray,
    nearest_brick_label: np.ndarray,
    d_brick: np.ndarray,
    d_mortar: np.ndarray,
    height_by_label: dict[int, float],
) -> list[tuple[int, float, float, float]]:
    if len(path_yx) < 2:
        return []
    pixels = np.rint(path_yx).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, d_brick.shape[0] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, d_brick.shape[1] - 1)
    path_d_brick = d_brick[pixels[:, 0], pixels[:, 1]]
    path_d_mortar = d_mortar[pixels[:, 0], pixels[:, 1]]
    component_ids = nearest_brick_label[pixels[:, 0], pixels[:, 1]].copy()
    component_ids[path_d_brick >= path_d_mortar] = 0
    segment_lengths = np.linalg.norm(np.diff(path_yx, axis=0), axis=1)
    runs: list[tuple[int, float, float, float]] = []
    current = 0.0
    current_component = 0
    for i, length in enumerate(segment_lengths):
        left_component = int(component_ids[i])
        right_component = int(component_ids[i + 1])
        in_same_brick = left_component > 0 and left_component == right_component
        if in_same_brick and (current_component == 0 or current_component == left_component):
            current_component = left_component
            current += float(length)
        elif current > 0.0:
            brick_height = height_by_label.get(current_component, float("nan"))
            normalised = current / brick_height if np.isfinite(brick_height) else float("nan")
            runs.append((current_component, current, brick_height, normalised))
            current = 0.0
            current_component = 0
    if current > 0.0:
        brick_height = height_by_label.get(current_component, float("nan"))
        normalised = current / brick_height if np.isfinite(brick_height) else float("nan")
        runs.append((current_component, current, brick_height, normalised))
    return runs


def safe_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if len(finite) else float("nan")


def descriptive_rows(series: dict[str, Iterable[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, raw_values in series.items():
        values = np.asarray(list(raw_values), dtype=float)
        values = values[np.isfinite(values)]
        row: dict[str, object] = {"metric": name, "count": int(len(values))}
        if len(values):
            row.update(
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                minimum=float(np.min(values)),
                q05=float(np.quantile(values, 0.05)),
                q25=float(np.quantile(values, 0.25)),
                median=float(np.median(values)),
                q75=float(np.quantile(values, 0.75)),
                q95=float(np.quantile(values, 0.95)),
                maximum=float(np.max(values)),
            )
        rows.append(row)
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_preview(
    output_path: Path,
    classes: np.ndarray,
    skeleton: np.ndarray,
    reconstructed_brick: np.ndarray,
) -> None:
    rgb = PALETTE_FEATURES[classes].clip(0, 255).astype(np.uint8)
    gradient = cv2.morphologyEx(
        reconstructed_brick.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    rgb[gradient] = (0, 128, 255)
    rgb[skeleton] = (0, 255, 255)
    Image.fromarray(rgb).save(output_path)


def main_features() -> None:
    args = parse_args_features()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = sorted(input_dir.glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No PNG masks were found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if args.preview_count:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    angle_rows: list[dict[str, object]] = []
    brick_run_rows: list[dict[str, object]] = []
    summary_series: dict[str, list[float]] = defaultdict(list)
    proximity_total = np.zeros(len(BRICK_PROXIMITY_BINS) - 1, dtype=np.int64)
    proximity_crack = np.zeros(len(BRICK_PROXIMITY_BINS) - 1, dtype=np.int64)

    for image_index, path in enumerate(paths, start=1):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        classes, off_palette = classify_rgb(rgb)
        mortar = classes == 0
        intact = classes == 1
        broken = classes == 2
        crack = classes == 3
        observed_brick = intact | broken
        roi = wall_roi(observed_brick | crack)
        reconstructed_brick, brick_crack, mortar_crack, ambiguous, _ = infer_crack_substrate(
            observed_brick, mortar & roi, crack
        )
        # Estimate scale from visible brick components before filling the crack.
        # Filling yellow pixels can otherwise join neighbouring damaged regions
        # and produce a wall-scale component rather than a brick-scale estimate.
        brick_height, brick_count, _, _ = brick_component_data(
            observed_brick & roi, args.min_brick_area
        )
        _, _, broken_component_labels, broken_height_by_label = brick_component_data(
            broken & roi, args.min_brick_area
        )
        nearest_broken_label = nearest_brick_component_labels(
            broken & roi, broken_component_labels
        )
        d_brick = distance_to_true(observed_brick)
        d_broken = distance_to_true(broken)
        d_intact = distance_to_true(intact)
        d_mortar = distance_to_true(mortar & roi)
        d_nonbroken = np.minimum(d_intact, d_mortar)

        if np.isfinite(brick_height) and brick_height > 0:
            distance_norm = d_brick / brick_height
            non_brick_candidates = roi & ~observed_brick
            proximity_total += np.histogram(
                distance_norm[non_brick_candidates], bins=BRICK_PROXIMITY_BINS
            )[0]
            proximity_crack += np.histogram(
                distance_norm[crack & roi], bins=BRICK_PROXIMITY_BINS
            )[0]

        skeleton = skeletonise(crack)
        n_components, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)

        edge = cv2.morphologyEx(
            (reconstructed_brick & roi).astype(np.uint8),
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), np.uint8),
        ).astype(bool)
        edge_distance = distance_to_true(edge)
        skeleton_edge = edge_distance[skeleton]
        skeleton_edge_norm = (
            skeleton_edge / brick_height
            if np.isfinite(brick_height) and brick_height > 0
            else np.full_like(skeleton_edge, np.nan)
        )

        total_endpoints = 0
        total_junctions = 0
        retained_components = 0
        for component_id in range(1, n_components):
            coords = np.column_stack(np.nonzero(labels == component_id)).astype(np.int32)
            if len(coords) < args.min_component_pixels:
                continue
            retained_components += 1
            adjacency, _ = graph_from_pixels(coords)
            degrees = np.asarray([len(items) for items in adjacency], dtype=int)
            endpoints = int(np.sum(degrees == 1))
            junctions = junction_cluster_count(coords, degrees, skeleton.shape)
            total_endpoints += endpoints
            total_junctions += junctions

            main_path = orient_path(approximate_graph_diameter(coords, adjacency))
            resampled, path_length = resample_path_features(main_path, args.resample_step)
            angles = turning_angles_features(resampled)
            endpoint_distance = float(np.linalg.norm(main_path[-1] - main_path[0]))
            endpoint_dy = float(abs(main_path[-1, 0] - main_path[0, 0]))
            endpoint_dx = float(abs(main_path[-1, 1] - main_path[0, 1]))
            displacement_sum = endpoint_dy + endpoint_dx
            vertical_displacement_fraction = (
                endpoint_dy / displacement_sum if displacement_sum > 0 else float("nan")
            )
            main_y = main_path[:, 0].astype(int)
            main_x = main_path[:, 1].astype(int)
            path_inferred_brick_fraction = float(
                np.mean(brick_crack[main_y, main_x])
            )
            tortuosity = path_length / endpoint_distance if endpoint_distance > 0 else float("nan")
            mean_progress, backward_fraction = progress_statistics(resampled)

            row = {
                "image": path.name,
                "component": component_id,
                "skeleton_pixels": len(coords),
                "endpoints": endpoints,
                "junction_clusters": junctions,
                "main_path_length_px": path_length,
                "endpoint_distance_px": endpoint_distance,
                "endpoint_vertical_displacement_px": endpoint_dy,
                "endpoint_horizontal_displacement_px": endpoint_dx,
                "vertical_displacement_fraction": vertical_displacement_fraction,
                "path_inferred_brick_fraction": path_inferred_brick_fraction,
                "tortuosity": tortuosity,
                "mean_turn_angle_deg": float(np.mean(angles)) if len(angles) else float("nan"),
                "median_turn_angle_deg": safe_quantile(angles, 0.5),
                "p90_turn_angle_deg": safe_quantile(angles, 0.9),
                "turns_over_45_fraction": float(np.mean(angles > 45.0)) if len(angles) else float("nan"),
                "mean_goal_progress_px": mean_progress,
                "nonpositive_goal_progress_fraction": backward_fraction,
            }
            row.update(
                path_morphology(
                    main_path,
                    reconstructed_brick & roi,
                    brick_height,
                    edge_distance=edge_distance,
                )
            )
            path_rows.append(row)
            for angle_index, angle in enumerate(angles, start=1):
                angle_rows.append(
                    {
                        "image": path.name,
                        "component": component_id,
                        "turn_index": angle_index,
                        "turn_angle_deg": float(angle),
                    }
                )
            for run_index, (brick_component, run_px, component_height, run_norm) in enumerate(
                brick_runs_along_path(
                    main_path,
                    nearest_broken_label,
                    d_broken,
                    d_nonbroken,
                    broken_height_by_label,
                ),
                start=1,
            ):
                brick_run_rows.append(
                    {
                        "image": path.name,
                        "component": component_id,
                        "run_index": run_index,
                        "brick_component": brick_component,
                        "run_length_px": run_px,
                        "brick_height_px": component_height,
                        "run_length_over_brick_height": run_norm,
                    }
                )

        crack_pixels = int(crack.sum())
        image_row = {
            "image": path.name,
            "height": rgb.shape[0],
            "width": rgb.shape[1],
            "off_palette_pixels": off_palette,
            "roi_pixels": int(roi.sum()),
            "brick_pixels": int(intact.sum()),
            "broken_brick_pixels": int(broken.sum()),
            "crack_pixels": crack_pixels,
            "mortar_background_pixels": int(mortar.sum()),
            "crack_inferred_brick_fraction": float(brick_crack.sum() / crack_pixels) if crack_pixels else float("nan"),
            "crack_inferred_mortar_fraction": float(mortar_crack.sum() / crack_pixels) if crack_pixels else float("nan"),
            "crack_ambiguous_fraction": float(ambiguous.sum() / crack_pixels) if crack_pixels else float("nan"),
            "median_brick_height_px": brick_height,
            "retained_brick_components": brick_count,
            "skeleton_pixels": int(skeleton.sum()),
            "retained_crack_components": retained_components,
            "skeleton_endpoints": total_endpoints,
            "skeleton_junction_clusters": total_junctions,
            "median_edge_distance_over_brick_height": safe_quantile(skeleton_edge_norm, 0.5),
            "p90_edge_distance_over_brick_height": safe_quantile(skeleton_edge_norm, 0.9),
        }
        image_rows.append(image_row)

        for key in (
            "crack_inferred_brick_fraction",
            "crack_inferred_mortar_fraction",
            "crack_ambiguous_fraction",
            "median_brick_height_px",
            "median_edge_distance_over_brick_height",
            "p90_edge_distance_over_brick_height",
        ):
            summary_series[key].append(float(image_row[key]))

        if image_index <= args.preview_count:
            save_preview(preview_dir / path.name, classes, skeleton, reconstructed_brick)
        if image_index % 50 == 0 or image_index == len(paths):
            print(f"Processed {image_index}/{len(paths)} masks", flush=True)

    for row in path_rows:
        for key in (
            "main_path_length_px",
            "endpoint_distance_px",
            "tortuosity",
            "mean_turn_angle_deg",
            "median_turn_angle_deg",
            "p90_turn_angle_deg",
            "turns_over_45_fraction",
            "mean_goal_progress_px",
            "nonpositive_goal_progress_fraction",
        ):
            summary_series[key].append(float(row[key]))
    summary_series["turn_angle_deg"] = [float(row["turn_angle_deg"]) for row in angle_rows]
    summary_series["inferred_run_length_over_brick_height"] = [
        float(row["run_length_over_brick_height"]) for row in brick_run_rows
    ]

    write_rows(output_dir / "image_features.csv", image_rows, list(image_rows[0].keys()))
    if path_rows:
        write_rows(output_dir / "path_features.csv", path_rows, list(path_rows[0].keys()))
    if angle_rows:
        write_rows(output_dir / "turn_angles.csv", angle_rows, list(angle_rows[0].keys()))
    if brick_run_rows:
        write_rows(
            output_dir / "inferred_brick_run_features.csv",
            brick_run_rows,
            list(brick_run_rows[0].keys()),
        )

    proximity_rows: list[dict[str, object]] = []
    for index, (lower, upper) in enumerate(zip(BRICK_PROXIMITY_BINS[:-1], BRICK_PROXIMITY_BINS[1:])):
        total = int(proximity_total[index])
        cracks = int(proximity_crack[index])
        proximity_rows.append(
            {
                "lower_distance_to_brick_over_brick_height": lower,
                "upper_distance_to_brick_over_brick_height": upper,
                "non_brick_roi_pixels": total,
                "crack_pixels": cracks,
                "empirical_crack_probability": cracks / total if total else float("nan"),
            }
        )
    write_rows(
        output_dir / "brick_proximity_profile.csv",
        proximity_rows,
        list(proximity_rows[0].keys()),
    )

    summary_rows = descriptive_rows(summary_series)
    write_rows(
        output_dir / "dataset_summary.csv",
        summary_rows,
        ["metric", "count", "mean", "std", "minimum", "q05", "q25", "median", "q75", "q95", "maximum"],
    )
    print(f"Wrote crack-feature analysis to {output_dir}")


# --------------------------------------------------------------------------
# Measure crack branching and complete mortar-brick-mortar traversals.
# Originally analyse_branch_and_traversal.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


def parse_args_branch() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=paths.test_annotations / 'masks_image')
    parser.add_argument(
        "--output-dir", type=Path,
        default=paths.analysis('branch_and_traversal'),
    )
    parser.add_argument(
        "--branch-ratio", nargs="+", type=float,
        default=(0.05, 0.10, 0.15, 0.20),
        help="Minimum terminal-arm length as a fraction of median brick height.",
    )
    parser.add_argument(
        "--minimum-arm-widths", nargs="+", type=float,
        default=(1.0, 2.0, 3.0),
        help="Minimum arm length as a multiple of crack width at attachment.",
    )
    parser.add_argument("--minimum-spur-pixels", type=float, default=8.0)
    parser.add_argument("--minimum-component-pixels", type=int, default=20)
    parser.add_argument(
        "--minimum-component-length-ratio", type=float, default=0.25,
        help="Minimum geodesic component length as a fraction of brick height.",
    )
    parser.add_argument("--mortar-guard-pixels", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def degrees_from_adjacency(adjacency: Sequence[Sequence[tuple[int, float]]]) -> np.ndarray:
    return np.asarray([len(neighbours) for neighbours in adjacency], dtype=np.int16)


def branch_arm_lengths(
    coords: np.ndarray,
    adjacency: Sequence[Sequence[tuple[int, float]]],
    main_path_yx: np.ndarray,
    radius_map: np.ndarray,
) -> tuple[list[tuple[float, float]], int, int]:
    """Measure endpoint arms outside the approximate graph-diameter path."""
    lookup = {tuple(map(int, point)): index for index, point in enumerate(coords)}
    main_indices = {
        lookup[tuple(map(int, point))]
        for point in np.rint(main_path_yx).astype(int)
        if tuple(map(int, point)) in lookup
    }
    distances = np.full(len(adjacency), np.inf, dtype=float)
    predecessors = np.full(len(adjacency), -1, dtype=np.int32)
    queue: list[tuple[float, int]] = []
    for index in main_indices:
        distances[index] = 0.0
        heapq.heappush(queue, (0.0, index))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, weight in adjacency[node]:
            candidate = distance + float(weight)
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                predecessors[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    degrees = degrees_from_adjacency(adjacency)
    endpoints = np.flatnonzero(degrees == 1)
    arms: list[tuple[float, float]] = []
    for index in endpoints:
        if index in main_indices or not np.isfinite(distances[index]):
            continue
        attachment = int(index)
        while attachment not in main_indices and predecessors[attachment] >= 0:
            attachment = int(predecessors[attachment])
        y, x = map(int, coords[attachment])
        local_width = max(1.0, 2.0 * float(radius_map[y, x]))
        arms.append((float(distances[index]), local_width))
    junctions = junction_cluster_count(coords, degrees, (512, 512))
    return arms, int(len(endpoints)), junctions


def component_height_data(mask: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, float | bool]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    data: dict[int, dict[str, float | bool]] = {}
    height, width = mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        # Horizontal cropping does not prevent measurement of vertical brick
        # height.  Exclude only bricks whose top or bottom is truncated.
        touches_height_border = y == 0 or y + h == height
        data[label] = {
            "left": float(x),
            "right": float(x + w - 1),
            "top": float(y),
            "bottom": float(y + h - 1),
            "height": float(h),
            "width": float(w),
            "area": float(area),
            "touches_height_border": touches_height_border,
            "touches_any_border": (
                x == 0 or x + w == width or y == 0 or y + h == height
            ),
        }
    return labels, data


def path_arc_length(path: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        return 0.0
    return float(np.linalg.norm(np.diff(path[start : stop + 1], axis=0), axis=1).sum())


def contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, values.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1] - 1)) for i in range(0, len(changes), 2)]


def complete_traversals(
    path_yx: np.ndarray,
    red_state: np.ndarray,
    mortar_state: np.ndarray,
    distance_to_mortar: np.ndarray,
    broken_labels: np.ndarray,
    broken_data: dict[int, dict[str, float | bool]],
    guard: int,
) -> list[dict[str, float | int]]:
    pixels = np.rint(path_yx).astype(int)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, red_state.shape[0] - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, red_state.shape[1] - 1)
    is_red = red_state[pixels[:, 0], pixels[:, 1]]
    is_mortar = mortar_state[pixels[:, 0], pixels[:, 1]]
    labels = broken_labels[pixels[:, 0], pixels[:, 1]]
    rows: list[dict[str, float | int]] = []

    # A connected yellow path can cross several bricks. Treat every contiguous
    # run through an inferred broken-brick region as a separate candidate and
    # retain it only when mortar occurs on both sides of that run.
    seen: set[tuple[int, int, int]] = set()
    for start, stop in contiguous_true_runs(is_red):
        run_labels = labels[start : stop + 1]
        positive_labels = run_labels[run_labels > 0]
        if len(positive_labels) < 3:
            continue
        component, count = Counter(map(int, positive_labels)).most_common(1)[0]
        info = broken_data.get(component)
        if not info or count / len(positive_labels) < 0.80:
            continue
        height = float(info["height"])
        if bool(info["touches_any_border"]) or float(info["area"]) < 100 or height < 3:
            continue
        margin = max(guard, int(math.ceil(0.05 * height)))
        before = is_mortar[max(0, start - margin) : start]
        after = is_mortar[stop + 1 : min(len(is_mortar), stop + 1 + margin)]
        if not (np.any(before) and np.any(after)):
            continue
        endpoint_yx = path_yx[[start, stop]]
        left = float(info["left"])
        right = float(info["right"])
        top = float(info["top"])
        bottom = float(info["bottom"])

        def nearest_side(point_yx: np.ndarray) -> str:
            point_y, point_x = map(float, point_yx)
            distances = {
                "top": abs(point_y - top) / max(height, 1.0),
                "bottom": abs(bottom - point_y) / max(height, 1.0),
                "left": abs(point_x - left) / max(float(info["width"]), 1.0),
                "right": abs(right - point_x) / max(float(info["width"]), 1.0),
            }
            return min(distances, key=distances.get)

        entry_side = nearest_side(endpoint_yx[0])
        exit_side = nearest_side(endpoint_yx[1])
        side_pair = frozenset((entry_side, exit_side))
        if side_pair == frozenset(("top", "bottom")):
            traversal_class = "top_bottom"
        elif side_pair == frozenset(("left", "right")):
            traversal_class = "left_right"
        elif entry_side == exit_side:
            traversal_class = "same_side"
        else:
            traversal_class = "adjacent_sides"
        key = (component, start, stop)
        if key in seen:
            continue
        seen.add(key)
        arc_length = path_arc_length(path_yx, start, stop)
        rows.append(
            {
                "broken_component": component,
                "event_type": "mortar_broken_brick_mortar_run",
                "entry_side": entry_side,
                "exit_side": exit_side,
                "traversal_class": traversal_class,
                "entry_path_index": start,
                "exit_path_index": stop,
                "arc_length_px": arc_length,
                "step_count_px": float(stop - start + 1),
                "brick_height_px": height,
                "arc_length_over_brick_height": arc_length / height,
                "step_count_over_brick_height": float(stop - start + 1) / height,
            }
        )
    return rows


def write_csv_branch(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "within_0.8_to_1.4_fraction": float(np.mean((array >= 0.8) & (array <= 1.4))),
    }


def main_branch() -> None:
    args = parse_args_branch()
    paths = sorted(args.input_dir.resolve().glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No PNG masks found in {args.input_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    component_rows: list[dict[str, object]] = []
    traversal_rows: list[dict[str, object]] = []
    branch_settings = [
        (ratio, width_multiple)
        for ratio in args.branch_ratio
        for width_multiple in args.minimum_arm_widths
    ]
    image_branch_flags = {setting: {} for setting in branch_settings}

    for image_index, mask_path in enumerate(paths, start=1):
        rgb = np.asarray(Image.open(mask_path).convert("RGB"))
        classes, _ = classify_rgb(rgb)
        mortar = classes == 0
        intact = classes == 1
        broken = classes == 2
        crack = classes == 3
        roi = wall_roi(intact | broken | crack)

        observed_units = (intact | broken) & roi
        brick_height, _, _, _ = brick_component_data(observed_units, 100)
        if not np.isfinite(brick_height):
            brick_height = 0.10 * min(crack.shape)

        d_broken = distance_to_true(broken)
        d_intact = distance_to_true(intact)
        d_mortar = distance_to_true(mortar & roi)
        red_state = crack & (d_broken <= d_intact) & (d_broken <= d_mortar)
        mortar_state = crack & (d_mortar < d_broken) & (d_mortar <= d_intact)
        reconstructed_broken = broken | red_state
        broken_labels, broken_data = component_height_data(reconstructed_broken & roi)

        component_count, component_labels = cv2.connectedComponents(
            crack.astype(np.uint8), connectivity=8
        )
        image_has_eligible = False
        for component in range(1, component_count):
            component_mask = component_labels == component
            if int(component_mask.sum()) < args.minimum_component_pixels:
                continue
            skeleton = skeletonise(component_mask)
            radius_map = cv2.distanceTransform(
                component_mask.astype(np.uint8), cv2.DIST_L2, 5
            )
            coords = np.column_stack(np.nonzero(skeleton))
            if len(coords) < 3:
                continue
            adjacency, _ = graph_from_pixels(coords)
            path_yx = approximate_graph_diameter(coords, adjacency)
            _, geodesic_length = resample_path_features(path_yx, 1.0)
            minimum_length = max(20.0, args.minimum_component_length_ratio * brick_height)
            if geodesic_length < minimum_length:
                continue
            image_has_eligible = True
            row: dict[str, object] = {
                "image": mask_path.name,
                "component": component,
                "geodesic_length_px": geodesic_length,
                "median_brick_height_px": brick_height,
            }
            arms, endpoints, junctions = branch_arm_lengths(
                coords, adjacency, path_yx, radius_map
            )
            row["skeleton_endpoints"] = endpoints
            row["skeleton_junction_clusters"] = junctions
            row["maximum_off_main_arm_px"] = max((a[0] for a in arms), default=0.0)
            row["maximum_off_main_arm_over_width"] = max(
                (a[0] / a[1] for a in arms), default=0.0
            )
            for ratio, width_multiple in branch_settings:
                threshold = max(args.minimum_spur_pixels, ratio * brick_height)
                qualifying_arms = sum(
                    length >= threshold and length >= width_multiple * local_width
                    for length, local_width in arms
                )
                branched = qualifying_arms > 0 and junctions > 0
                token = (
                    str(ratio).replace(".", "p")
                    + "_width_"
                    + str(width_multiple).replace(".", "p")
                )
                row[f"branch_ratio_{token}"] = int(branched)
                row[f"qualifying_arms_ratio_{token}"] = qualifying_arms
                setting = (ratio, width_multiple)
                image_branch_flags[setting][mask_path.name] = (
                    image_branch_flags[setting].get(mask_path.name, False) or branched
                )
            component_rows.append(row)

            for traversal in complete_traversals(
                path_yx,
                red_state,
                mortar_state,
                d_mortar,
                broken_labels,
                broken_data,
                args.mortar_guard_pixels,
            ):
                traversal_rows.append(
                    {"image": mask_path.name, "crack_component": component, **traversal}
                )

        if image_has_eligible:
            for setting in branch_settings:
                image_branch_flags[setting].setdefault(mask_path.name, False)
        if image_index % 10 == 0 or image_index == len(paths):
            print(f"Processed {image_index}/{len(paths)} masks", flush=True)

    branch_summary: list[dict[str, object]] = []
    for ratio, width_multiple in branch_settings:
        token = (
            str(ratio).replace(".", "p")
            + "_width_"
            + str(width_multiple).replace(".", "p")
        )
        component_values = [int(row[f"branch_ratio_{token}"]) for row in component_rows]
        image_values = list(image_branch_flags[(ratio, width_multiple)].values())
        branch_summary.append(
            {
                "minimum_branch_length_over_brick_height": ratio,
                "minimum_branch_length_over_local_crack_width": width_multiple,
                "eligible_crack_components": len(component_values),
                "branched_components": int(sum(component_values)),
                "branched_component_fraction": float(np.mean(component_values)),
                "eligible_images": len(image_values),
                "images_with_branch": int(sum(image_values)),
                "image_branch_fraction": float(np.mean(image_values)),
            }
        )

    traversal_summary = []
    if traversal_rows:
        traversal_groups = {
            "all_mortar_brick_mortar": traversal_rows,
            "all_distinct_sides": [
                row for row in traversal_rows if row["traversal_class"] != "same_side"
            ],
        }
        for traversal_class in ("top_bottom", "left_right", "adjacent_sides", "same_side"):
            traversal_groups[traversal_class] = [
                row for row in traversal_rows
                if row["traversal_class"] == traversal_class
            ]
        for traversal_class, group in traversal_groups.items():
            if not group:
                continue
            for metric in ("arc_length_over_brick_height", "step_count_over_brick_height"):
                summary = quantiles([float(row[metric]) for row in group])
                traversal_summary.append(
                    {"traversal_class": traversal_class, "metric": metric, **summary}
                )

    write_csv_branch(output_dir / "branch_components.csv", component_rows)
    write_csv_branch(output_dir / "branch_summary.csv", branch_summary)
    write_csv_branch(output_dir / "complete_traversals.csv", traversal_rows)
    write_csv_branch(output_dir / "traversal_summary.csv", traversal_summary)

    print("\nBranch sensitivity:")
    for row in branch_summary:
        print(
            "  minimum arm {minimum_branch_length_over_brick_height:.0%} of brick height "
            "and {minimum_branch_length_over_local_crack_width:g} crack widths: "
            "{branched_components}/{eligible_crack_components} components "
            "({component_pct:.1f}%), {images_with_branch}/{eligible_images} images "
            "({image_pct:.1f}%)".format(
                **row,
                component_pct=100.0 * float(row["branched_component_fraction"]),
                image_pct=100.0 * float(row["image_branch_fraction"]),
            )
        )
    print("\nComplete mortar-broken-brick-mortar traversals:")
    for row in traversal_summary:
        print(
            "  {traversal_class}, {metric}: n={count}, median={median:.3f}, "
            "IQR={q25:.3f}-{q75:.3f}, "
            "5th-95th={q05:.3f}-{q95:.3f}, within 0.8-1.4={within:.1f}%".format(
                **row, within=100.0 * float(row["within_0.8_to_1.4_fraction"])
            )
        )


# --------------------------------------------------------------------------
# Extract high-recall crack traversals through reconstructed broken bricks.
# Originally extract_high_recall_traversals.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


NEIGHBOURS_TRAVERSALS = NEIGHBOURS_FEATURES


def parse_args_traversals() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=paths.test_annotations / 'masks_image')
    parser.add_argument(
        "--output-dir", type=Path,
        default=paths.analysis('brick_traversal_high_recall'),
    )
    parser.add_argument("--minimum-inside-pixels", type=int, default=3)
    parser.add_argument(
        "--side-tolerance", type=float, default=0.15,
        help="Maximum endpoint-to-side distance, normalised by the matching brick dimension.",
    )
    parser.add_argument(
        "--mortar-tolerance", type=float, default=0.15,
        help="Maximum distance to mortar as a fraction of brick height (minimum 3 pixels).",
    )
    parser.add_argument("--montage-count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def boundary_nodes(
    coords: np.ndarray,
    skeleton: np.ndarray,
    brick_labels: np.ndarray,
    brick_label: int,
) -> list[int]:
    output: list[int] = []
    height, width = skeleton.shape
    for index, (y_value, x_value) in enumerate(coords):
        y, x = int(y_value), int(x_value)
        for dy, dx in NEIGHBOURS_TRAVERSALS:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < height and 0 <= xx < width):
                continue
            if skeleton[yy, xx] and int(brick_labels[yy, xx]) != brick_label:
                output.append(index)
                break
    return output


def cluster_indices(coords: np.ndarray, indices: Sequence[int]) -> list[list[int]]:
    """Cluster spatially adjacent boundary nodes without joining through interior nodes."""
    remaining = set(map(int, indices))
    lookup = {tuple(map(int, coords[index])): int(index) for index in remaining}
    clusters: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        cluster = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            y, x = map(int, coords[current])
            for dy, dx in NEIGHBOURS_TRAVERSALS:
                neighbour = lookup.get((y + dy, x + dx))
                if neighbour is not None and neighbour in remaining:
                    remaining.remove(neighbour)
                    cluster.append(neighbour)
                    stack.append(neighbour)
        clusters.append(cluster)
    return clusters


def shortest_path_between_clusters(
    adjacency: Sequence[Sequence[tuple[int, float]]],
    first: Sequence[int],
    second: Sequence[int],
) -> tuple[float, list[int]]:
    targets = set(map(int, second))
    distances = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=np.int32)
    queue: list[tuple[float, int]] = []
    for source in map(int, first):
        distances[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    reached = -1
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node in targets:
            reached = node
            break
        for neighbour, weight in adjacency[node]:
            candidate = distance + float(weight)
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    if reached < 0:
        return float("nan"), []
    path = [reached]
    while previous[path[-1]] >= 0:
        path.append(int(previous[path[-1]]))
    path.reverse()
    return float(distances[reached]), path


def nearest_side(point_yx: np.ndarray, info: dict[str, float | bool]) -> tuple[str, float]:
    y, x = map(float, point_yx)
    height = max(float(info["height"]), 1.0)
    width = max(float(info["width"]), 1.0)
    distances = {
        "top": abs(y - float(info["top"])) / height,
        "bottom": abs(float(info["bottom"]) - y) / height,
        "left": abs(x - float(info["left"])) / width,
        "right": abs(float(info["right"]) - x) / width,
    }
    side = min(distances, key=distances.get)
    return side, float(distances[side])


def traversal_class(first: str, second: str) -> str:
    pair = frozenset((first, second))
    if pair == frozenset(("top", "bottom")):
        return "top_bottom"
    if pair == frozenset(("left", "right")):
        return "left_right"
    if first == second:
        return "same_side"
    return "adjacent_sides"


def write_csv_traversals(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_row(name: str, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    ratios = np.asarray([float(row["arc_length_over_brick_height"]) for row in rows])
    if len(ratios) == 0:
        return {
            "candidate_class": name, "count": 0, "images": 0,
            "crack_components": 0, "minimum": "", "median": "", "maximum": "",
            "within_0.7_to_1.4_count": 0, "within_0.7_to_1.4_fraction": "",
        }
    return {
        "candidate_class": name,
        "count": int(len(rows)),
        "images": len({str(row["image"]) for row in rows}),
        "crack_components": len({(row["image"], row["crack_component"]) for row in rows}),
        "minimum": float(np.min(ratios)),
        "median": float(np.median(ratios)),
        "maximum": float(np.max(ratios)),
        "within_0.7_to_1.4_count": int(np.sum((ratios >= 0.7) & (ratios <= 1.4))),
        "within_0.7_to_1.4_fraction": float(np.mean((ratios >= 0.7) & (ratios <= 1.4))),
    }


def make_montage(
    records: Sequence[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str]],
    path: Path,
) -> None:
    if not records:
        return
    thumb_size = 256
    columns = 6
    rows = math.ceil(len(records) / columns)
    canvas = np.full((rows * thumb_size, columns * thumb_size, 3), 255, dtype=np.uint8)
    for index, (rgb, path_yx, bbox, label) in enumerate(records):
        overlay = rgb.copy()
        y0, x0, y1, x1 = bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
        points = np.rint(path_yx[:, ::-1]).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, (255, 255, 255), 3)
            cv2.circle(overlay, tuple(points[0]), 5, (255, 0, 255), -1)
            cv2.circle(overlay, tuple(points[-1]), 5, (0, 255, 255), -1)
        cv2.putText(
            overlay, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (0, 0, 0), 1, cv2.LINE_AA,
        )
        thumb = cv2.resize(overlay, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        canvas[
            row * thumb_size : (row + 1) * thumb_size,
            column * thumb_size : (column + 1) * thumb_size,
        ] = thumb
    Image.fromarray(canvas).save(path)


def main_traversals() -> None:
    args = parse_args_traversals()
    paths = sorted(args.input_dir.resolve().glob("*.png"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No masks found in {args.input_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    montage_candidates: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str]] = []
    event_id = 0

    for image_index, mask_path in enumerate(paths, start=1):
        rgb = np.asarray(Image.open(mask_path).convert("RGB"))
        classes, _ = classify_rgb(rgb)
        mortar = classes == 0
        intact = classes == 1
        broken = classes == 2
        crack = classes == 3
        roi = wall_roi(intact | broken | crack)

        d_broken = distance_to_true(broken)
        d_intact = distance_to_true(intact)
        d_mortar = distance_to_true(mortar & roi)
        red_state = crack & (d_broken <= d_intact) & (d_broken <= d_mortar)
        reconstructed_broken = broken | red_state
        brick_labels, brick_data = component_height_data(reconstructed_broken & roi)
        skeleton = skeletonise(crack)
        crack_count, crack_labels = cv2.connectedComponents(skeleton.astype(np.uint8), 8)

        candidate_brick_labels = np.unique(brick_labels[skeleton & (brick_labels > 0)])
        for brick_label_value in candidate_brick_labels:
            brick_label = int(brick_label_value)
            info = brick_data.get(brick_label)
            if info is None:
                continue
            if float(info["area"]) < 100 or float(info["height"]) < 3:
                continue
            crop_top = max(0, int(info["top"]) - 2)
            crop_bottom = min(skeleton.shape[0], int(info["bottom"]) + 3)
            crop_left = max(0, int(info["left"]) - 2)
            crop_right = min(skeleton.shape[1], int(info["right"]) + 3)
            inside_crop = (
                skeleton[crop_top:crop_bottom, crop_left:crop_right]
                & (
                    brick_labels[crop_top:crop_bottom, crop_left:crop_right]
                    == brick_label
                )
            )
            inside_count, inside_labels = cv2.connectedComponents(
                inside_crop.astype(np.uint8), 8
            )
            for inside_component in range(1, inside_count):
                coords = np.column_stack(np.nonzero(inside_labels == inside_component))
                coords[:, 0] += crop_top
                coords[:, 1] += crop_left
                if len(coords) < args.minimum_inside_pixels:
                    continue
                adjacency, _ = graph_from_pixels(coords)
                transitions = boundary_nodes(coords, skeleton, brick_labels, brick_label)
                clusters = cluster_indices(coords, transitions)
                if len(clusters) < 2:
                    continue
                crack_component_values = crack_labels[coords[:, 0], coords[:, 1]]
                positive = crack_component_values[crack_component_values > 0]
                if len(positive) == 0:
                    continue
                crack_component = int(np.bincount(positive).argmax())
                height = float(info["height"])
                mortar_limit = max(3.0, args.mortar_tolerance * height)

                for first_index in range(len(clusters) - 1):
                    for second_index in range(first_index + 1, len(clusters)):
                        arc_length, path_indices = shortest_path_between_clusters(
                            adjacency, clusters[first_index], clusters[second_index]
                        )
                        if not path_indices or not np.isfinite(arc_length):
                            continue
                        path_yx = coords[np.asarray(path_indices, dtype=int)].astype(float)
                        first_side, first_side_distance = nearest_side(path_yx[0], info)
                        second_side, second_side_distance = nearest_side(path_yx[-1], info)
                        category = traversal_class(first_side, second_side)
                        endpoint_pixels = np.rint(path_yx[[0, -1]]).astype(int)
                        mortar_distances = d_mortar[
                            endpoint_pixels[:, 0], endpoint_pixels[:, 1]
                        ]
                        mortar_evidence = bool(np.all(mortar_distances <= mortar_limit))
                        side_evidence = (
                            first_side_distance <= args.side_tolerance
                            and second_side_distance <= args.side_tolerance
                        )
                        distinct_sides = first_side != second_side
                        complete_distinct = bool(
                            mortar_evidence and side_evidence and distinct_sides
                            and not bool(info["touches_height_border"])
                        )
                        accepted_transverse = complete_distinct and category == "top_bottom"
                        event_id += 1
                        row = {
                            "event_id": event_id,
                            "image": mask_path.name,
                            "crack_component": crack_component,
                            "broken_component": int(brick_label),
                            "inside_component": inside_component,
                            "entry_side": first_side,
                            "exit_side": second_side,
                            "traversal_class": category,
                            "entry_side_distance_ratio": first_side_distance,
                            "exit_side_distance_ratio": second_side_distance,
                            "entry_distance_to_mortar_px": float(mortar_distances[0]),
                            "exit_distance_to_mortar_px": float(mortar_distances[1]),
                            "mortar_evidence": int(mortar_evidence),
                            "side_evidence": int(side_evidence),
                            "complete_distinct_sides": int(complete_distinct),
                            "accepted_transverse": int(accepted_transverse),
                            "touches_image_border": int(bool(info["touches_any_border"])),
                            "touches_height_border": int(bool(info["touches_height_border"])),
                            "entry_y": float(path_yx[0, 0]),
                            "entry_x": float(path_yx[0, 1]),
                            "exit_y": float(path_yx[-1, 0]),
                            "exit_x": float(path_yx[-1, 1]),
                            "arc_length_px": arc_length,
                            "step_count_px": len(path_indices),
                            "brick_height_px": height,
                            "brick_width_px": float(info["width"]),
                            "arc_length_over_brick_height": arc_length / height,
                            "step_count_over_brick_height": len(path_indices) / height,
                        }
                        rows.append(row)
                        if accepted_transverse:
                            bbox = (
                                int(info["top"]), int(info["left"]),
                                int(info["bottom"]), int(info["right"]),
                            )
                            label = f"{mask_path.stem[:12]} L/H={arc_length / height:.2f}"
                            montage_candidates.append((rgb.copy(), path_yx.copy(), bbox, label))

        if image_index % 25 == 0 or image_index == len(paths):
            print(f"Processed {image_index}/{len(paths)} masks; events={len(rows)}", flush=True)

    accepted = [row for row in rows if int(row["accepted_transverse"]) == 1]
    complete = [row for row in rows if int(row["complete_distinct_sides"]) == 1]
    groups: list[tuple[str, Sequence[dict[str, object]]]] = [
        ("all_boundary_pair_candidates", rows),
        ("complete_distinct_sides", complete),
        ("accepted_transverse_top_bottom", accepted),
    ]
    for category in ("top_bottom", "left_right", "adjacent_sides", "same_side"):
        groups.append((
            f"all_{category}",
            [row for row in rows if row["traversal_class"] == category],
        ))
    summary = [summary_row(name, group) for name, group in groups]
    write_csv_traversals(output_dir / "all_candidates.csv", rows)
    write_csv_traversals(output_dir / "accepted_transverse.csv", accepted)
    write_csv_traversals(output_dir / "summary.csv", summary)

    rng = random.Random(args.seed)
    montage_count = min(args.montage_count, len(montage_candidates))
    selected = rng.sample(montage_candidates, montage_count) if montage_count else []
    make_montage(selected, output_dir / "accepted_transverse_montage.png")

    print("\nSummary")
    for row in summary:
        fraction = row["within_0.7_to_1.4_fraction"]
        fraction_text = "NA" if fraction == "" else f"{100.0 * float(fraction):.1f}%"
        print(
            f"{row['candidate_class']}: n={row['count']}, images={row['images']}, "
            f"median={row['median']}, within 0.7-1.4={fraction_text}"
        )


# --------------------------------------------------------------------------
# Calibrate gamma and lambda on fixed learned layout-probability maps.
# Originally calibrate_gamma_lambda_with_maps.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


NEIGHBOURS_CALIB_MAPS = NEIGHBOURS_CALIB


def parse_args_calib_maps() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-dir", type=Path,
        default=paths.analysis('crack_coordinates'),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=paths.checkpoint,
    )
    parser.add_argument(
        "--model-dir", type=Path, default=paths.unet_code_dir()
    )
    parser.add_argument(
        "--real-path-features", type=Path,
        default=paths.analysis('crack_mask_features')/'path_features.csv',
    )
    parser.add_argument(
        "--real-turn-angles", type=Path,
        default=paths.analysis('crack_mask_features')/'turn_angles.csv',
    )
    parser.add_argument(
        "--real-image-features", type=Path,
        default=paths.analysis('crack_mask_features')/'image_features.csv',
    )
    parser.add_argument(
        "--verticality-min", type=float, default=0.70,
        help=(
            "Minimum fraction of endpoint displacement in the vertical direction "
            "for real calibration paths."
        ),
    )
    parser.add_argument(
        "--real-min-span-rows", type=float, default=0.0,
        help="Minimum real-path vertical span measured in median brick heights.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=paths.analysis('gamma_lambda_calibration_with_maps'),
    )
    parser.add_argument(
        "--gamma-values", nargs="+", type=float,
        default=(0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    )
    parser.add_argument(
        "--lambda-values", nargs="+", type=float,
        default=(0.4, 0.6, 0.8, 0.9, 0.95),
    )
    parser.add_argument("--maps", type=int, default=30)
    parser.add_argument(
        "--map-selection", choices=("even", "far", "far_long"), default="even",
        help="Select masks evenly by filename or select far-view masks by brick scale.",
    )
    parser.add_argument(
        "--real-target-dir", type=Path, default=paths.test_annotations / 'masks_image'
    )
    parser.add_argument(
        "--real-target-selection", choices=("same", "even", "far", "far_long"),
        default="same",
    )
    parser.add_argument("--paths-per-map", type=int, default=2)
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--resample-step", type=float, default=5.0)
    parser.add_argument("--maximum-step-factor", type=float, default=8.0)
    parser.add_argument("--minimum-maximum-steps", type=int, default=7000)
    parser.add_argument("--revisit-factor", type=float, default=0.01)
    parser.add_argument(
        "--smooth", action=argparse.BooleanOptionalAction, default=True,
        help="Apply the documented coordinate refinement before measuring path morphology.",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--score-mode", choices=("conventional", "morphology"),
        default="conventional",
    )
    return parser.parse_args()


def load_model(model_dir: Path, checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    import torch
    sys.path.insert(0, str(model_dir.resolve()))
    from models import UNet  # type: ignore

    state = torch.load(checkpoint, map_location="cpu")
    model = UNet(in_ch=int(state["in_ch"]), base=32, p_drop=0.15)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def recover_layout(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    red, green, blue = (rgb[..., index].astype(np.int16) for index in range(3))
    intact = (red <= 20) & (green >= 235) & (blue <= 20)
    broken_red = (red >= 235) & (green <= 20) & (blue <= 20)
    broken_purple = (
        (np.abs(red - 128) <= 25)
        & (green <= 25)
        & (np.abs(blue - 128) <= 25)
    )
    brick = (intact | broken_red | broken_purple).astype(np.uint8)
    # Reconnect brick regions interrupted by the thin yellow preview path while
    # retaining the wider black mortar joints.
    return cv2.morphologyEx(
        brick, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)


def probability_map(
    model: torch.nn.Module,
    brick: np.ndarray,
    mc_samples: int,
    seed: int,
) -> np.ndarray:
    import torch
    mortar = ~brick
    tensor = torch.from_numpy(
        np.stack((brick.astype(np.float32), mortar.astype(np.float32)))
    ).unsqueeze(0)
    torch.manual_seed(seed)
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.lower().startswith("dropout"):
            module.train()
    predictions = []
    with torch.no_grad():
        for _ in range(mc_samples):
            predictions.append(
                torch.sigmoid(model(tensor))[0, 0].cpu().numpy().astype(np.float64)
            )
    return np.mean(np.stack(predictions, axis=0), axis=0)


def endpoint_candidates(brick: np.ndarray) -> np.ndarray:
    mortar = ~brick
    distance_to_brick = cv2.distanceTransform(
        mortar.astype(np.uint8), cv2.DIST_L2, 3
    )
    ys, xs = np.nonzero(brick)
    if not len(ys):
        return np.empty((0, 2), dtype=np.int32)
    roi = np.zeros_like(brick, dtype=bool)
    roi[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = True
    candidate = mortar & roi & (distance_to_brick >= 1.0) & (distance_to_brick <= 12.0)
    return np.column_stack(np.nonzero(candidate)).astype(np.int32)


def choose_endpoints(
    candidates: np.ndarray,
    target_distance: float,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(candidates) < 2:
        return None
    start = candidates[int(rng.integers(0, len(candidates)))]
    distances = np.linalg.norm(candidates - start[None, :], axis=1)
    valid = distances >= 0.10 * math.sqrt(2.0 * 512.0**2)
    if not np.any(valid):
        return None
    indices = np.flatnonzero(valid)
    errors = np.abs(distances[indices] - target_distance)
    nearest_count = min(30, len(indices))
    nearest = indices[np.argpartition(errors, nearest_count - 1)[:nearest_count]]
    goal = candidates[int(rng.choice(nearest))]
    return tuple(map(int, start)), tuple(map(int, goal))


def choose_vertical_endpoints(
    candidates: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(candidates) < 2:
        return None
    minimum_y = int(np.min(candidates[:, 0]))
    maximum_y = int(np.max(candidates[:, 0]))
    band = 3
    top = candidates[candidates[:, 0] <= minimum_y + band]
    bottom = candidates[candidates[:, 0] >= maximum_y - band]
    if not len(top) or not len(bottom):
        return None
    start = top[int(rng.integers(0, len(top)))]
    goal = bottom[int(rng.integers(0, len(bottom)))]
    return tuple(map(int, start)), tuple(map(int, goal))


def estimate_brick_height(mortar: np.ndarray) -> float:
    row_ratio = mortar.mean(axis=1)
    mortar_rows = np.flatnonzero(row_ratio >= 0.55)
    if len(mortar_rows) < 2:
        return float(max(8, int(round(mortar.shape[0] / 10))))
    centres: list[int] = []
    start = previous = int(mortar_rows[0])
    for row in mortar_rows[1:]:
        row = int(row)
        if row == previous + 1:
            previous = row
            continue
        centres.append((start + previous) // 2)
        start = previous = row
    centres.append((start + previous) // 2)
    differences = np.diff(np.asarray(centres, dtype=np.int32))
    differences = differences[differences >= 4]
    if not len(differences):
        return float(max(8, int(round(mortar.shape[0] / 10))))
    return float(np.median(differences))


def sample_path_on_map(
    prob: np.ndarray,
    mortar: np.ndarray,
    brick_labels: np.ndarray,
    brick_height: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    gamma: float,
    inertia: float,
    seed: int,
    revisit_factor: float,
    max_steps: int,
) -> tuple[np.ndarray, bool, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    height, width = prob.shape
    current = np.asarray(start, dtype=np.int32)
    goal_array = np.asarray(goal, dtype=np.int32)
    original_goal = goal_array.copy()
    temporary_goal: np.ndarray | None = None
    previous_direction = np.sign(goal_array - current).astype(np.int16)
    if not np.any(previous_direction):
        previous_direction = np.asarray((0, 1), dtype=np.int16)
    path = [tuple(map(int, current))]
    visited = {path[0]}
    mortar_dilated = cv2.dilate(
        mortar.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    sampling_brick = ~mortar_dilated
    distance_to_mortar = cv2.distanceTransform(
        (~mortar).astype(np.uint8), cv2.DIST_L2, 5
    )
    minimum_brick_run = int(0.8 * brick_height)
    maximum_brick_run = int(1.4 * brick_height)
    forbidden_bricks: set[int] = set()
    last_forbidden_brick: int | None = None
    entry: dict[str, object] | None = None
    brick_run_length = 0
    rollback_count = 0
    accepted_bricks: set[int] = set()

    def nearest_mortar_waypoint(point: np.ndarray, radius: int = 30) -> np.ndarray | None:
        y0 = max(0, int(point[0]) - radius)
        y1 = min(height, int(point[0]) + radius + 1)
        x0 = max(0, int(point[1]) - radius)
        x1 = min(width, int(point[1]) + radius + 1)
        local_y, local_x = np.nonzero(mortar_dilated[y0:y1, x0:x1])
        if not len(local_y):
            return None
        candidates = np.column_stack((local_y + y0, local_x + x0)).astype(np.int32)
        offset = candidates - point[None, :]
        within = np.linalg.norm(offset, axis=1) <= radius
        candidates = candidates[within]
        candidates = np.asarray(
            [item for item in candidates if tuple(map(int, item)) not in visited],
            dtype=np.int32,
        )
        if not len(candidates):
            return None
        distance = np.abs(candidates - original_goal[None, :]).sum(axis=1)
        return candidates[int(np.argmin(distance))]

    for _ in range(max_steps):
        active_goal = temporary_goal if temporary_goal is not None else original_goal
        if int(np.abs(active_goal - current).sum()) <= 2:
            if temporary_goal is not None:
                temporary_goal = None
                if last_forbidden_brick is not None:
                    forbidden_bricks.discard(last_forbidden_brick)
                    last_forbidden_brick = None
                continue
            pixels = np.asarray(path, dtype=np.int32)
            original_brick = ~mortar
            brick_fraction = float(np.mean(original_brick[pixels[:, 0], pixels[:, 1]]))
            lateral_fraction = float(
                np.mean(np.abs(np.diff(pixels[:, 1])) > 0)
            ) if len(pixels) > 1 else 0.0
            return np.asarray(path, dtype=float), True, {
                "brick_fraction": brick_fraction,
                "mortar_fraction": 1.0 - brick_fraction,
                "rollback_count": rollback_count,
                "accepted_bricks": len(accepted_bricks),
                "lateral_step_fraction": lateral_fraction,
            }
        candidates = current[None, :] + NEIGHBOURS_CALIB_MAPS
        valid = (
            (candidates[:, 0] >= 0) & (candidates[:, 0] < height)
            & (candidates[:, 1] >= 0) & (candidates[:, 1] < width)
        )
        candidates = candidates[valid]
        directions = NEIGHBOURS_CALIB_MAPS[valid]
        if not len(candidates):
            break

        current_is_brick = bool(sampling_brick[current[0], current[1]])
        if not current_is_brick and forbidden_bricks:
            keep = []
            for index, candidate in enumerate(candidates):
                candidate_is_brick = bool(sampling_brick[candidate[0], candidate[1]])
                candidate_label = int(brick_labels[candidate[0], candidate[1]])
                keep.append(not (candidate_is_brick and candidate_label in forbidden_bricks))
            keep_array = np.asarray(keep, dtype=bool)
            candidates = candidates[keep_array]
            directions = directions[keep_array]
            if not len(candidates):
                break

        current_distance = float(np.linalg.norm(current - active_goal))
        candidate_distance = np.linalg.norm(candidates - active_goal[None, :], axis=1)
        progress = current_distance - candidate_distance
        goal_weight = np.exp(np.clip(gamma * progress, -50.0, 50.0))

        previous_norm = float(np.linalg.norm(previous_direction))
        direction_norm = np.linalg.norm(directions, axis=1)
        cosine = (
            directions @ previous_direction.astype(float)
            / np.maximum(direction_norm * previous_norm, 1e-12)
        )
        direction_weight = 1.0 + inertia * cosine
        map_weight = prob[candidates[:, 0], candidates[:, 1]]
        revisit_weight = np.asarray(
            [revisit_factor if tuple(map(int, item)) in visited else 1.0 for item in candidates],
            dtype=float,
        )
        weights = map_weight * goal_weight * direction_weight * revisit_weight
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        choice = int(rng.choice(len(candidates), p=weights / total))
        next_point = candidates[choice].astype(np.int32)
        next_is_brick = bool(sampling_brick[next_point[0], next_point[1]])

        if current_is_brick and not next_is_brick and entry is not None:
            midpoint_index = int(entry["path_length"]) + brick_run_length // 2
            midpoint_index = min(midpoint_index, len(path) - 1)
            midpoint = path[midpoint_index]
            centre_distance = float(distance_to_mortar[midpoint[0], midpoint[1]])
            acceptable = (
                minimum_brick_run < brick_run_length < maximum_brick_run
                and centre_distance >= 15.0
            )
            if not acceptable:
                rollback_count += 1
                forbidden = int(entry["brick_id"])
                forbidden_bricks.add(forbidden)
                last_forbidden_brick = forbidden
                current = np.asarray(entry["point"], dtype=np.int32)
                path = list(entry["path"])
                visited = set(path)
                temporary_goal = nearest_mortar_waypoint(current)
                entry = None
                brick_run_length = 0
                continue
            accepted_bricks.add(int(entry["brick_id"]))
            entry = None
            brick_run_length = 0

        if next_is_brick and not current_is_brick:
            entry = {
                "point": tuple(map(int, current)),
                "path_length": len(path),
                "path": tuple(path),
                "brick_id": int(brick_labels[next_point[0], next_point[1]]),
            }
            brick_run_length = 1

        previous_direction = (next_point - current).astype(np.int16)
        current = next_point
        item = tuple(map(int, current))
        path.append(item)
        visited.add(item)
        if next_is_brick and current_is_brick:
            brick_run_length += 1
    pixels = np.asarray(path, dtype=np.int32)
    original_brick = ~mortar
    brick_fraction = float(np.mean(original_brick[pixels[:, 0], pixels[:, 1]]))
    lateral_fraction = float(np.mean(np.abs(np.diff(pixels[:, 1])) > 0)) if len(pixels) > 1 else 0.0
    return np.asarray(path, dtype=float), False, {
        "brick_fraction": brick_fraction,
        "mortar_fraction": 1.0 - brick_fraction,
        "rollback_count": rollback_count,
        "accepted_bricks": len(accepted_bricks),
        "lateral_step_fraction": lateral_fraction,
    }


def select_map_paths(
    map_dir: Path,
    count: int,
    selection: str,
    image_feature_file: Path,
    path_feature_file: Path | None = None,
) -> tuple[list[Path], list[dict[str, object]]]:
    paths = sorted(
        path for path in map_dir.glob("*.png")
        if "layout" not in path.stem and "_0_" not in path.stem
    )
    if not paths:
        raise FileNotFoundError(f"No generated PNG previews in {map_dir}")
    if selection in {"far", "far_long"}:
        by_name = {path.name: path for path in paths}
        candidates: list[tuple[float, int, float, Path, float]] = []
        maximum_span_rows: dict[str, float] = {}
        if selection == "far_long":
            if path_feature_file is None:
                raise ValueError("far_long selection requires path features")
            image_heights: dict[str, float] = {}
            with image_feature_file.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    height = float(row["median_brick_height_px"])
                    if np.isfinite(height) and height > 0:
                        image_heights[row["image"]] = height
            with path_feature_file.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    height = image_heights.get(row["image"])
                    if height is None:
                        continue
                    if float(row["vertical_displacement_fraction"]) < 0.70:
                        continue
                    span = float(row["endpoint_vertical_displacement_px"]) / height
                    maximum_span_rows[row["image"]] = max(
                        maximum_span_rows.get(row["image"], 0.0), span
                    )
        with image_feature_file.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                path = by_name.get(row["image"])
                if path is None:
                    continue
                brick_height = float(row["median_brick_height_px"])
                components = int(row["retained_brick_components"])
                roi_pixels = max(int(row["roi_pixels"]), 1)
                brick_fraction = (
                    int(row["brick_pixels"]) + int(row["broken_brick_pixels"])
                ) / roi_pixels
                reliable_far = (
                    np.isfinite(brick_height) and brick_height > 0
                    and components >= (20 if selection == "far_long" else 30)
                    and (0.45 if selection == "far_long" else 0.50)
                    <= brick_fraction <= 0.90
                    and brick_height <= (70.0 if selection == "far_long" else np.inf)
                )
                long_enough = (
                    selection != "far_long"
                    or maximum_span_rows.get(row["image"], 0.0) >= 5.0
                )
                if reliable_far and long_enough:
                    candidates.append(
                        (
                            brick_height,
                            components,
                            brick_fraction,
                            path,
                            maximum_span_rows.get(row["image"], float("nan")),
                        )
                    )
        candidates.sort(key=lambda item: (item[0], -item[1], item[3].name))
        selected = candidates[:count]
        if len(selected) < count:
            raise RuntimeError(f"Only {len(selected)} reliable far-view masks found")
        audit = [
            {
                "map": path.name,
                "median_brick_height_px": brick_height,
                "estimated_visible_brick_rows": 512.0 / brick_height,
                "maximum_vertical_crack_span_rows": maximum_span,
                "retained_brick_components": components,
                "brick_fraction": brick_fraction,
            }
            for brick_height, components, brick_fraction, path, maximum_span in selected
        ]
        return [item[3] for item in selected], audit
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    selected_paths = [paths[int(index)] for index in indices]
    return selected_paths, [{"map": path.name} for path in selected_paths]


def safe_summary(values: np.ndarray) -> dict[str, float]:
    if len(values):
        return summarise(values)
    return {key: float("nan") for key in ("count", "mean", "median", "q05", "q95")}


def read_vertical_real_distributions(
    path_file: Path,
    angle_file: Path,
    verticality_min: float,
    selected_images: set[str] | None = None,
    image_feature_file: Path | None = None,
    minimum_span_rows: float = 0.0,
) -> tuple[dict[str, np.ndarray], int]:
    selected: set[tuple[str, str]] = set()
    values: dict[str, list[float]] = {
        "tortuosity": [],
        "nonpositive_goal_progress_fraction": [],
        "brick_fraction": [],
        "turn_angle_deg": [],
        "lateral_rms_over_brick_height": [],
        "lateral_p95_over_brick_height": [],
        "lateral_range_over_brick_height": [],
        "prominent_extrema_per_brick_row": [],
        "brick_edge_follow_fraction": [],
        "in_brick_path_fraction": [],
    }
    brick_heights: dict[str, float] = {}
    if image_feature_file is not None and minimum_span_rows > 0:
        with image_feature_file.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                value = float(row["median_brick_height_px"])
                if np.isfinite(value) and value > 0:
                    brick_heights[row["image"]] = value
    with path_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if selected_images is not None and row["image"] not in selected_images:
                continue
            verticality = float(row["vertical_displacement_fraction"])
            if not np.isfinite(verticality) or verticality < verticality_min:
                continue
            if minimum_span_rows > 0:
                brick_height = brick_heights.get(row["image"])
                if brick_height is None:
                    continue
                span_rows = (
                    float(row["endpoint_vertical_displacement_px"])
                    / brick_height
                )
                if span_rows < minimum_span_rows:
                    continue
            key = (row["image"], row["component"])
            selected.add(key)
            for source, target in (
                ("tortuosity", "tortuosity"),
                (
                    "nonpositive_goal_progress_fraction",
                    "nonpositive_goal_progress_fraction",
                ),
                ("path_inferred_brick_fraction", "brick_fraction"),
                (
                    "lateral_rms_over_brick_height",
                    "lateral_rms_over_brick_height",
                ),
                (
                    "lateral_p95_over_brick_height",
                    "lateral_p95_over_brick_height",
                ),
                (
                    "lateral_range_over_brick_height",
                    "lateral_range_over_brick_height",
                ),
                (
                    "prominent_extrema_per_brick_row",
                    "prominent_extrema_per_brick_row",
                ),
                ("brick_edge_follow_fraction", "brick_edge_follow_fraction"),
                ("in_brick_path_fraction", "in_brick_path_fraction"),
            ):
                value = float(row[source])
                if np.isfinite(value):
                    values[target].append(value)
    with angle_file.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row["image"], row["component"]) not in selected:
                continue
            value = float(row["turn_angle_deg"])
            if np.isfinite(value):
                values["turn_angle_deg"].append(value)
    distributions = {
        name: np.asarray(series, dtype=float) for name, series in values.items()
    }
    if not selected or any(not len(series) for series in distributions.values()):
        raise RuntimeError("The vertical real-path calibration subset is empty")
    return distributions, len(selected)


def main_calib_maps() -> None:
    args = parse_args_calib_maps()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint_state = load_model(args.model_dir, args.checkpoint)

    selected_map_paths, selection_audit = select_map_paths(
        args.map_dir,
        args.maps,
        args.map_selection,
        args.real_image_features,
        args.real_path_features,
    )
    write_csv_calib(output_dir / "map_selection_audit.csv", selection_audit)
    if args.real_target_selection == "same":
        selected_target_paths = selected_map_paths
        target_selection_audit = selection_audit
    else:
        selected_target_paths, target_selection_audit = select_map_paths(
            args.real_target_dir,
            args.maps,
            args.real_target_selection,
            args.real_image_features,
            args.real_path_features,
        )
    write_csv_calib(
        output_dir / "real_target_selection_audit.csv", target_selection_audit
    )
    selected_image_names = {path.name for path in selected_target_paths}

    real_distributions, real_vertical_path_count = read_vertical_real_distributions(
        args.real_path_features,
        args.real_turn_angles,
        args.verticality_min,
        selected_image_names,
        args.real_image_features,
        args.real_min_span_rows,
    )
    scale_distributions, scale_vertical_path_count = read_vertical_real_distributions(
        args.real_path_features,
        args.real_turn_angles,
        args.verticality_min,
        image_feature_file=args.real_image_features,
        minimum_span_rows=args.real_min_span_rows,
    )
    scales = {
        name: robust_scale(scale_distributions[name])
        for name in real_distributions
    }

    rng = np.random.default_rng(args.seed)
    tasks: list[dict[str, object]] = []
    map_records: list[dict[str, object]] = []
    for map_index, path in enumerate(selected_map_paths, start=1):
        brick = recover_layout(path)
        mortar = ~brick
        _, brick_labels = cv2.connectedComponents(
            brick.astype(np.uint8), connectivity=8
        )
        brick_height = estimate_brick_height(mortar)
        edge_distance = brick_edge_distance(brick)
        candidates = np.column_stack(np.nonzero(mortar)).astype(np.int32)
        prob = probability_map(
            model,
            brick,
            mc_samples=args.mc_samples,
            seed=args.seed + map_index,
        )
        map_records.append(
            {
                "map": path.name,
                "brick_fraction": float(np.mean(brick)),
                "candidate_mortar_pixels": len(candidates),
                "probability_min": float(np.min(prob)),
                "probability_mean": float(np.mean(prob)),
                "probability_max": float(np.max(prob)),
            }
        )
        for path_index in range(1, args.paths_per_map + 1):
            endpoints = choose_vertical_endpoints(candidates, rng)
            if endpoints is None:
                continue
            start, goal = endpoints
            tasks.append(
                {
                    "map": path.name,
                    "map_index": map_index,
                    "path_index": path_index,
                    "probability": prob,
                    "mortar": mortar,
                    "brick_labels": brick_labels,
                    "brick_height": brick_height,
                    "edge_distance": edge_distance,
                    "start": start,
                    "goal": goal,
                    "seed": int(rng.integers(0, np.iinfo(np.int32).max)),
                }
            )
    if not tasks:
        raise RuntimeError("No calibration tasks could be constructed")

    detail_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    for gamma in args.gamma_values:
        for inertia in args.lambda_values:
            tortuosities: list[float] = []
            all_angles: list[float] = []
            progress_fractions: list[float] = []
            brick_fractions: list[float] = []
            rollback_counts: list[float] = []
            accepted_brick_counts: list[float] = []
            lateral_step_fractions: list[float] = []
            raw_step_angles: list[float] = []
            morphology_values: dict[str, list[float]] = {
                key: []
                for key in (
                    "lateral_rms_over_brick_height",
                    "lateral_p95_over_brick_height",
                    "lateral_range_over_brick_height",
                    "prominent_extrema_per_brick_row",
                    "brick_edge_follow_fraction",
                    "in_brick_path_fraction",
                )
            }
            completed = 0
            for task_index, task in enumerate(tasks, start=1):
                start = task["start"]
                goal = task["goal"]
                endpoint_distance = float(
                    np.linalg.norm(np.asarray(goal) - np.asarray(start))
                )
                max_steps = max(
                    args.minimum_maximum_steps,
                    int(math.ceil(args.maximum_step_factor * endpoint_distance)),
                )
                path, reached, routing = sample_path_on_map(
                    prob=np.asarray(task["probability"]),
                    mortar=np.asarray(task["mortar"]),
                    brick_labels=np.asarray(task["brick_labels"]),
                    brick_height=float(task["brick_height"]),
                    start=start,
                    goal=goal,
                    gamma=float(gamma),
                    inertia=float(inertia),
                    seed=int(task["seed"]),
                    revisit_factor=float(args.revisit_factor),
                    max_steps=max_steps,
                )
                row: dict[str, object] = {
                    "gamma": gamma,
                    "lambda": inertia,
                    "task": task_index,
                    "map": task["map"],
                    "endpoint_distance_px": endpoint_distance,
                    "completed": int(reached),
                    "path_points": len(path),
                    "brick_fraction": routing["brick_fraction"],
                    "mortar_fraction": routing["mortar_fraction"],
                    "rollback_count": routing["rollback_count"],
                    "accepted_bricks": routing["accepted_bricks"],
                    "lateral_step_fraction": routing["lateral_step_fraction"],
                }
                step_angles = turning_angles_calib(path)
                raw_step_angles.extend(map(float, step_angles))
                row.update(
                    {
                        "raw_step_turn_median_deg": (
                            float(np.median(step_angles)) if len(step_angles) else ""
                        ),
                        "raw_direction_change_fraction": (
                            float(np.mean(step_angles > 1e-6)) if len(step_angles) else ""
                        ),
                        "raw_sharp_turn_fraction": (
                            float(np.mean(step_angles >= 90.0)) if len(step_angles) else ""
                        ),
                    }
                )
                if reached:
                    completed += 1
                    measured_path = smooth_path(path) if args.smooth else path
                    features = path_features(measured_path, args.resample_step)
                    angles = np.asarray(features["turn_angles"], dtype=float)
                    tortuosities.append(float(features["tortuosity"]))
                    all_angles.extend(map(float, angles))
                    progress_fractions.append(
                        float(features["nonpositive_goal_progress_fraction"])
                    )
                    brick_fractions.append(float(routing["brick_fraction"]))
                    morphology = path_morphology(
                        path,
                        np.asarray(task["mortar"]) == 0,
                        float(task["brick_height"]),
                        edge_distance=np.asarray(task["edge_distance"]),
                    )
                    for key, value in morphology.items():
                        morphology_values[key].append(float(value))
                    row.update(morphology)
                    row.update(
                        {
                            "tortuosity": features["tortuosity"],
                            "median_turn_angle_deg": (
                                float(np.median(angles)) if len(angles) else ""
                            ),
                            "nonpositive_goal_progress_fraction": features[
                                "nonpositive_goal_progress_fraction"
                            ],
                        }
                    )
                else:
                    row.update(
                        {
                            "tortuosity": "",
                            "median_turn_angle_deg": "",
                            "nonpositive_goal_progress_fraction": "",
                        }
                    )
                detail_rows.append(row)
                rollback_counts.append(float(routing["rollback_count"]))
                accepted_brick_counts.append(float(routing["accepted_bricks"]))
                lateral_step_fractions.append(
                    float(routing["lateral_step_fraction"])
                )

            synthetic = {
                "tortuosity": np.asarray(tortuosities, dtype=float),
                "turn_angle_deg": np.asarray(all_angles, dtype=float),
                "nonpositive_goal_progress_fraction": np.asarray(
                    progress_fractions, dtype=float
                ),
                "brick_fraction": np.asarray(brick_fractions, dtype=float),
                **{
                    key: np.asarray(values, dtype=float)
                    for key, values in morphology_values.items()
                },
            }
            metric_names = (
                (
                    "lateral_rms_over_brick_height",
                    "lateral_p95_over_brick_height",
                    "prominent_extrema_per_brick_row",
                    "brick_edge_follow_fraction",
                    "in_brick_path_fraction",
                )
                if args.score_mode == "morphology"
                else (
                    "tortuosity",
                    "turn_angle_deg",
                    "nonpositive_goal_progress_fraction",
                    "brick_fraction",
                )
            )
            distances = {
                name: (
                    wasserstein_distance_1d(real_distributions[name], values)
                    / scales[name]
                    if len(values) else float("inf")
                )
                for name, values in synthetic.items()
                if name in metric_names
            }
            failure = 1.0 - completed / len(tasks)
            combined = float(np.mean(list(distances.values())) + failure)
            tortuosity_summary = safe_summary(synthetic["tortuosity"])
            angle_summary = safe_summary(synthetic["turn_angle_deg"])
            progress_summary = safe_summary(
                synthetic["nonpositive_goal_progress_fraction"]
            )
            brick_summary = safe_summary(synthetic["brick_fraction"])
            raw_step_angle_array = np.asarray(raw_step_angles, dtype=float)
            morphology_summaries = {
                key: safe_summary(values)
                for key, values in synthetic.items()
                if key in morphology_values
            }
            grid_rows.append(
                {
                    "gamma": gamma,
                    "lambda": inertia,
                    "requested_paths": len(tasks),
                    "completed_paths": completed,
                    "completion_fraction": completed / len(tasks),
                    "tortuosity_median": tortuosity_summary["median"],
                    "tortuosity_q95": tortuosity_summary["q95"],
                    "turn_angle_median_deg": angle_summary["median"],
                    "turn_angle_q95_deg": angle_summary["q95"],
                    "nonpositive_progress_mean": progress_summary["mean"],
                    "brick_fraction_median": brick_summary["median"],
                    "brick_fraction_q95": brick_summary["q95"],
                    "mean_rollbacks_per_path": float(np.mean(rollback_counts)),
                    "mean_accepted_bricks_per_path": float(
                        np.mean(accepted_brick_counts)
                    ),
                    "mean_lateral_step_fraction": float(
                        np.mean(lateral_step_fractions)
                    ),
                    "raw_step_turn_median_deg": (
                        float(np.median(raw_step_angle_array))
                        if len(raw_step_angle_array) else float("nan")
                    ),
                    "raw_direction_change_fraction": (
                        float(np.mean(raw_step_angle_array > 1e-6))
                        if len(raw_step_angle_array) else float("nan")
                    ),
                    "raw_sharp_turn_fraction": (
                        float(np.mean(raw_step_angle_array >= 90.0))
                        if len(raw_step_angle_array) else float("nan")
                    ),
                    **{
                        f"{key}_median": summary["median"]
                        for key, summary in morphology_summaries.items()
                    },
                    **{
                        f"distance_{key}": distances.get(key, float("nan"))
                        for key in morphology_values
                    },
                    "distance_tortuosity": distances.get("tortuosity", float("nan")),
                    "distance_turn_angle": distances.get("turn_angle_deg", float("nan")),
                    "distance_nonpositive_progress": distances.get(
                        "nonpositive_goal_progress_fraction", float("nan")
                    ),
                    "distance_brick_fraction": distances.get(
                        "brick_fraction", float("nan")
                    ),
                    "failure_penalty": failure,
                    "combined_distance": combined,
                }
            )
            print(
                f"gamma={gamma:.3g}, lambda={inertia:.3g}: "
                f"completed={completed}/{len(tasks)}, score={combined:.6f}",
                flush=True,
            )

    grid_rows.sort(key=lambda row: float(row["combined_distance"]))
    for rank, row in enumerate(grid_rows, start=1):
        row["rank"] = rank
    write_csv_calib(output_dir / "calibration_grid.csv", grid_rows)
    write_csv_calib(output_dir / "synthetic_path_features.csv", detail_rows)
    write_csv_calib(output_dir / "probability_map_audit.csv", map_records)
    metadata = {
        "calibration_scope": "full geometric transition weights on fixed learned probability maps",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": {
            key: value for key, value in checkpoint_state.items()
            if key not in {"model", "optimizer"}
        },
        "checkpoint_mask_encoding": checkpoint_state.get("mask_encoding", "not recorded"),
        "maps": len(map_records),
        "map_selection": args.map_selection,
        "real_target_selection": args.real_target_selection,
        "paths_per_map": args.paths_per_map,
        "mc_samples": args.mc_samples,
        "tasks": len(tasks),
        "real_verticality_min": args.verticality_min,
        "real_vertical_path_count": real_vertical_path_count,
        "real_min_span_rows": args.real_min_span_rows,
        "scale_vertical_path_count": scale_vertical_path_count,
        "coordinate_refinement": (
            "5-point median plus 11-point cubic Savitzky-Golay"
            if args.smooth else "none"
        ),
        "seed": args.seed,
        "gamma_values": args.gamma_values,
        "lambda_values": args.lambda_values,
        "score": (
            "mean of selected robustly normalised Wasserstein distances plus "
            "failure fraction"
        ),
        "score_mode": args.score_mode,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    best = grid_rows[0]
    print(
        "\nBest candidate: gamma={gamma}, lambda={lambda}, score={combined_distance:.6f}".format(
            **best
        )
    )


# --------------------------------------------------------------------------
# Compare smoothed generated crack coordinates with real crack paths.
# Originally compare_generated_crack_paths.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


POINT = re.compile(r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)")


def parse_args_compare() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir", type=Path,
        default=paths.analysis('crack_coordinates'),
    )
    parser.add_argument(
        "--real-path-features", type=Path,
        default=paths.analysis('crack_mask_features')/'path_features.csv',
    )
    parser.add_argument(
        "--real-turn-angles", type=Path,
        default=paths.analysis('crack_mask_features')/'turn_angles.csv',
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=paths.analysis('generated_vs_real_path_morphology'),
    )
    parser.add_argument("--coordinate-scale", type=float, default=1000.0)
    parser.add_argument("--resample-step", type=float, default=5.0)
    return parser.parse_args()


def read_segments(path: Path, scale: float) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        points = [(float(first), float(second)) for first, second in POINT.findall(line)]
        if len(points) >= 3:
            segments.append(scale * np.asarray(points, dtype=float))
    return segments


def main_compare() -> None:
    args = parse_args_compare()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real = {
        "tortuosity": read_float_column(args.real_path_features, "tortuosity"),
        "turn_angle_deg": read_float_column(args.real_turn_angles, "turn_angle_deg"),
        "nonpositive_goal_progress_fraction": read_float_column(
            args.real_path_features, "nonpositive_goal_progress_fraction"
        ),
    }
    scales = {name: robust_scale(values) for name, values in real.items()}

    feature_rows: list[dict[str, object]] = []
    tortuosities: list[float] = []
    angles: list[float] = []
    progress_fractions: list[float] = []
    files = sorted(args.generated_dir.glob("*_crack_*.txt"))
    for path in files:
        for segment_index, segment in enumerate(
            read_segments(path, args.coordinate_scale), start=1
        ):
            features = path_features(segment, args.resample_step)
            local_angles = np.asarray(features["turn_angles"], dtype=float)
            tortuosity = float(features["tortuosity"])
            progress = float(features["nonpositive_goal_progress_fraction"])
            if not np.isfinite(tortuosity) or not np.isfinite(progress):
                continue
            tortuosities.append(tortuosity)
            angles.extend(map(float, local_angles))
            progress_fractions.append(progress)
            feature_rows.append(
                {
                    "file": path.name,
                    "segment": segment_index,
                    "points": len(segment),
                    "tortuosity": tortuosity,
                    "median_turn_angle_deg": (
                        float(np.median(local_angles)) if len(local_angles) else ""
                    ),
                    "p95_turn_angle_deg": (
                        float(np.quantile(local_angles, 0.95)) if len(local_angles) else ""
                    ),
                    "nonpositive_goal_progress_fraction": progress,
                }
            )

    generated = {
        "tortuosity": np.asarray(tortuosities, dtype=float),
        "turn_angle_deg": np.asarray(angles, dtype=float),
        "nonpositive_goal_progress_fraction": np.asarray(
            progress_fractions, dtype=float
        ),
    }
    summary_rows: list[dict[str, object]] = []
    normalised_distances: list[float] = []
    for metric in real:
        real_summary = summarise(real[metric])
        generated_summary = summarise(generated[metric])
        distance = (
            wasserstein_distance_1d(real[metric], generated[metric]) / scales[metric]
        )
        normalised_distances.append(distance)
        summary_rows.append(
            {
                "metric": metric,
                "real_count": real_summary["count"],
                "real_mean": real_summary["mean"],
                "real_q05": real_summary["q05"],
                "real_median": real_summary["median"],
                "real_q95": real_summary["q95"],
                "generated_count": generated_summary["count"],
                "generated_mean": generated_summary["mean"],
                "generated_q05": generated_summary["q05"],
                "generated_median": generated_summary["median"],
                "generated_q95": generated_summary["q95"],
                "normalised_wasserstein_distance": distance,
            }
        )

    write_csv_calib(output_dir / "generated_path_features.csv", feature_rows)
    write_csv_calib(output_dir / "distribution_comparison.csv", summary_rows)
    metadata = {
        "generated_coordinate_files": len(files),
        "retained_generated_segments": len(feature_rows),
        "coordinate_scale_to_pixel_equivalent": args.coordinate_scale,
        "resample_step": args.resample_step,
        "mean_normalised_wasserstein_distance": float(np.mean(normalised_distances)),
        "note": "Generated text coordinates had already undergone the production refinement step and were not smoothed again.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    for row in summary_rows:
        print(
            "{metric}: real median={real_median:.6f}, generated median={generated_median:.6f}, "
            "real q95={real_q95:.6f}, generated q95={generated_q95:.6f}, "
            "distance={normalised_wasserstein_distance:.6f}".format(**row)
        )


# --------------------------------------------------------------------------
# Generate clean 512 px procedural wall layouts for far-range calibration.
# Originally generate_procedural_far_layouts.py
# --------------------------------------------------------------------------

#!/usr/bin/env python3


def parse_args_layouts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main_layouts() -> None:
    args = parse_args_layouts()
    sys.path.insert(0, str(paths.layout_code_dir()))
    from procedural_layout import (  # type: ignore
        build_wall_grid,
        choose_wall_type,
        generate_wall_bricks,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    audit: list[dict[str, object]] = []

    image_size = 512
    rows, columns = 10, 7
    scale_factor = 1.8
    resolution = 1000
    margin = 0.05
    anchor_x, anchor_z = 0.3, -0.02

    for index in range(1, args.count + 1):
        wall_type = choose_wall_type()
        bricks, _ = generate_wall_bricks(
            wall_type, num_rows=rows, num_cols=columns
        )
        minimum_x = min(item[0] - item[2] / 2 for item in bricks)
        minimum_z = min(item[1] - item[3] / 2 for item in bricks)
        bricks = [
            (x - minimum_x, z - minimum_z, width, height, angle)
            for x, z, width, height, angle in bricks
        ]
        minimum_x = min(item[0] - item[2] / 2 for item in bricks)
        maximum_x = max(item[0] + item[2] / 2 for item in bricks)
        minimum_z = min(item[1] - item[3] / 2 for item in bricks)

        grid = np.flipud(
            build_wall_grid(bricks, resolution=resolution, margin=margin)
        )
        height, width = grid.shape
        crop_pixels = int(0.512 * scale_factor * resolution)
        left = int((anchor_x - (minimum_x - margin)) * resolution)
        bottom = int(
            height - 1 - (anchor_z - (minimum_z - margin)) * resolution
        )
        top = bottom - crop_pixels + 1
        side = min(crop_pixels, height, width)
        top = min(max(0, top), height - side)
        left = min(max(0, left), width - side)
        crop = grid[top:top + side, left:left + side]
        layout = np.asarray(
            Image.fromarray((crop * 255).astype(np.uint8)).resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
        ) > 127

        rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb[layout] = (0, 255, 0)
        filename = f"procedural_far_{index:02d}.png"
        Image.fromarray(rgb).save(output_dir / filename)

        component_count = cv2.connectedComponents(
            layout.astype(np.uint8), connectivity=8
        )[0] - 1
        audit.append(
            {
                "map": filename,
                "wall_type": wall_type,
                "source_bricks": len(bricks),
                "retained_components": component_count,
                "brick_fraction": float(np.mean(layout)),
                "crop_scale": scale_factor,
                "rows": rows,
                "columns": columns,
            }
        )

    with (output_dir / "layout_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(f"Generated {len(audit)} layouts in {output_dir}")


# --------------------------------------------------------------------------
# Make deterministic mask-reconstruction previews and ordered crack polylines.
# Originally preview_crack_masks.py
# --------------------------------------------------------------------------

def estimate_scale(rgb, minimum_band=24):
    """Image-scale proxies from connected intact-brick regions and crack width."""
    green = np.all(rgb == (0, 255, 0), axis=2)
    red = np.all(rgb == (255, 0, 0), axis=2)
    crack = np.all(rgb == (255, 255, 0), axis=2)
    def areas(mask):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        return sizes[sizes >= 25]
    ga = areas(green)
    fallback = areas(red) if not len(ga) else ga
    mean_area = float(fallback.mean()) if len(fallback) else 12000.0
    radius = cv2.distanceTransform(crack.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    width = 2*float(np.percentile(radius[crack], 90)) if crack.any() else 0.0
    # Keep the accepted fine-crack setting; strengthen larger masks with moderate
    # cracks and all wide-crack masks. Thresholds are preview heuristics only.
    active = bool(width > 12 or (mean_area > 30000 and width > 8))
    band = int(np.ceil(max(minimum_band, 0.25*np.sqrt(mean_area), 1.4*width))) if active else minimum_band
    band = max(minimum_band, min(band, int(min(green.shape)*0.3)))
    return dict(green_component_count=len(ga), green_mean_area_px2=float(ga.mean()) if len(ga) else None,
                area_scale_source='green components' if len(ga) else ('red components fallback' if len(fallback) else 'default'),
                reference_area_px2=mean_area, crack_width_proxy_px=width,
                adaptive_active=active, effective_repair_band_px=band)


def strengthen_red_bricks(restored, baseline, original_red, crack, green, band_px=24, adaptive=False):
    """Fill crack-adjacent notches inside each reconstructed red component hull.

    This is a preview heuristic, not an inferred brick instance ground truth.
    Eroded interiors separate narrow connections across mortar joints. Hulls
    use original red pixels assigned to each surviving interior seed.
    """
    red = np.all(restored == (255, 0, 0), axis=2)
    near_crack = cv2.distanceTransform((~crack).astype(np.uint8), cv2.DIST_L2,
                                      cv2.DIST_MASK_PRECISE) <= band_px if crack.any() else crack.copy()
    seeds = cv2.erode(red.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    repair = np.zeros(red.shape, dtype=np.uint8)
    groups = []
    if seeds.any():
        _, labels = cv2.distanceTransformWithLabels(1-seeds, cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_CCOMP)
        for label in np.unique(labels[original_red]):
            ys, xs = np.where(original_red & (labels == label))
            if len(xs) < 25:
                continue
            points = np.column_stack([xs, ys]).astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.drawContours(repair, [hull], -1, 1, cv2.FILLED)
            groups.append((hull, cv2.boundingRect(hull)))
    # Rejoin separated pieces occupying the same brick course when the gap is
    # predominantly crack-adjacent and contains substantial yellow evidence.
    for i, (ha, (xa, ya, wa, hta)) in enumerate(groups):
        for hb, (xb, yb, wb, htb) in groups[i+1:]:
            overlap = max(0, min(ya+hta, yb+htb)-max(ya, yb))
            horizontal_gap = max(xa, xb)-min(xa+wa, xb+wb)
            overlap_ratio = overlap / (min(hta, htb) if adaptive else max(hta, htb))
            if overlap_ratio < (0.5 if adaptive else 0.65) or horizontal_gap > 2*band_px:
                continue
            merged = np.zeros_like(repair)
            cv2.drawContours(merged, [cv2.convexHull(np.concatenate([ha, hb]))], -1, 1, cv2.FILLED)
            separate = np.zeros_like(repair)
            cv2.drawContours(separate, [ha, hb], -1, 1, cv2.FILLED)
            gap = merged.astype(bool) & ~separate.astype(bool)
            size = int(gap.sum())
            if size < 1 or size > (0.5 if adaptive else 0.25)*int(merged.sum()) or green[gap].any():
                continue
            if near_crack[gap].mean() >= (0.7 if adaptive else 0.8) and crack[gap].mean() >= (0.1 if adaptive else 0.2):
                repair[gap] = 1
    added = repair.astype(bool) & near_crack & ~original_red & ~green
    result = baseline.copy()
    result[added] = (255, 0, 0)
    return result


def reconstruct(rgb, red_strength=3.0, repair_band_px=24, adaptive=False):
    red = np.all(rgb == (255, 0, 0), axis=2)
    green = np.all(rgb == (0, 255, 0), axis=2)
    black = np.all(rgb == (0, 0, 0), axis=2)
    crack = np.all(rgb == (255, 255, 0), axis=2)
    if not np.all(red | green | black | crack):
        raise ValueError('Unexpected palette: inspect labels before processing.')
    if crack.any() and not (red.any() or green.any() or black.any()):
        raise ValueError('No brick or background evidence for reconstruction.')
    dr = cv2.distanceTransform((~red).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if red.any() else np.full(red.shape, np.inf)
    dg = cv2.distanceTransform((~green).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if green.any() else np.full(green.shape, np.inf)
    db = cv2.distanceTransform((~black).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if black.any() else np.full(black.shape, np.inf)
    # Prefer broken-brick continuation within the yellow overlay for this v2 preview.
    dbrick = np.minimum(dr / red_strength, dg)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nr = cv2.filter2D((red | green).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    nb = cv2.filter2D(black.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    fill_brick = crack & ((dbrick < db) | ((dbrick == db) & (nr > nb)))
    restored = rgb.copy()
    restored[crack] = (0, 0, 0)
    restored[fill_brick & (dr <= dg)] = (255, 0, 0)
    restored[fill_brick & (dg < dr)] = (0, 255, 0)
    unweighted = np.minimum(dr, dg)
    base_fill = crack & ((unweighted < db) | ((unweighted == db) & (nr > nb)))
    baseline = rgb.copy()
    baseline[crack] = 0
    baseline[base_fill & (dr <= dg)] = (255, 0, 0)
    baseline[base_fill & (dg < dr)] = (0, 255, 0)
    restored = strengthen_red_bricks(restored, baseline, red, crack, green, repair_band_px, adaptive)
    occupancy = np.any(restored != 0, axis=2)
    layout = np.zeros_like(rgb)
    layout[occupancy] = (255, 0, 0)
    return restored, layout, crack, occupancy


def thin_crack(crack):
    """Vectorised Zhang-Suen thinning with a zero-padded image boundary."""
    a = np.pad(crack.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i+1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if phase == 0:
                permitted = (p[0]*p[2]*p[4] == 0) & (p[2]*p[4]*p[6] == 0)
            else:
                permitted = (p[0]*p[2]*p[6] == 0) & (p[0]*p[4]*p[6] == 0)
            remove = (a[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & permitted
            if remove.any():
                a[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1] > 0


def trace_curves(crack):
    """Partition a one-pixel skeleton graph into ordered maximal polylines.

    Split at endpoints/junctions; also retain loops and isolated pixels.
    Suppress diagonal shortcut edges when an orthogonal connection exists.
    All skeleton pixels and all retained graph edges are covered, without
    smoothing, pruning or artificial connections across disconnected cracks.
    """
    skeleton = thin_crack(crack)
    points = {tuple(p) for p in np.argwhere(skeleton).tolist()}
    graph = {}
    for y, x in sorted(points):
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (dy == dx == 0) or q not in points:
                    continue
                if dy and dx and ((y + dy, x) in points or (y, x + dx) in points):
                    continue
                neighbours.append(q)
        graph[(y, x)] = sorted(neighbours)
    visited = set()
    paths = []

    def edge(a, b):
        return tuple(sorted((a, b)))

    def walk(start, nxt):
        path = [start, nxt]
        visited.add(edge(start, nxt))
        prev, current = start, nxt
        while len(graph[current]) == 2:
            following = next(p for p in graph[current] if p != prev)
            if edge(current, following) in visited:
                break
            visited.add(edge(current, following))
            path.append(following)
            prev, current = current, following
        return path

    for p, neighbours in graph.items():
        if not neighbours:
            paths.append([p])
        elif len(neighbours) != 2:
            for q in neighbours:
                if edge(p, q) not in visited:
                    paths.append(walk(p, q))
    for p, neighbours in graph.items():
        for q in neighbours:
            if edge(p, q) not in visited:
                paths.append(walk(p, q))
    assert len(visited) == sum(map(len, graph.values())) // 2
    assert {p for path in paths for p in path} == points
    paths.sort(key=lambda p: (-len(p), p[0]))
    curves = []
    for i, path in enumerate(paths, 1):
        xy = [[x, y] for y, x in path]
        delta = np.diff(np.asarray(xy, dtype=float), axis=0)
        curves.append(dict(curve_id=i, closed=len(path) > 2 and path[0] == path[-1],
                           length_px=float(np.linalg.norm(delta, axis=1).sum()),
                           points_xy=xy))
    return skeleton, curves


def font(size):
    for family in ('Arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(family, size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_contact_sheet(out, batch, first_index):
    sheet = Image.new('RGB', (900, sum(p.height for p in batch)), 'white')
    y = 0
    for picture in batch:
        sheet.paste(picture, (0, y))
        y += picture.height
    sheet.save(out / 'contact_sheets' / f'{first_index:03d}-{first_index+len(batch)-1:03d}.jpg', quality=92)


def main_preview():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=paths.test_annotations / 'masks_image')
    parser.add_argument('--output', type=Path, default=paths.analysis('mask_preview'))
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--red-strength', type=float, default=3.0)
    parser.add_argument('--repair-band-px', type=int, default=24)
    parser.add_argument('--fixed-scale', action='store_true', help='Use the V2 fixed-scale reconstruction rule.')
    args = parser.parse_args()
    if args.red_strength < 1 or args.repair_band_px < 0:
        raise ValueError('Red strength must be >= 1 and repair band must be >= 0.')
    files = sorted(args.source.glob('*.png'), key=lambda p: p.name)
    if not 1 <= args.count <= len(files):
        raise ValueError('Count must be between one and the available image count.')
    selected = [files[i] for i in np.linspace(0, len(files)-1, args.count, dtype=int)]
    out = args.output.resolve()
    source = args.source.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Use an output directory separate from the source images.')
    folders = ['restored_colours', 'layout_red_black', 'brick_binary', 'crack_binary',
               'skeleton', 'curves', 'previews', 'contact_sheets']
    for folder in folders:
        (out / folder).mkdir(parents=True, exist_ok=True)
    rows, cards, thumbnails = [], [], []
    for number, path in enumerate(selected, 1):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        rgb = np.array(Image.open(path).convert('RGB'))
        # One source image has dark-yellow pixels around its yellow crack
        # boundary. Decode this exact observed colour in memory; retain original.
        dark_yellow = np.all(rgb == (32, 32, 0), axis=2)
        decoded = rgb.copy()
        decoded[dark_yellow] = (255, 255, 0)
        scale = estimate_scale(decoded, args.repair_band_px)
        if args.fixed_scale:
            scale.update(adaptive_active=False, effective_repair_band_px=args.repair_band_px)
        restored, layout, crack, occupancy = reconstruct(decoded, args.red_strength, scale['effective_repair_band_px'], scale['adaptive_active'])
        skeleton, curves = trace_curves(crack)
        h, w = crack.shape
        stem = path.stem
        original_bricks = np.all(rgb == (255, 0, 0), axis=2) | np.all(rgb == (0, 255, 0), axis=2)
        assert np.array_equal(restored[original_bricks], rgb[original_bricks])
        changed_black = np.all(rgb == 0, axis=2) & np.any(restored != 0, axis=2)
        distance_to_crack = cv2.distanceTransform((~crack).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if crack.any() else np.full(crack.shape, np.inf)
        assert np.all(distance_to_crack[changed_black] <= scale['effective_repair_band_px'])
        assert not np.any(np.all(restored == (255, 255, 0), axis=2))
        assert np.all(crack[skeleton])
        for folder, array in [('restored_colours', restored), ('layout_red_black', layout),
                              ('brick_binary', occupancy.astype(np.uint8)*255),
                              ('crack_binary', crack.astype(np.uint8)*255),
                              ('skeleton', skeleton.astype(np.uint8)*255)]:
            Image.fromarray(array).save(out / folder / path.name)
        metadata = dict(source_file=path.name, source_sha256=before, width=w, height=h,
                        coordinates='Pixel centres: integer (x,y), origin top-left, x right, y down.',
                        curve_representation='Ordered skeleton polylines, split at junctions. No pruning or smoothing.',
                        curves=curves)
        if dark_yellow.any():
            metadata['palette_decoding'] = dict(source_rgb=[32, 32, 0], decoded_rgb=[255, 255, 0],
                                               pixels=int(dark_yellow.sum()),
                                               basis='Visual inspection: dark-yellow edge around yellow crack; treated as crack in this output.')
        (out / 'curves' / f'{stem}.json').write_text(json.dumps(metadata, separators=(',', ':')), encoding='utf-8')
        with (out / 'curves' / f'{stem}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['curve_id', 'point_index', 'x', 'y'])
            for curve in curves:
                writer.writerows((curve['curve_id'], i, x, y) for i, (x, y) in enumerate(curve['points_xy']))
        overlay = Image.fromarray(layout.copy())
        draw = ImageDraw.Draw(overlay)
        for curve in curves:
            pts = [tuple(p) for p in curve['points_xy']]
            if len(pts) > 1:
                draw.line(pts, fill=(0, 255, 255), width=2)
            else:
                draw.point(pts[0], fill=(0, 255, 255))
        preview = Image.new('RGB', (w*3, h+66), 'white')
        draw = ImageDraw.Draw(preview)
        draw.text((12, 6), f'{number:02d}  {path.name}', font=font(17), fill='black')
        for col, (label, picture) in enumerate([
            ('Original mask', Image.fromarray(rgb)),
            ('Crack removed: original classes', Image.fromarray(restored)),
            ('Brick layout + crack centreline (cyan)', overlay),
        ]):
            draw.text((col*w+12, 36), label, font=font(17), fill='black')
            preview.paste(picture, (col*w, 66))
        preview.save(out / 'previews' / path.name)
        thumbnails.append(preview.resize((900, round(preview.height*900/preview.width))))
        if len(thumbnails) == 10:
            save_contact_sheet(out, thumbnails, number-9)
            thumbnails.clear()
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(index=number, filename=path.name, width=w, height=h,
                         **scale,
                         dark_yellow_pixels_decoded=int(dark_yellow.sum()),
                         crack_pixels=int(crack.sum()), skeleton_pixels=int(skeleton.sum()),
                         original_black_pixels_filled=int(changed_black.sum()),
                         curves=len(curves), isolated_points=sum(len(c['points_xy']) == 1 for c in curves),
                         short_curves_under_5px=sum(c['length_px'] < 5 for c in curves),
                         source_sha256=before))
        cards.append(f'<section><h2>{number:02d}. {html.escape(path.name)}</h2>'
                     f'<a href="previews/{path.name}"><img loading="lazy" src="previews/{path.name}"></a>'
                     f'<p><a href="restored_colours/{path.name}">Reconstructed layout, red and green classes kept</a> · '
                     f'<a href="layout_red_black/{path.name}">Crack-free wall, red and black</a> · '
                     f'<a href="curves/{stem}.json">Curves, JSON</a> · '
                     f'<a href="curves/{stem}.csv">Coordinates, CSV</a> · '
                     f'{len(curves)} polylines, including short branches and isolated points · '
                     f'mean green area {scale["green_mean_area_px2"] if scale["green_mean_area_px2"] is not None else "no green region"} px² · '
                     f'repair band {scale["effective_repair_band_px"]} px</p></section>')
        if number % 50 == 0 or number == args.count:
            print(f'Processed {number}/{args.count}', flush=True)
    if thumbnails:
        save_contact_sheet(out, thumbnails, len(selected)-len(thumbnails)+1)
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = dict(available_images=len(files), processed_images=len(selected), preview_images=len(selected),
                  selection='All PNG source files in lexicographic order.' if len(selected) == len(files) else 'Evenly spaced indices in lexicographically sorted PNG filenames; no performance-based selection.',
                  source=str(source), output=str(out),
                  palette_exceptions=[dict(filename=r['filename'], dark_yellow_pixels_decoded=r['dark_yellow_pixels_decoded'])
                                      for r in rows if r['dark_yellow_pixels_decoded']],
                  reconstruction='Reconstruction with an adaptive repair band from the mean green-component area and a crack-width proxy. Coarse cases permit unequal same-course fragments and larger crack gaps; fine cases use the narrower settings. Original red and green labels remain unchanged.',
                  adaptive_scale_parameters=dict(enabled=not args.fixed_scale, minimum_component_area_px2=25,
                    crack_width_proxy='twice the 90th percentile of distance-to-crack-boundary over yellow pixels',
                    activation='width > 12 OR (mean area > 30000 AND width > 8)',
                    band_formula='ceil(max(minimum_band, 0.25*sqrt(mean area), 1.4*width)), capped at 0.3*min(image height,width)',
                    coarse_overlap='>=0.5 of shorter fragment height', coarse_max_gap_fraction=0.5,
                    coarse_min_vicinity_fraction=0.7, coarse_min_yellow_fraction=0.1),
                  preview_parameters=dict(red_strength=args.red_strength, repair_band_px=args.repair_band_px,
                                          seed_erosion_diameter_px=15, minimum_original_red_pixels_per_hull=25, status='Visual-review heuristics, not calibrated experimental parameters'),
                  fragment_join_parameters=dict(min_vertical_overlap=0.65, max_horizontal_gap_px=2*args.repair_band_px,
                                                max_hull_gap_fraction=0.25, min_crack_vicinity_fraction=0.8,
                                                min_original_yellow_fraction=0.2, forbid_original_green=True),
                  verified=['Source hashes unchanged', 'Original red and green pixels unchanged',
                            'Changes to original black pixels limited to the recorded crack vicinity',
                            'No yellow pixels remain in reconstruction', 'Every skeleton pixel lies inside the original crack',
                            'All skeleton pixels and graph edges represented in curves'],
                  totals={key: sum(r[key] for r in rows) for key in ['crack_pixels', 'skeleton_pixels', 'curves', 'isolated_points', 'original_black_pixels_filled']})
    (out / 'run_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    introduction = f'''<h1>Mask reconstruction and crack-curve preview for {len(selected)} masks, scaled by brick size and crack width</h1>
<p>Each row shows, from left to right, the original mask, the class map after the yellow crack is removed, and the red and black wall with the crack centreline in cyan. 
Click an image to see it at full size. The second column keeps the red broken-brick and green intact-brick classes, while the crack-free layout merges both into red.</p>
<p>The yellow region extends the red brick fragments. An eroded brick interior then separates the units on either side of a narrow connection, and the black gap near the crack is filled from the convex hull of the original red pixels of each group. 
The mean area of a single green connected region estimates the brick scale in the image, and this is combined with the yellow crack width to widen the repair band. Close-range wide cracks may join red fragments of different sizes, while narrow cracks keep the V2 setting. 
Red fragments separated by the crack within the same brick course are merged when the gap carries enough yellow evidence. 
The original red and green brick pixels are kept, and only the black crack neighbourhood is filled. The parameters are recorded in run_summary.json. 
This is a shape estimate for manual inspection; brick corners and the mortar joints between units should be checked for over-filling.</p>
<p>Each curve is a centreline polyline in connection order, split at every junction, and broken segments, short branches, closed loops and isolated points are retained. 
Coordinates are in source-image pixels (x, y) with (0, 0) at the top left, x to the right and y downwards. The original crack width is stored in crack_binary.</p>
<p>{len(files)} source files are available and {len(selected)} were processed in filename order. This batch is a preprocessing result.</p>'''
    (out / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Mask reconstruction</title><style>body{font:16px Arial,sans-serif;max-width:1600px;margin:24px auto;padding:0 16px;background:#eee}'
        'section{background:white;padding:12px;margin:22px 0}h2{font-size:16px}img{width:100%}p{line-height:1.6}a{color:#125b9b}</style>'
        + introduction + ''.join(cards) + '</html>', encoding='utf-8')
    (out / 'README.md').write_text(f'''# Mask reconstruction

Open `index.html` to inspect all {len(selected)} comparisons. `contact_sheets/` has {(len(selected)+9)//10} overview sheets.

- `restored_colours/`: yellow pixels reconstructed and crack-adjacent black notches in red bricks repaired. Original red/green pixels unchanged.
- `layout_red_black/`: all bricks red, mortar/background black; no crack overlay or damaged/intact distinction.
- `brick_binary/`: brick occupancy 255, mortar/background 0. Its complement is the second layout channel.
- `crack_binary/`: original yellow crack target as 255, background 0.
- `skeleton/`: one-pixel crack centreline as 255.
- `curves/`: JSON ordered polylines and CSV columns curve_id, point_index, x, y.
- `previews/`: original, reconstructed original classes, unified layout with cyan centrelines.
- `manifest.csv`: selected filenames, dimensions, counts and source hashes.
- `run_summary.json`: actual counts, reconstruction rule and automated checks.

Palette exception: exact RGB (32,32,0), observed as a dark-yellow crack edge in one source
image, is decoded to yellow (255,255,0) in memory. This inferred class assignment affects
the crack target and curve for that image. The source is retained; affected pixels are
counted in manifest.csv, with the exception also recorded in run_summary.json and its
curve JSON. Other unexpected colours cause an error rather than silent conversion.

Coordinates refer to original pixel centres, origin top-left, x right, y down. Branch junctions
can occur in multiple polylines. Closed loops repeat their starting point. Isolated pixels
are retained as one-point paths. No short branches are removed and no disconnected cracks
are joined. Polyline count is not the number of physical cracks. The original binary target
preserves crack width; skeletonisation does not preserve the full crack contour.

The reconstruction estimates the mean area of green connected components of at least 25 pixels. This is an
image-scale proxy, not measured camera distance or guaranteed brick-instance area. If no
green component exists, red component area is used; if neither exists, the fallback is 12000.
Crack width proxy = twice the 90th percentile of interior distance over yellow pixels.
Adaptive mode activates for width > 12, or area > 30000 and width > 8. The repair band is
ceil(max(minimum band, 0.25*sqrt(area), 1.4*width)), capped at 0.3 of the shorter image side.
All per-image measurements and effective bands are recorded in manifest.csv.
Coarse fragments require 0.5 overlap relative to the shorter height, allow a gap up to 50
percent of the combined hull, and require 70 percent crack vicinity and 10 percent yellow
coverage. The green-pixel veto remains. Otherwise the V2 parameters below apply.
Use --fixed-scale to reproduce V2. All of these thresholds are visual-review heuristics.

This preview divides red distance by --red-strength (default 3) when assigning yellow
pixels provisionally to bricks or black background to identify brick interiors. Green distance remains unweighted. Equal distances
use original eight-neighbour brick/black counts, then black if still tied.
Original-class previews use the nearer red/green colour for restored brick pixels, red on ties.
Then red regions are eroded using a 15-pixel elliptical kernel to separate narrow bridges
across joints. Original red pixels are assigned to their nearest surviving interior seed
(OpenCV labelled distance transform). Each seed group's convex hull supplies candidate repairs
within --repair-band-px (default 24) of the original yellow crack. Groups with fewer than
25 original red pixels are excluded. This can replace original black pixels, but never original
green/red pixels. Outside hull repairs the final output retains the V1 unweighted nearest-class
reconstruction, so the provisional weighting cannot extend red into unsupported joint regions.
Separated fragments may be joined when their vertical overlap is at least 0.65 of the larger
height and horizontal gap is at most twice the repair band. The joint hull's new gap must
occupy at most 25 percent of its area, contain no green pixels, have at least 80 percent
within the crack vicinity and at least 20 percent original yellow pixels. This is a geometric
same-course heuristic; yellow cracks in real joints between two red bricks remain ambiguous.
Thin regions without surviving seeds require review. Parameters are visual-review
heuristics, not calibrated experimental parameters. No manuscript or historical model is changed.
If a class is absent, the available eligible class is used. Hidden labels cannot be uniquely recovered; wide
cracks, mortar continuity and brick corners need visual review. This is preprocessing,
not a validation result for any U-Net. Source count is {len(files)}, processed count is {len(selected)}.

Run from workspace root: `python scripts/preview_crack_masks.py --count {args.count} --output "{args.output.as_posix()}"`.
Dependencies: numpy, Pillow, opencv-python. Zhang-Suen thinning is implemented in NumPy.
''', encoding='utf-8')
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------
# Validate saved mask products, source integrity and accepted samples.
# Originally verify_reconstructed_masks.py
# --------------------------------------------------------------------------

def main_verify():
    parser = argparse.ArgumentParser(
        description='Validate saved mask products, source integrity and accepted samples.')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'],
                        help='Which partition the colour masks came from. Default: test.')
    parser.add_argument('--source', type=Path, default=None,
                        help='The colour masks that were reconstructed. '
                             'Default: <partition annotations>/masks_image.')
    parser.add_argument('--output', type=Path, default=None,
                        help='The reconstruction to validate. Default: the configured train masks.')
    parser.add_argument('--accepted', type=Path, default=None,
                        help='The visually reviewed subset. Default: analysis/mask_preview.')
    parser.add_argument('--expect-accepted', type=int, default=50,
                        help='How many accepted samples to require. The original script required 50.')
    args = parser.parse_args()
    annotations = (paths.test_annotations if args.partition == 'test'
                   else paths.workspace / 'masks' / (args.partition + '_annotations'))
    source = args.source if args.source is not None else annotations / 'masks_image'
    output = args.output if args.output is not None else paths.train_masks
    accepted = args.accepted if args.accepted is not None else paths.analysis('mask_preview')
    rows = list(csv.DictReader((output / 'manifest.csv').open(encoding='utf-8-sig')))
    names = {p.name for p in source.glob('*.png')}
    assert len(rows) == len(names)
    assert {r['filename'] for r in rows} == names
    folders = ['restored_colours', 'layout_red_black', 'brick_binary', 'crack_binary', 'skeleton', 'previews']
    for folder in folders:
        assert {p.name for p in (output / folder).glob('*.png')} == names
    for suffix in ['.json', '.csv']:
        assert {p.stem for p in (output / 'curves').glob('*' + suffix)} == {Path(n).stem for n in names}
    curve_count = no_crack_count = 0
    for row in rows:
        name = row['filename']
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == row['source_sha256']
        original = np.array(Image.open(source / name).convert('RGB'))
        restored = np.array(Image.open(output / 'restored_colours' / name))
        layout = np.array(Image.open(output / 'layout_red_black' / name))
        brick = np.array(Image.open(output / 'brick_binary' / name)) > 0
        target = np.array(Image.open(output / 'crack_binary' / name)) > 0
        skeleton = np.array(Image.open(output / 'skeleton' / name)) > 0
        assert restored.shape == layout.shape == original.shape
        red = np.all(original == [255, 0, 0], axis=2)
        green = np.all(original == [0, 255, 0], axis=2)
        assert np.array_equal(restored[red | green], original[red | green])
        assert np.all(np.all(restored == [255, 0, 0], axis=2) |
                      np.all(restored == [0, 255, 0], axis=2) | np.all(restored == 0, axis=2))
        expected_target = np.all(original == [255, 255, 0], axis=2) | np.all(original == [32, 32, 0], axis=2)
        assert np.array_equal(target, expected_target)
        assert np.array_equal(brick, np.any(restored != 0, axis=2))
        assert np.all(layout[brick] == [255, 0, 0]) and np.all(layout[~brick] == 0)
        assert np.all(target[skeleton])
        metadata = json.loads((output / 'curves' / (Path(name).stem + '.json')).read_text(encoding='utf-8'))
        assert metadata['source_sha256'] == row['source_sha256']
        traced = np.zeros_like(target)
        for curve in metadata['curves']:
            xy = np.asarray(curve['points_xy'])
            assert xy.ndim == 2 and xy.shape[1] == 2 and len(xy) > 0
            assert np.all((xy[:, 0] >= 0) & (xy[:, 0] < target.shape[1]))
            assert np.all((xy[:, 1] >= 0) & (xy[:, 1] < target.shape[0]))
            assert np.all(np.max(np.abs(np.diff(xy, axis=0)), axis=1) <= 1)
            traced[xy[:, 1], xy[:, 0]] = True
        assert np.array_equal(traced, skeleton)
        assert len(metadata['curves']) == int(row['curves'])
        curve_count += len(metadata['curves'])
        no_crack_count += not target.any()
    accepted_names = [p.name for p in (accepted / 'restored_colours').glob('*.png')]
    assert len(accepted_names) == args.expect_accepted
    for name in accepted_names:
        for folder in folders[:-1]:
            assert (accepted / folder / name).read_bytes() == (output / folder / name).read_bytes()
        for extension in ['.json', '.csv']:
            filename = Path(name).stem + extension
            assert (accepted / 'curves' / filename).read_bytes() == (output / 'curves' / filename).read_bytes()
    report = dict(status='passed', source_images=len(names), completed_images=len(rows),
                  accepted_samples_identical=len(accepted_names), ordered_polylines=curve_count,
                  images_without_decoded_crack_pixels=int(no_crack_count),
                  contact_sheets=len(list((output / 'contact_sheets').glob('*.jpg'))),
                  checks=['Complete filename coverage in six image directories and both curve formats',
                          'All source SHA256 hashes unchanged', 'Original red/green pixels unchanged',
                          'Output palette and occupancy agreement', 'Targets match decoded source labels',
                          'Saved curve coordinates reconstruct the saved skeletons',
                          f'All {len(accepted_names)} accepted masks, targets, skeletons and coordinates are byte-identical'])
    (output / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------
# Rasterise the YOLO annotations of one dataset partition.
# Originally draw_test_masks.py
# --------------------------------------------------------------------------

SOURCE_DRAW = paths.dataset_root
OUT_DRAW = paths.test_annotations
PALETTE_DRAW = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0], [255, 255, 0]], dtype=np.uint8)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_sheet(batch, end, out=None):
    sheet = Image.new('RGB', (960, sum(im.height for im in batch)), 'white')
    top = 0
    for im in batch:
        sheet.paste(im, (0, top))
        top += im.height
    out = OUT_DRAW if out is None else out
    sheet.save(out / 'contact_sheets' / f'{end-len(batch)+1:03d}-{end:03d}.jpg', quality=92)


def main_draw():
    parser = argparse.ArgumentParser(
        description='Rasterise the YOLO annotations of one dataset partition.')
    parser.add_argument('--partition', default='test', choices=['train', 'valid', 'test'],
                        help='Which dataset_root partition to rasterise. Default: test.')
    parser.add_argument('--output', type=Path, default=None,
                        help='Where to write. The test partition defaults to test_annotations; '
                             'another partition to <workspace>/masks/<partition>_annotations.')
    parser.add_argument('--expect-images', type=int, default=None,
                        help='Assert this image count. The original script asserted 150.')
    args = parser.parse_args()
    partition = args.partition
    out = args.output if args.output is not None else (
        OUT_DRAW if partition == 'test'
        else paths.workspace / 'masks' / (partition + '_annotations'))
    yaml_text = (SOURCE_DRAW / 'data.yaml').read_text(encoding='utf-8')
    if "names: ['brick', 'broken_brick', 'crack']" not in yaml_text:
        raise ValueError('Inspect the class mapping before rasterising.')
    images = sorted(p for p in (SOURCE_DRAW / partition / 'images').iterdir()
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})
    labels = {p.stem: p for p in (SOURCE_DRAW / partition / 'labels').glob('*.txt')}
    total = len(images)
    assert total == len(labels)
    assert len({p.stem for p in images}) == total and {p.stem for p in images} == labels.keys()
    if args.expect_images is not None and total != args.expect_images:
        raise ValueError('Expected %d images, found %d.' % (args.expect_images, total))
    folders = ['masks_image', 'masks_native_640', 'class_ids_512', 'class_ids_native_640',
               'crack_binary', 'previews', 'contact_sheets']
    for folder in folders:
        (out / folder).mkdir(parents=True, exist_ok=True)
    label_font = font(18)
    rows, cards, batch = [], [], []
    totals = Counter()
    for index, path in enumerate(images, 1):
        label_path = labels[path.stem]
        image_hash, label_hash = sha(path), sha(label_path)
        with Image.open(path) as handle:
            original = handle.convert('RGB')
        width, height = original.size
        assert (width, height) == (640, 640)
        polygons = {i: [] for i in range(3)}
        collapsed = []
        for line_index, line in enumerate(label_path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if fields[0] not in {'0', '1', '2'} or len(fields) < 7 or (len(fields)-1) % 2:
                raise ValueError(f'Invalid polygon: {label_path.name}:{line_index}')
            coords = np.asarray(fields[1:], dtype=np.float64).reshape(-1, 2)
            if not np.all(np.isfinite(coords)) or np.any((coords < 0) | (coords > 1)):
                raise ValueError('Polygon coordinates must be finite and normalised.')
            # YOLO coordinates are fractions of width/height. Truncate the
            # nonnegative pixel positions; clip the inclusive right/bottom edge.
            xy = (coords * np.array([width, height])).astype(np.int32)
            xy[:, 0] = np.clip(xy[:, 0], 0, width-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, height-1)
            if len(np.unique(xy, axis=0)) < 3:
                collapsed.append(dict(line=line_index, yolo_class=int(fields[0]),
                                      distinct_pixel_vertices=len(np.unique(xy, axis=0))))
            polygons[int(fields[0])].append(xy)
        layers = []
        counts = {}
        for cls in range(3):
            layer = np.zeros((height, width), dtype=np.uint8)
            # Draw each polygon separately to union same-class overlaps rather
            # than applying even/odd filling jointly to multiple polygons.
            for poly in polygons[cls]:
                cv2.fillPoly(layer, [poly], 1, lineType=cv2.LINE_8)
            layers.append(layer.astype(bool))
            counts[f'class_{cls}_polygons'] = len(polygons[cls])
            totals[str(cls)] += len(polygons[cls])
        ids = np.zeros((height, width), dtype=np.uint8)
        for cls, layer in enumerate(layers):
            ids[layer] = cls+1
        native = PALETTE_DRAW[ids]
        ids512 = np.array(Image.fromarray(ids).resize((512, 512), Image.Resampling.NEAREST))
        rgb512 = PALETTE_DRAW[ids512]
        target = (ids512 == 3).astype(np.uint8)*255
        name = path.stem + '.png'
        products = {'masks_native_640': native, 'masks_image': rgb512,
                    'class_ids_native_640': ids, 'class_ids_512': ids512,
                    'crack_binary': target}
        for folder, array in products.items():
            Image.fromarray(array).save(out / folder / name)
        # Validate files after writing, not just the in-memory representation.
        saved_ids = np.array(Image.open(out / 'class_ids_512' / name))
        saved_rgb = np.array(Image.open(out / 'masks_image' / name))
        saved_target = np.array(Image.open(out / 'crack_binary' / name))
        assert np.array_equal(saved_rgb, PALETTE_DRAW[saved_ids])
        assert np.array_equal(saved_target > 0, saved_ids == 3)
        assert np.array_equal(ids == 3, layers[2])
        preview = Image.new('RGB', (1536, 568), 'white')
        draw = ImageDraw.Draw(preview)
        draw.text((10, 5), f'{index:03d}  {path.name}', fill='black', font=label_font)
        small_image = original.resize((512, 512), Image.Resampling.LANCZOS)
        overlay = np.array(small_image).copy()
        marked = ids512 > 0
        overlay[marked] = np.rint(0.55*overlay[marked]+0.45*rgb512[marked]).astype(np.uint8)
        for column, (title, im) in enumerate([
            ('Original test image', small_image),
            ('Annotation mask (512 x 512)', Image.fromarray(rgb512)),
            ('Annotation overlay', Image.fromarray(overlay)),
        ]):
            draw.text((column*512+10, 31), title, fill='black', font=label_font)
            preview.paste(im, (column*512, 56))
        preview.save(out / 'previews' / name)
        batch.append(preview.resize((960, 355)))
        if len(batch) == 10:
            save_sheet(batch, index, out)
            batch.clear()
        assert sha(path) == image_hash and sha(label_path) == label_hash
        rows.append(dict(index=index, image=path.name, label=label_path.name, mask=name,
                         source_width=width, source_height=height, output_width=512, output_height=512,
                         **counts, empty_annotation=not any(counts.values()),
                         collapsed_polygon_count=len(collapsed), collapsed_polygons=json.dumps(collapsed),
                         cross_class_overlap_pixels=int((sum(x.astype(np.uint8) for x in layers)>1).sum()),
                         brick_pixels_512=int((ids512 == 1).sum()), broken_brick_pixels_512=int((ids512 == 2).sum()),
                         crack_pixels_512=int((ids512 == 3).sum()), image_sha256=image_hash,
                         label_sha256=label_hash, mask_sha256=sha(out / 'masks_image' / name)))
        cards.append(f'<section><h2>{index:03d}. {html.escape(path.name)}</h2>'
                     f'<a href="previews/{name}"><img loading="lazy" src="previews/{name}"></a>'
                     f'<p><a href="masks_image/{name}">512 mask</a> · '
                     f'<a href="masks_native_640/{name}">640 native-size mask</a> · '
                     f'<a href="crack_binary/{name}">binary crack label</a></p></section>')
        if index % 30 == 0:
            print(f'Generated and checked {index}/{total}', flush=True)
    if batch:
        save_sheet(batch, len(images), out)
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='passed', source=str(SOURCE_DRAW / partition), output=str(out), image_count=total,
                  label_count=total, polygon_counts=dict(totals),
                  colour_mapping={'background': [0,0,0], 'brick': [0,255,0], 'broken_brick': [255,0,0], 'crack': [255,255,0]},
                  class_id_mapping={'background': 0, 'brick': 1, 'broken_brick': 2, 'crack': 3},
                  native_size=[640,640], model_size=[512,512],
                  rasterisation='Normalised xy * image width/height, truncation to integers, clipping at border, cv2.fillPoly per polygon.',
                  overlap_order='brick, then broken_brick, then crack; same-class polygons unioned',
                  resize='Native integer class map resized with nearest neighbour, then colour lookup.',
                  empty_label_images=sum(r['empty_annotation'] for r in rows),
                  images_without_crack_pixels=sum(r['crack_pixels_512'] == 0 for r in rows),
                  collapsed_polygons=[dict(label=r['label'], polygons=json.loads(r['collapsed_polygons']))
                                      for r in rows if r['collapsed_polygon_count']],
                  source_yaml_sha256=sha(SOURCE_DRAW/'data.yaml'),
                  checks=[f'{total} images paired with {total} labels', 'All polygon fields/classes/ranges valid',
                          'Saved RGB and binary masks match class IDs', 'Crack wins all overlaps',
                          'All source image and label hashes unchanged'],
                  scope=f'Rasterised existing {partition} annotations only. No predictions, reconstruction, training or performance evaluation.')
    (out / 'generation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    heading = f'{total} annotated {partition} masks'
    (out / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>' + heading + '</title>'
        '<style>body{font:16px Arial,sans-serif;background:#eee;max-width:1600px;margin:24px auto;padding:0 16px}'
        'section{background:white;padding:12px;margin:24px 0}h2{font-size:16px}img{width:100%}p{line-height:1.6}a{color:#125b9b}</style>'
        '<h1>' + heading + f'</h1><p>Original polygon annotations from the {partition} partition of the dataset. '
        'Green: intact brick. Red: broken brick. Yellow: crack. Black: background or unlabelled area. '
        'Left: source image. Centre: annotation mask. Right: overlay. Click an image to enlarge.</p>'
        '<p>Saved at the native 640x640 size and as a nearest-neighbour 512x512 version. Overlap priority: crack over broken brick over intact brick. '
        'This step draws the annotation masks only. No crack reconstruction or model prediction is performed.</p>' + ''.join(cards) + '</html>', encoding='utf-8')
    (out / 'README.md').write_text(f'''# {partition.capitalize()} annotation masks ({total} images)

Source: <dataset_root>/{partition}/images and {partition}/labels; classes are defined in data.yaml.
These are rasterised supplied annotations, not predicted masks.

- masks_image/: 512x512 RGB colour masks for the next preprocessing step.
- masks_native_640/: 640x640 RGB masks at the source image dimensions.
- class_ids_512/ and class_ids_native_640/: 0 background/unannotated, 1 brick, 2 broken brick, 3 crack.
- crack_binary/: 512x512 crack target, 255 crack and 0 elsewhere.
- previews/: original image, RGB mask and annotation overlay.
- contact_sheets/: {(total + 9) // 10} sheets, ten images each.
- index.html: all {total} previews with links to the generated masks.
- manifest.csv: source/output pairing, instance/pixel counts and source SHA256 hashes.
- generation_report.json: class mapping, rasterisation choices and checks.

RGB colours: intact brick green (0,255,0), broken brick red (255,0,0), crack yellow
(255,255,0), background/unannotated black (0,0,0). Black is not automatically verified mortar.
Draw each YOLO polygon separately, with intact brick first, broken brick second and crack last.
Polygons collapsing to fewer than three distinct native pixel vertices are still rasterised
by OpenCV as their resulting point/line; they are recorded in the manifest and report.
Normalised coordinates are multiplied by the native width/height, truncated and clipped.
Nearest-neighbour resizing of the native class map produces the 512x512 version; no smoothing.
This exports semantic masks; original instance polygons remain intact in the source labels.
No crack removal, input reconstruction, model training, prediction or evaluation has been run.

Regenerate with: python real_mask_analysis.py draw-test-masks --partition {partition}
Dependencies: numpy, Pillow, opencv-python.
''', encoding='utf-8')
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------
# Read the frozen evaluation outputs and create manuscript statistics and a figure.
# --------------------------------------------------------------------------

SOURCE_ASSETS=paths.prior_evaluation
OUT_ASSETS=paths.analysis('manuscript_assets')


def family(name):
    stem=re.split(r'_(?:jpg|jpeg|png)(?:[._]|$)',name,flags=re.I)[0]
    stem=re.sub(r'\.rf\..*$','',stem)
    if re.fullmatch(r'\d+(?:[-_]\d+)*',stem):return 'numeric_'+str(int(re.split('[-_]',stem)[0]))
    if re.fullmatch(r'(?:a_\d+|l\d+)_\d+',stem):return stem.rsplit('_',1)[0]
    return stem


def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main_assets():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    rows=read_csv(SOURCE_ASSETS/'per_image.csv')
    learned={r['filename']:r for r in rows if r['method']=='unet'}
    uniform={r['filename']:r for r in rows if r['method']=='no_unet'}
    names=sorted(learned)
    assert set(names)==set(uniform), 'per_image.csv is not paired across the two methods'
    source_rows=read_csv(paths.unet_run/'data_manifest.csv')
    source_families={family(r['filename']) for r in source_rows}
    retained=[n for n in names if family(n) not in source_families]
    summaries=[]
    for subset,selected in [('all',names),('no_detected_filename_family_overlap',retained)]:
        for method,data in [('uniform',uniform),('learned',learned)]:
            summaries.append(dict(subset=subset,method=method,images=len(selected),
                **{k:float(np.mean([float(data[n][k]) for n in selected])) for k in ['max_f1','mean_f1','completion']}))
    groups={}
    for i,n in enumerate(names):groups.setdefault(family(n),[]).append(i)
    grouped=list(groups.values());rng=np.random.default_rng(20260917)
    differences=np.array([[float(learned[n][k])-float(uniform[n][k]) for k in ['max_f1','mean_f1','completion']] for n in names])
    samples=[]
    for _ in range(10000):
        indices=np.concatenate([grouped[j] for j in rng.integers(0,len(grouped),len(grouped))])
        samples.append(differences[indices].mean(0))
    ordered=sorted((float(learned[n]['max_f1'])-float(uniform[n]['max_f1']),n) for n in names)
    cases=[ordered[round(q*(len(ordered)-1))][1] for q in [.1,.5,.9]]
    stats=dict(summary=summaries,paired_difference=differences.mean(0).tolist(),
        paired_percentile95=np.quantile(samples,[.025,.975],axis=0).tolist(),
        bootstrap=dict(repetitions=10000,seed=20260917,filename_family_clusters=len(grouped),
            columns=['max_f1','mean_f1','completion'],scope='Descriptive within-collection resampling; does not remove adaptive development bias'),
        family_sensitivity=dict(rule='Numeric prefix with leading zeros removed and underscore/hyphen crop suffixes grouped; a_/l prefixes grouped by final crop index',
            excluded_eligible_images=len(names)-len(retained),retained_names=retained),
        figure_cases=cases,figure_selection='10th, 50th and 90th percentile ranks of paired best-of-five difference, sorted by difference then filename; all five fixed seeds shown',
        sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE_ASSETS/'per_image.csv',SOURCE_ASSETS/'results.json',paths.unet_run/'data_manifest.csv']})
    (OUT_ASSETS/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    with (OUT_ASSETS/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,4,figsize=(7.3,6.4))
    colours=['#0072B2','#D55E00','#009E73','#CC79A7','#6F46A6']
    titles=['Layout and reference','Mean predicted probability','Uniform spatial prior','Learned spatial prior']
    for j,name in enumerate(cases):
        record=json.loads((SOURCE_ASSETS/'paths'/(Path(name).stem+'.json')).read_text())
        reference=np.array(record['reference_main_path_xy'])
        brick=np.asarray(Image.open(paths.test_masks/'brick_binary'/name))>0
        layout=np.ones((*brick.shape,3))*.97;layout[brick]=[.80,.64,.53]
        probability=np.load(SOURCE_ASSETS/'probabilities/test'/(Path(name).stem+'.npy')).mean(0)
        for k,ax in enumerate(axes[j]):
            if k==1:heat=ax.imshow(probability,cmap='inferno',vmin=0,vmax=1)
            else:
                ax.imshow(layout)
                ax.plot(reference[:,0],reference[:,1],color='black',linewidth=.65,linestyle='--')
            if k>=2:
                method='no_unet' if k==2 else 'unet'
                records=sorted([r for r in record['paths'] if r['method']==method],key=lambda r:r['seed'])
                for colour,row in zip(colours,records):
                    xy=np.array(row['points_xy']);ax.plot(xy[:,0],xy[:,1],color=colour,linewidth=.5,alpha=.9)
                row=(uniform if k==2 else learned)[name]
                ax.set_xlabel(f'Max {float(row["max_f1"]):.3f}; mean {float(row["mean_f1"]):.3f}',fontsize=8)
            for key in ['known_start_xy','known_goal_xy']:
                x,y=record[key];ax.scatter(x,y,s=9,c='#FFFF00',edgecolors='black',linewidths=.3,zorder=10)
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_title(titles[k],fontsize=8,pad=7)
        axes[j,0].set_ylabel(f'({chr(97+j)})',rotation=0,labelpad=12,fontsize=10)
    fig.subplots_adjust(left=.04,right=.98,top=.94,bottom=.11,wspace=.06,hspace=.19)
    handles=[Line2D([0],[0],color='black',linestyle='--',linewidth=.8,label='Reference')]
    handles += [Line2D([0],[0],color=c,linewidth=1,label=f'Sample {i+1}') for i,c in enumerate(colours)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=6,frameon=False,fontsize=7)
    cbax=fig.add_axes([.365,.068,.16,.012]);fig.colorbar(heat,cax=cbax,orientation='horizontal',ticks=[0,.5,1])
    cbax.tick_params(labelsize=6,pad=1)
    for ext in ['pdf','png']:
        fig.savefig(OUT_ASSETS/f'learned_prior_path_comparison.{ext}',dpi=240)
    plt.close(fig)
    print(json.dumps(dict(summary=summaries,paired_percentile95=stats['paired_percentile95'],figure_cases=cases),indent=2))


# --------------------------------------------------------------------------
# Command-line dispatch
# --------------------------------------------------------------------------

COMMANDS = {
    "calibrate": main_calib,
    "crack-features": main_features,
    "branch-traversal": main_branch,
    "high-recall-traversals": main_traversals,
    "calibrate-with-maps": main_calib_maps,
    "compare-paths": main_compare,
    "far-layouts": main_layouts,
    "preview-masks": main_preview,
    "verify-masks": main_verify,
    "draw-test-masks": main_draw,
    "paper-assets": main_assets,
}


def main(argv=None):
    """Run one analysis, passing the remaining arguments to it unchanged."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    command = argv[0]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(f"Available: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    sys.argv = [f"{sys.argv[0]} {command}"] + argv[1:]
    return COMMANDS[command]()


if __name__ == "__main__":
    raise SystemExit(main())
