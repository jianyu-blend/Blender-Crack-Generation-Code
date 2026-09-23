"""Sweep the sampler's goal bias and inertia and score every setting by centreline F1.

The probability maps written by evaluate.py are reused unchanged, so this sweep
varies only the path sampler. Each setting is evaluated for both the learned
prior and the uniform prior, using the five fixed seeds, and reports the mean
sampled-path F1 and the mean best-of-five F1.

Outputs go to <workspace>/analysis/gamma_lambda_sweep and never touch the frozen evaluation.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcg_config import paths
from workspace import TEST, SEEDS, read_json, load_mask
from task_data import task_data
from metrics import geometry, reference_geometry, score, morphology, diversity
from path_sampler import generate

EVALUATION = paths.prior_evaluation
OUT = paths.analysis("gamma_lambda_sweep")

BASE = dict(
    id="flip_random", gamma=0.9, inertia=0.9, heading_steps=1, brick_penalty=1.0,
    route_sigma=0.35, field_sigma=0.15, prior_power=1.0, prior_sigma=1.0,
    lookahead=1.0, fusion="random",
)


def run_setting(gamma, inertia, names, protocol):
    """Return per-image records for one (gamma, inertia) setting."""
    config = dict(BASE, gamma=gamma, inertia=inertia)
    records = []
    for name in names:
        task = task_data(TEST, name)
        if task is None:
            continue
        geo = geometry(task)
        width = reference_geometry(load_mask(TEST, "crack_binary", name), task)
        probability = np.load(EVALUATION / "probabilities/test" / (Path(name).stem + ".npy"))
        for method in ("unet", "no_unet"):
            prob = probability if method == "unet" else np.ones_like(probability)
            f1s, paths, completed = [], [], 0
            in_brick, tortuosity = [], []
            reference_in_brick = reference_tortuosity = 0.0
            for seed in SEEDS:
                path, details = generate(
                    prob, geo, seed, config,
                    protocol["primary_attempts"], protocol["attempt_budget"],
                )
                metrics = score(path, task, details["success"], width)
                f1s.append(float(metrics["centreline_f1_at_4px"]))
                # How much of the path runs inside bricks, against the real crack
                # on the same layout: a path that detours along mortar scores lower.
                in_brick.append(float(metrics["generated_in_brick_fraction"]))
                tortuosity.append(float(metrics["generated_tortuosity"]))
                reference_in_brick = float(metrics["reference_in_brick_fraction"])
                reference_tortuosity = float(metrics["reference_tortuosity"])
                paths.append(path)
                completed += int(bool(details["success"]))
            # Geometric spread between the five sampled cracks, in brick heights.
            div = diversity(paths, geo["brick"].shape, task["height"])
            records.append(dict(
                gamma=gamma, inertia=inertia, filename=name, method=method,
                mean_f1=float(np.mean(f1s)), max_f1=float(max(f1s)),
                min_f1=float(min(f1s)), sd_f1=float(np.std(f1s, ddof=1)),
                pairwise_distance=float(div["mean_pairwise_distance_over_height"]),
                unique_paths=int(div["unique_seed_paths"]),
                in_brick=float(np.mean(in_brick)),
                reference_in_brick=reference_in_brick,
                in_brick_error=abs(float(np.mean(in_brick)) - reference_in_brick),
                tortuosity=float(np.mean(tortuosity)),
                reference_tortuosity=reference_tortuosity,
                tortuosity_error=abs(float(np.mean(tortuosity)) - reference_tortuosity),
                completion=completed / len(SEEDS),
                **{f"f1_seed_{s}": v for s, v in zip(SEEDS, f1s)},
            ))
    return records


def summarise(records):
    rows = []
    keys = sorted({(r["gamma"], r["inertia"], r["method"]) for r in records})
    for gamma, inertia, method in keys:
        sel = [r for r in records if (r["gamma"], r["inertia"], r["method"]) == (gamma, inertia, method)]
        rows.append(dict(
            gamma=gamma, inertia=inertia, method=method, images=len(sel),
            mean_f1=float(np.mean([r["mean_f1"] for r in sel])),
            mean_max_f1=float(np.mean([r["max_f1"] for r in sel])),
            mean_min_f1=float(np.mean([r["min_f1"] for r in sel])),
            mean_sd_f1=float(np.mean([r["sd_f1"] for r in sel])),
            mean_pairwise_distance=float(np.mean([r["pairwise_distance"] for r in sel])),
            five_distinct_fraction=float(np.mean([r["unique_paths"] == len(SEEDS) for r in sel])),
            mean_in_brick=float(np.mean([r["in_brick"] for r in sel])),
            mean_reference_in_brick=float(np.mean([r["reference_in_brick"] for r in sel])),
            mean_in_brick_error=float(np.mean([r["in_brick_error"] for r in sel])),
            mean_tortuosity=float(np.mean([r["tortuosity"] for r in sel])),
            mean_reference_tortuosity=float(np.mean([r["reference_tortuosity"] for r in sel])),
            mean_tortuosity_error=float(np.mean([r["tortuosity_error"] for r in sel])),
            completion=float(np.mean([r["completion"] for r in sel])),
        ))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True,
                        help="semicolon-separated gamma,inertia pairs, e.g. 0.7,0.9;0.9,0.9")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images (timing probe)")
    parser.add_argument("--tag", default="sweep", help="output file stem")
    args = parser.parse_args()

    grid = [tuple(float(v) for v in pair.split(",")) for pair in args.grid.split(";") if pair]
    protocol = read_json(EVALUATION / "protocol.json")
    names = protocol["test_names"]
    if args.limit:
        names = names[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    all_records = []
    for gamma, inertia in grid:
        started = time.time()
        records = run_setting(gamma, inertia, names, protocol)
        all_records.extend(records)
        elapsed = time.time() - started
        images = len({r["filename"] for r in records})
        print(f"gamma={gamma} inertia={inertia}  {images} images  {elapsed:.1f}s "
              f"({elapsed / max(images, 1):.2f}s per image)", flush=True)
        # Flush after every setting so an interrupted sweep keeps its completed work.
        write_csv(OUT / f"{args.tag}_per_image.csv", all_records)
        write_csv(OUT / f"{args.tag}_summary.csv", summarise(all_records))

    summary = summarise(all_records)
    (OUT / f"{args.tag}_provenance.json").write_text(json.dumps(dict(
        base_config=BASE, grid=grid, seeds=SEEDS, images=len(names),
        probability_source=str(EVALUATION / "probabilities/test"),
        note="Probability maps reused from the frozen evaluation; only the sampler varies.",
    ), indent=2), encoding="utf-8")

    print()
    for row in summary:
        print(f"  gamma={row['gamma']:<4} inertia={row['inertia']:<5} {row['method']:<8} "
              f"mean_f1={row['mean_f1']:.4f}  mean_max_f1={row['mean_max_f1']:.4f}  "
              f"completion={row['completion']:.3f}")


if __name__ == "__main__":
    main()
