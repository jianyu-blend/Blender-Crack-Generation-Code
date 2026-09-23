"""Verify and summarise the paired eighteen-setting path experiment.

Usage: python build_prior_ablation.py SOURCE_PER_IMAGE SOURCE_SUMMARY OUTPUT
The source files are read only. No new paths or model predictions are generated.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np


METRICS = {
    "mean_best_f1": ("max_f1", "mean_max_f1"),
    "mean_sample_f1": ("mean_f1", "mean_f1"),
    "mean_in_brick_error": ("in_brick_error", "mean_in_brick_error"),
    "mean_path_distance": ("pairwise_distance", "mean_pairwise_distance"),
    "completion": ("completion", "completion"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def family(name):
    """The filename-family proxy used to cluster the bootstrap intervals."""
    stem = re.split(r"_(?:jpg|jpeg|png)(?:[._]|$)", name, flags=re.I)[0]
    stem = re.sub(r"\.rf\..*$", "", stem)
    if re.fullmatch(r"\d+(?:[-_]\d+)*", stem):
        return "numeric_" + str(int(re.split("[-_]", stem)[0]))
    if re.fullmatch(r"(?:a_\d+|l\d+)_\d+", stem):
        return stem.rsplit("_", 1)[0]
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_image", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expect-settings", type=int, default=None,
                        help="fail unless the sweep holds this many gamma/lambda settings; "
                             "the manuscript grid has 18")
    parser.add_argument("--expect-images", type=int, default=None,
                        help="fail unless each setting scores this many images; "
                             "the manuscript grid has 149")
    parser.add_argument("--expect-families", type=int, default=None,
                        help="fail unless the images fall into this many filename families; "
                             "the manuscript grid has 95")
    args = parser.parse_args()

    records = read_csv(args.per_image)
    archived = {
        (float(row["gamma"]), float(row["inertia"]), row["method"]): row
        for row in read_csv(args.summary)
    }
    groups = defaultdict(dict)
    for row in records:
        key = (float(row["gamma"]), float(row["inertia"]), row["method"])
        name = row["filename"]
        if name in groups[key]:
            raise ValueError(f"Duplicate image: {key} {name}")
        groups[key][name] = row

    settings = sorted({(gamma, inertia) for gamma, inertia, _ in groups})
    image_names = sorted(groups[(settings[0][0], settings[0][1], "unet")])
    images = len(image_names)
    if len(records) != len(settings) * 2 * images:
        raise ValueError(
            f"Unbalanced sweep: {len(settings)} settings and {images} images give "
            f"{len(settings) * 2 * images} expected records, found {len(records)}"
        )
    if args.expect_settings is not None and len(settings) != args.expect_settings:
        raise ValueError(f"Expected {args.expect_settings} settings, found {len(settings)}")
    if args.expect_images is not None and images != args.expect_images:
        raise ValueError(f"Expected {args.expect_images} images, found {images}")

    families = {name: family(name) for name in image_names}
    family_names = list(dict.fromkeys(families[name] for name in image_names))
    if args.expect_families is not None and len(family_names) != args.expect_families:
        raise ValueError(
            f"Expected {args.expect_families} filename-family groups, found {len(family_names)}"
        )
    rng = np.random.default_rng(20260917)
    draws = rng.integers(0, len(family_names), size=(10000, len(family_names)))
    family_counts = np.array([sum(families[name] == family for name in image_names) for family in family_names])

    output_rows = []
    for gamma, inertia in settings:
        uniform = groups[(gamma, inertia, "no_unet")]
        learned = groups[(gamma, inertia, "unet")]
        if len(uniform) != images or set(uniform) != set(learned):
            raise ValueError(f"Unpaired layouts at {(gamma, inertia)}")
        metric_values = {}
        for method, image_rows in (("uniform", uniform), ("unet", learned)):
            archive_row = archived[(gamma, inertia, "no_unet" if method == "uniform" else "unet")]
            for label, (image_field, archive_field) in METRICS.items():
                value = fmean(float(row[image_field]) for row in image_rows.values())
                if abs(value - float(archive_row[archive_field])) > 1e-10:
                    raise ValueError(f"Archive mismatch: {(gamma, inertia)} {method} {label}")
                metric_values[f"{method}_{label}"] = value
        intervals = {}
        for label, image_field in (("best", "max_f1"), ("sample", "mean_f1")):
            family_sums = np.array([
                sum(float(learned[name][image_field]) - float(uniform[name][image_field])
                    for name in image_names if families[name] == family)
                for family in family_names
            ])
            bootstrap_means = family_sums[draws].sum(axis=1) / family_counts[draws].sum(axis=1)
            intervals[f"delta_{label}_ci_low"], intervals[f"delta_{label}_ci_high"] = (
                float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5])
            )
        output_rows.append({
            "gamma": gamma,
            "lambda": inertia,
            "images": images,
            **metric_values,
            "delta_best_f1": metric_values["unet_mean_best_f1"] - metric_values["uniform_mean_best_f1"],
            "delta_sample_f1": metric_values["unet_mean_sample_f1"] - metric_values["uniform_mean_sample_f1"],
            **intervals,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: f"{value:.9f}" if isinstance(value, float) else value for key, value in row.items()})

    print(f"Verified {len(records)} image-method records across {len(settings)} settings")
    print(f"Best-of-five gain range: {min(row['delta_best_f1'] for row in output_rows):.6f} to {max(row['delta_best_f1'] for row in output_rows):.6f}")
    print(f"Mean sampled-path gain range: {min(row['delta_sample_f1'] for row in output_rows):.6f} to {max(row['delta_sample_f1'] for row in output_rows):.6f}")
    print(f"All best-of-five intervals above zero: {all(row['delta_best_ci_low'] > 0 for row in output_rows)}")
    print(f"All mean sampled-path intervals above zero: {all(row['delta_sample_ci_low'] > 0 for row in output_rows)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
