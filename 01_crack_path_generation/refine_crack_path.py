"""Median and Savitzky-Golay smoothing of the exported crack coordinates.

Reads every ``*crack*.txt`` in a directory and rewrites it in place. A file is
refined only when it holds one or two cracks; anything else is left untouched.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import medfilt, savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ========= Tunable parameters =========
MEDIAN_KERNEL = 5        # Median-filter window (odd)
SAVGOL_WINDOW = 11       # Savitzky-Golay window (odd)
SAVGOL_POLY = 3          # Polynomial order
DIST_THRESHOLD = 1e-4    # Duplicate-point removal threshold
# ==================================

def read_points(file_path):
    segments = []
    with open(file_path, 'r') as f:
        for line in f:
            content = line.strip()
            if not content:
                continue
            pts = content.split(') (')
            pts[0] = pts[0].replace('(', '')
            pts[-1] = pts[-1].replace(')', '')
            data = []
            for p in pts:
                if p.strip():
                    # If the source files contain stray non-coordinate characters, add a
                    # replace or a regular-expression filter here.
                    x, y = p.replace('(', '').replace(')', '').split(',')
                    data.append([float(x), float(y)])
            if data:
                segments.append(np.array(data))
    return segments

def remove_duplicate_and_small_jumps(points, threshold):
    cleaned = [points[0]]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - cleaned[-1])
        if dist > threshold:
            cleaned.append(points[i])
    return np.array(cleaned)

def smooth_curve(points):
    x = points[:, 0]
    y = points[:, 1]

    # Median filter
    x_med = medfilt(x, MEDIAN_KERNEL)
    y_med = medfilt(y, MEDIAN_KERNEL)

    # Savitzky-Golay smoothing
    x_smooth = savgol_filter(x_med, SAVGOL_WINDOW, SAVGOL_POLY)
    y_smooth = savgol_filter(y_med, SAVGOL_WINDOW, SAVGOL_POLY)

    return np.column_stack((x_smooth, y_smooth))

def save_points(segments, file_path):
    with open(file_path, 'w') as f:
        for seg in segments:
            for p in seg:
                f.write(f"({p[0]:.8f},{p[1]:.8f}) ")
            f.write("\n")

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".txt") and "crack" in file:
            full_path = os.path.join(folder_path, file)

            segments = read_points(full_path)
            num_cracks = len(segments)
            
            # Refine only when the file holds one or two cracks.
            if num_cracks == 1 or num_cracks == 2:
                print(f"Processing: {file} ({num_cracks} crack(s))")
                processed_segments = []
                for seg in segments:
                    if len(seg) > 1:  # Only segments with enough points
                        seg = remove_duplicate_and_small_jumps(seg, DIST_THRESHOLD)
                        seg = smooth_curve(seg)
                        processed_segments.append(seg)
                save_points(processed_segments, full_path)
            else:
                # More than two or none at all: skip and keep the original file.
                print(f"Skipping: {file} ({num_cracks} crack(s); outside the refine condition)")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?",
                        help="directory of *crack*.txt coordinate files; "
                             "defaults to <workspace>/analysis/crack_coordinates")
    args = parser.parse_args()

    folder = args.folder
    if folder is None:
        from bcg_config import paths
        folder = paths.analysis("crack_coordinates")
    if not folder.is_dir():
        raise SystemExit(f"No such directory: {folder}")
    process_folder(str(folder))


if __name__ == "__main__":
    main()