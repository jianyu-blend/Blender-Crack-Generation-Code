"""Convert rendered flat-colour label images into YOLO segmentation polygons.

The label pass of ``wall_generator.py`` renders every class as a flat emission
colour: green for intact brick, red for broken brick, blue for mortar and yellow
for crack. This script separates those regions into object instances and writes
one YOLO segmentation annotation file per image, with class IDs 0, 1 and 2 for
brick, broken brick and crack. Mortar is background and receives no annotation.

The same conversion is used for the external CSG2 dataset, whose binary crack
masks are read in ``--binary`` mode and written with the crack class ID, so that
both synthetic sources are annotated by an identical procedure.

Usage
-----
    python masks_to_yolo_polygons.py --masks <dir> --output <dir>
    python masks_to_yolo_polygons.py --masks <dir> --output <dir> --binary --class-id 2

Each YOLO line has the form ``<class-id> x1 y1 x2 y2 ...`` with coordinates
normalised to the image width and height.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# Flat colours emitted by the label pass, as RGB.
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "brick": (0, 255, 0),
    "broken_brick": (255, 0, 0),
    "crack": (255, 255, 0),
    "mortar": (0, 0, 255),
}

# Downstream instance-segmentation class IDs. Mortar is background.
CLASS_IDS: Dict[str, int] = {
    "brick": 0,
    "broken_brick": 1,
    "crack": 2,
}

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def class_masks(image_rgb: np.ndarray, tolerance: int) -> Dict[str, np.ndarray]:
    """Assign every pixel to its nearest class colour within ``tolerance``.

    Anti-aliasing and the emission shader leave intermediate colours at region
    boundaries. Nearest-colour assignment keeps those pixels with the region they
    belong to instead of discarding them, while pixels further than ``tolerance``
    from every class colour are left unassigned.
    """

    names = list(CLASS_COLOURS)
    reference = np.array([CLASS_COLOURS[n] for n in names], dtype=np.int16)

    flat = image_rgb.reshape(-1, 3).astype(np.int16)
    distances = np.linalg.norm(flat[:, None, :] - reference[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    smallest = distances.min(axis=1)

    height, width = image_rgb.shape[:2]
    assigned = np.where(smallest <= tolerance, nearest, -1).reshape(height, width)

    return {name: (assigned == index).astype(np.uint8)
            for index, name in enumerate(names)}


def polygons_from_mask(mask: np.ndarray, min_area: int,
                       epsilon_ratio: float) -> List[np.ndarray]:
    """Return one simplified outer polygon per connected region of ``mask``."""

    count, labels = cv2.connectedComponents(mask, connectivity=8)
    polygons: List[np.ndarray] = []

    for index in range(1, count):
        component = (labels == index).astype(np.uint8)
        if int(component.sum()) < min_area:
            continue

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(simplified) < 3:
                continue
            polygons.append(simplified.reshape(-1, 2).astype(np.float32))

    return polygons


def format_line(class_id: int, polygon: np.ndarray,
                width: int, height: int, precision: int) -> str:
    normalised = polygon.copy()
    normalised[:, 0] = np.clip(normalised[:, 0] / width, 0.0, 1.0)
    normalised[:, 1] = np.clip(normalised[:, 1] / height, 0.0, 1.0)
    values = " ".join(f"{v:.{precision}f}" for v in normalised.reshape(-1))
    return f"{class_id} {values}"


def convert_colour_mask(path: Path, tolerance: int, min_area: int,
                        epsilon_ratio: float, precision: int) -> List[str]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = class_masks(image_rgb, tolerance)
    lines: List[str] = []
    for name, class_id in CLASS_IDS.items():
        for polygon in polygons_from_mask(masks[name], min_area, epsilon_ratio):
            lines.append(format_line(class_id, polygon, width, height, precision))
    return lines


def convert_binary_mask(path: Path, class_id: int, min_area: int,
                        epsilon_ratio: float, precision: int) -> List[str]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    mask = (image > 0).astype(np.uint8)

    return [format_line(class_id, polygon, width, height, precision)
            for polygon in polygons_from_mask(mask, min_area, epsilon_ratio)]


def iter_masks(directory: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(directory.rglob(pattern)):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert label images into YOLO segmentation polygons.")
    parser.add_argument("--masks", required=True, type=Path,
                        help="Directory holding the label images.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory that receives the .txt annotations.")
    parser.add_argument("--pattern", default="*_M.png",
                        help="Filename pattern of the label images.")
    parser.add_argument("--binary", action="store_true",
                        help="Read single-class binary masks instead of flat-colour labels.")
    parser.add_argument("--class-id", type=int, default=CLASS_IDS["crack"],
                        help="Class ID written in binary mode.")
    parser.add_argument("--tolerance", type=int, default=90,
                        help="Maximum RGB distance for nearest-colour assignment.")
    parser.add_argument("--min-area", type=int, default=32,
                        help="Smallest region retained, in pixels.")
    parser.add_argument("--epsilon-ratio", type=float, default=0.002,
                        help="Polygon simplification tolerance as a fraction of the perimeter.")
    parser.add_argument("--precision", type=int, default=6,
                        help="Decimal places in the normalised coordinates.")
    parser.add_argument("--strip-suffix", default="_M",
                        help="Suffix removed from the label filename to name the annotation.")
    args = parser.parse_args(argv)

    if not args.masks.is_dir():
        print(f"Mask directory not found: {args.masks}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    pattern = "*" if args.binary and args.pattern == "*_M.png" else args.pattern
    written = 0
    empty = 0

    for path in iter_masks(args.masks, pattern):
        if args.binary:
            lines = convert_binary_mask(path, args.class_id, args.min_area,
                                        args.epsilon_ratio, args.precision)
        else:
            lines = convert_colour_mask(path, args.tolerance, args.min_area,
                                        args.epsilon_ratio, args.precision)

        stem = path.stem
        if args.strip_suffix and stem.endswith(args.strip_suffix):
            stem = stem[: -len(args.strip_suffix)]
        (args.output / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written += 1
        if not lines:
            empty += 1

    print(f"Wrote {written} annotation files to {args.output} ({empty} empty).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
