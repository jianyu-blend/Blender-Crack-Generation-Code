"""Generate the crack-coordinate file pairs that the Blender stage renders.

This is the production generator: it turns a procedural masonry layout and the
trained crack-probability U-Net into the two text files that
`02_blender_generation/wall_generator.py` reads for one scene.

    <PREFIX>_wall_<N>.txt     one brick per line, with its damage flag
    <PREFIX>_crack_<N>.txt    one polyline per line, the main path first

The five stages follow Section 3.2 of the manuscript:

1. A parametric wall is built by `procedural_layout` and rasterised at
   `RESOLUTION` pixels per metre.
2. A square region is cropped from that raster and resized to 512 x 512. The
   crop lies wholly inside the raster and holds every course. Its origin and
   resize scale are kept so that a sampled pixel can be mapped back to wall
   coordinates in metres.
3. The U-Net predicts the cropped layout and its horizontal, vertical and
   double flips. The four predictions are aligned back to the original
   orientation and fused by `path_sampler.fuse_views`.
4. `path_sampler.generate` walks one path from a mortar start point to a mortar
   end point under the fused prior. Branches are sampled the same way, from a
   mortar pixel on the retained main path.
5. Every brick the path crosses is flagged as broken, the remaining bricks are
   flagged by which side of the crack they lie on, and the two files are
   written.

The sampler, the four-view fusion and the 0.7 to 1.4 in-brick crossing rule are
the released modules in `03_prior_evaluation/`, so generation and the reported
evaluation walk with the same code.

One wall carries several sampled cracks, as the production generator does: the
file pairs of a wall share its prefix and its geometry and differ only in the
crack and the damage flags that follow from it. The defaults write two walls of
three cracks each into the workspace:

    python generate_crack_coordinates.py --walls 2 --paths-per-wall 3
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as connected_components

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / '03_prior_evaluation'))
sys.path.insert(0, str(REPO_ROOT / '01_crack_path_generation/unet'))

from bcg_config import paths                                    # noqa: E402
import procedural_layout as layout                              # noqa: E402
import path_sampler                                             # noqa: E402
from models import UNet                                         # noqa: E402


# ---------------------------------------------------------------
# Generation geometry
# ---------------------------------------------------------------

# Rows, columns and crop scale per view range. The crop side is
# CROP_BASE_M * scale, so `step` gives the 0.922 m extent and the ten brick
# courses the manuscript reports for crack-coordinate generation.
VIEW_PRESETS = {
    'close':  dict(rows=3,  cols=3,  scale=0.5),
    'middle': dict(rows=7,  cols=6,  scale=1.0),
    'far':    dict(rows=10, cols=10, scale=1.6),
    'step':   dict(rows=10, cols=7,  scale=1.8),
}

CROP_BASE_M = 0.512
IMAGE_SIZE = 512
MARGIN_M = 0.05

# The crop is anchored by its lower-left corner at this wall coordinate, which
# is how the production generator places it. Section 3.2.2 of the manuscript
# describes the same point as the centre of the region; the anchoring here is
# the behaviour that produced the released dataset.
ANCHOR_XZ = (0.3, -0.02)

# Goal bias and inertia are the retained values of the grid in
# `03_prior_evaluation/`. The remaining entries are the frozen sampler
# settings that the reported evaluation uses.
SAMPLER_CONFIG = dict(
    gamma=0.7,
    inertia=0.8,
    lookahead=1.0,
    brick_penalty=1.0,
    route_sigma=0.35,
    field_sigma=0.15,
    prior_power=1.0,
    prior_sigma=1.0,
    fusion='half_mean',
)

# Damage flags, as `02_blender_generation/README.md` defines them.
FLAG_CROSSED = 7
FLAG_INTACT = 8
FLAG_SIDE_A = 9
FLAG_SIDE_B = 10

# A crack pixel counts towards breaking a brick only when it lies at least this
# far inside it, measured to the nearest mortar or background pixel. A path
# running along a joint clips the outermost row of the bricks it passes without
# ever entering them, and those contacts are not damage.
MIN_INTERIOR_DEPTH_PX = 2.0

BRANCH_MIN_PX = 40
BRANCH_MAX_PX = 100
BRANCH_CLEARANCE_PX = 50


# ---------------------------------------------------------------
# Layout and crop
# ---------------------------------------------------------------

def build_wall(preset, rng):
    """Draw one parametric wall and rasterise it in image orientation."""
    wall_type = layout.choose_wall_type()
    bricks, _ = layout.generate_wall_bricks(
        wall_type, num_rows=preset['rows'], num_cols=preset['cols'])

    # Shift the wall so that its lower-left corner sits at the origin, which is
    # the frame the wall file and the crack file are both written in.
    origin_x = min(b[0] - b[2] / 2 for b in bricks)
    origin_z = min(b[1] - b[3] / 2 for b in bricks)
    bricks = [(x - origin_x, z - origin_z, w, h, t) for x, z, w, h, t in bricks]

    extent = dict(
        min_x=min(b[0] - b[2] / 2 for b in bricks),
        max_x=max(b[0] + b[2] / 2 for b in bricks),
        min_z=min(b[1] - b[3] / 2 for b in bricks),
        max_z=max(b[1] + b[3] / 2 for b in bricks),
    )

    grid = layout.build_wall_grid(bricks, resolution=layout.RESOLUTION, margin=MARGIN_M)
    grid = np.flipud(grid)
    labels = connected_components(grid > 0.5)[0].astype(np.int32)
    return wall_type, bricks, grid, labels, extent


def crop_region(grid, extent, scale):
    """Crop the square generation region and resize it to 512 x 512.

    The crop is anchored by its lower-left corner at `ANCHOR_XZ` and lies wholly
    inside the wall raster, so every pixel the sampler sees is real layout: the
    region outside the wall is never padded in, because the sampler would read a
    background pixel as mortar and walk into it.

    The joint widths are drawn per wall or per joint, so ten courses do not
    rasterise to a fixed height. The side is therefore the nominal extent, but
    never less than the span the bricks occupy, so that no draw can leave a
    course outside the crop, and never more than the raster itself, so that no
    padding is needed. The window is then placed against the anchor and slid
    back inside the raster if the anchor sits too close to an edge.

    Returns the cropped brick mask and the mapping needed to take a pixel of the
    resized crop back to wall coordinates.
    """
    height, width = grid.shape
    resolution = layout.RESOLUTION
    nominal = int(CROP_BASE_M * scale * resolution)

    anchor_x, anchor_z = ANCHOR_XZ
    x_left = int((anchor_x - (extent['min_x'] - MARGIN_M)) * resolution)
    y_bottom = int(height - 1 - (anchor_z - (extent['min_z'] - MARGIN_M)) * resolution)
    y_bottom = min(max(y_bottom, 0), height - 1)

    # `build_wall_grid` leaves MARGIN_M of empty raster on each side, so the
    # bricks span this many rows. A crop at least this tall always holds them.
    brick_span = height - 2 * int(MARGIN_M * resolution)
    side = min(max(nominal, brick_span), height, width)

    y_top = max(0, min(y_bottom - side + 1, height - side))
    x_left = max(0, min(x_left, width - side))

    square = grid[y_top:y_top + side, x_left:x_left + side]
    if square.shape != (side, side):
        raise RuntimeError('The crop window left the raster; this should not happen.')

    if side != IMAGE_SIZE:
        brick = (np.array(Image.fromarray((square * 255).astype(np.uint8))
                          .resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
    else:
        brick = (square > 0.5).astype(np.uint8)

    top_row = int(MARGIN_M * resolution)
    mapping = dict(y0=y_top, x0=x_left, pad_y=0, pad_x=0,
                   scale=side / IMAGE_SIZE, height=height, width=width,
                   min_x=extent['min_x'], min_z=extent['min_z'],
                   resolution=resolution, margin=MARGIN_M, side=side,
                   nominal=nominal,
                   holds_all_courses=bool(y_top <= top_row
                                          and y_top + side >= height - top_row))
    return brick.astype(np.uint8), mapping


def pixel_to_wall(y_pix, x_pix, mapping):
    """Map one pixel of the resized crop back to wall coordinates in metres."""
    y_orig = mapping['y0'] + (y_pix - mapping['pad_y']) * mapping['scale']
    x_orig = mapping['x0'] + (x_pix - mapping['pad_x']) * mapping['scale']
    wx = (mapping['min_x'] - mapping['margin']) + x_orig / mapping['resolution']
    wz = (mapping['min_z'] - mapping['margin']) + \
         (mapping['height'] - 1 - y_orig) / mapping['resolution']
    return wx, wz


# ---------------------------------------------------------------
# Crack-probability model
# ---------------------------------------------------------------

def load_checkpoint(path, device):
    """Load the crack-probability U-Net and check its mask encoding."""
    if not path.is_file():
        raise FileNotFoundError(
            f'A trained crack-probability checkpoint is required: {path}\n'
            'Train one with 01_crack_path_generation/unet/train.py, or set '
            '`unet_checkpoint` in config.yaml.')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('mask_encoding') != 'adaptive_brick_and_complement':
        raise RuntimeError(
            'The checkpoint uses an unknown or obsolete mask encoding. Retrain '
            'the U-Net with the crack-free layout encoding.')
    model = UNet(in_ch=checkpoint['in_ch'], base=checkpoint['base'],
                 p_drop=checkpoint['p_drop']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def four_view_probability(model, brick, device):
    """Predict the layout and its three flips, aligned back to the original.

    The four aligned maps are what `path_sampler.fuse_views` expects.
    """
    occupancy = torch.from_numpy(brick.astype(np.float32))
    x = torch.stack([occupancy, 1 - occupancy]).to(device)
    dims = [(), (-1,), (-2,), (-2, -1)]
    batch = torch.stack([x if not d else torch.flip(x, d) for d in dims])
    with torch.inference_mode():
        predictions = model(batch).sigmoid()[:, 0]
    aligned = np.stack([(predictions[j] if not d else torch.flip(predictions[j], d)).cpu().numpy()
                        for j, d in enumerate(dims)])
    if aligned.shape != (4, IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(aligned).all():
        raise RuntimeError('The checkpoint produced an unusable prediction.')
    return aligned.astype(np.float64)


# ---------------------------------------------------------------
# Sampler geometry
# ---------------------------------------------------------------

def build_geometry(brick):
    """Per-brick height and long axis, in the form the crossing rule reads.

    This is the geometry that `03_prior_evaluation/task_data.py` builds for an
    annotated mask, derived here from the cropped procedural layout. The
    components are labelled on the cropped mask itself, because
    `crossing_rules.centres_for` visits every label from 1 to the count and a
    component map carried over from the full wall would leave gaps.
    """
    count, component, _, _ = cv2.connectedComponentsWithStats(brick.astype(np.uint8), 8)
    heights = np.ones(count, dtype=np.float64) * 51.2
    axes = np.zeros((count, 2), dtype=np.float64)
    axes[:, 0] = 1
    measured = []
    for label in range(1, count):
        ys, xs = np.where(component == label)
        if len(xs) < 4:
            continue
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        box = cv2.boxPoints(rect)
        vectors = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(vectors, axis=1)
        index = int(np.argmin(lengths))
        heights[label] = max(float(lengths[index]), 2.0)
        axes[label] = vectors[index, ::-1] / max(float(lengths[index]), 1e-12)
        if len(xs) >= 25:
            measured.append(heights[label])
    return dict(brick=brick.astype(np.uint8),
                labels=component.astype(np.int32),
                heights=heights, axes=axes,
                height=float(np.median(measured)) if measured else 51.2)


def pick_endpoints(brick, rng, vertical=True):
    """Draw a start and an end point from the mortar pixels.

    A vertical crack runs between the topmost and the bottommost mortar band,
    which is what the view presets are framed for.
    """
    ys, xs = np.where(brick == 0)
    if len(ys) < 2:
        return None
    if vertical:
        top = np.flatnonzero(ys <= ys.min() + 3)
        bottom = np.flatnonzero(ys >= ys.max() - 3)
        if not len(top) or not len(bottom):
            return None
        a = int(rng.choice(top))
        b = int(rng.choice(bottom))
        return (int(ys[a]), int(xs[a])), (int(ys[b]), int(xs[b]))
    minimum = 0.1 * float(np.hypot(*brick.shape))
    for _ in range(200):
        i, j = rng.choice(len(ys), size=2, replace=False)
        if np.hypot(ys[i] - ys[j], xs[i] - xs[j]) >= minimum:
            return (int(ys[i]), int(xs[i])), (int(ys[j]), int(xs[j]))
    return None


def sample_path(probability, geometry, start, goal, seed):
    """Walk one path under the fused prior between two mortar endpoints."""
    geometry = dict(geometry)
    geometry['start'] = np.asarray(start, dtype=np.int32)
    geometry['goal'] = np.asarray(goal, dtype=np.int32)
    path, details = path_sampler.generate(
        probability, geometry, seed, SAMPLER_CONFIG, endpoint_mode='full_wall')
    return np.asarray(path), details


def sample_branch(probability, geometry, main_path, rng, seed):
    """Draw one branch from a mortar pixel of the retained main path."""
    brick = geometry['brick']
    height, width = brick.shape
    interior = [i for i in range(5, len(main_path) - 5)
                if brick[main_path[i][0], main_path[i][1]] == 0]
    if not interior:
        return None, None
    start = tuple(int(v) for v in main_path[int(rng.choice(interior))])
    occupied = np.zeros(brick.shape, dtype=bool)
    occupied[main_path[:, 0], main_path[:, 1]] = True
    for _ in range(100):
        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(BRANCH_MIN_PX, BRANCH_MAX_PX)
        gy = int(np.clip(start[0] + distance * np.sin(angle), 0, height - 1))
        gx = int(np.clip(start[1] + distance * np.cos(angle), 0, width - 1))
        if brick[gy, gx] or occupied[gy, gx]:
            continue
        separation = np.min(np.hypot(main_path[:, 0] - gy, main_path[:, 1] - gx))
        if separation < BRANCH_CLEARANCE_PX or np.hypot(gy - start[0], gx - start[1]) < BRANCH_MIN_PX:
            continue
        try:
            path, details = sample_path(probability, geometry, start, (gy, gx), seed)
        except (AssertionError, ValueError):
            continue
        if details['success']:
            return path, details
    return None, None


# ---------------------------------------------------------------
# Brick damage flags
# ---------------------------------------------------------------

def brick_flags(bricks, component_full, path_points_full, crossed, extent, shape):
    """Flag every brick as crossed, or by the side of the crack it lies on."""
    height, width = shape
    resolution = layout.RESOLUTION
    by_row = {}
    for y, x in path_points_full:
        by_row.setdefault(y, []).append(x)
    centreline = {y: float(np.mean(v)) for y, v in by_row.items()}
    fallback = float(np.mean([x for _, x in path_points_full])) if path_points_full else width / 2
    rows = np.array(sorted(centreline)) if centreline else np.empty(0)

    flags = []
    for x_centre, z_centre, _, _, _ in bricks:
        pix_x = int((x_centre - (extent['min_x'] - MARGIN_M)) * resolution)
        pix_y = height - 1 - int((z_centre - (extent['min_z'] - MARGIN_M)) * resolution)
        pix_x = min(max(pix_x, 0), width - 1)
        pix_y = min(max(pix_y, 0), height - 1)
        label = int(component_full[pix_y, pix_x])
        if label == 0:
            flags.append(FLAG_INTACT)
        elif label in crossed:
            flags.append(FLAG_CROSSED)
        else:
            if len(rows):
                nearest = rows[int(np.argmin(np.abs(rows - pix_y)))]
                reference = centreline.get(pix_y, centreline[nearest])
            else:
                reference = fallback
            flags.append(FLAG_SIDE_A if pix_x < reference else FLAG_SIDE_B)
    return flags


def count_crack_pixels(component, points, depth=None, min_depth=0.0):
    """Crack pixels per brick, keyed by component id. Background is dropped.

    With a depth map, only pixels at least `min_depth` inside a brick are
    counted, so a path that runs along a joint and grazes the brick edge leaves
    nothing behind.
    """
    counts = {}
    for y, x in points:
        label = int(component[y, x])
        if not label:
            continue
        if depth is not None and depth[y, x] < min_depth:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def crossed_from_counts(counts, minimum):
    """Bricks the crack occupies for at least `minimum` pixels.

    The counts this reads are of interior pixels only, so a path travelling
    along a joint contributes nothing to the brick beside it. A path that does
    enter a brick but only clips its corner leaves a handful of pixels and is
    not a crossing either. The geometric rule in `crossing_rules` already requires
    a complete transverse crossing to run 0.7 to 1.4 brick heights, so this
    threshold only removes the incidental contacts that rule never sees: the
    partial runs at the two supplied endpoints, and the bricks a path grazes
    while travelling along a joint.
    """
    return {label for label, count in counts.items() if count >= minimum}


# ---------------------------------------------------------------
# Output files
# ---------------------------------------------------------------

def write_wall_file(path, bricks, flags):
    """One brick per line: `x z type rotation flag`, in metres."""
    with path.open('w', encoding='utf-8') as handle:
        for (x_centre, z_centre, width, _, theta), flag in zip(bricks, flags):
            brick_type = 1 if width == layout.BRICK_SHORT_W else 2
            handle.write(f'{x_centre} {z_centre} {brick_type} {theta} {flag}\n')


def write_crack_file(path, polylines, mapping):
    """One polyline per line, the main path first, as `(x,z)` pairs in metres."""
    with path.open('w', encoding='utf-8') as handle:
        for points in polylines:
            if not len(points):
                continue
            for y_pix, x_pix in points:
                wx, wz = pixel_to_wall(y_pix, x_pix, mapping)
                handle.write(f'({wx:.8f},{wz:.8f}) ')
            handle.write('\n')


def render_preview(brick, component, polylines, crossed, endpoints):
    """Layout, crossed bricks, crack and endpoints, in the production colours."""
    image = np.zeros((*brick.shape, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)
    image[brick == 0] = (0, 0, 0)
    image[brick > 0] = (0, 255, 0)
    for label in crossed:
        image[component == label] = (128, 0, 128)
    for points in polylines:
        for y, x in points:
            image[y, x] = (255, 255, 0)
    for y, x in endpoints:
        y0, y1 = max(0, y - 5), min(image.shape[0], y + 6)
        x0, x1 = max(0, x - 5), min(image.shape[1], x + 6)
        image[y0:y1, x0:x1] = (255, 0, 0)
    return Image.fromarray(image)


def scene_prefix(rng):
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(rng.choice(list(alphabet)) for _ in range(8))


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------

def generate_wall(model, device, preset, rng, seed, prefix, out_dir, write_preview,
                  min_crossing_pixels, min_interior_depth, paths_per_wall):
    """Draw one wall and sample several cracks on it.

    The wall, its crop and the probability map are produced once and shared, as
    the production generator does: the file pairs of one wall differ only in the
    crack that was sampled and in the damage flags that follow from it.
    """
    wall_type, bricks, grid, component_full, extent = build_wall(preset, rng)
    brick, mapping = crop_region(grid, extent, preset['scale'])
    if not brick.any() or brick.all():
        return []

    probability = four_view_probability(model, brick, device)
    geometry = build_geometry(brick)
    component = geometry['labels']

    # Depth inside a brick, in each of the two label spaces the flags use.
    depth_crop = distance_transform_edt(brick > 0)
    depth_full = distance_transform_edt(grid > 0.5)

    records = []
    for index in range(1, paths_per_wall + 1):
        path_seed = seed + 97 * index
        record = _sample_and_write(
            bricks, grid, component_full, component, extent, brick, mapping,
            probability, geometry, preset, rng, path_seed, index, prefix, out_dir,
            write_preview, min_crossing_pixels, min_interior_depth,
            depth_crop, depth_full, wall_type)
        if record is not None:
            records.append(record)
    return records


def _sample_and_write(bricks, grid, component_full, component, extent, brick, mapping,
                      probability, geometry, preset, rng, seed, index, prefix, out_dir,
                      write_preview, min_crossing_pixels, min_interior_depth,
                      depth_crop, depth_full, wall_type):
    """Sample one crack on a prepared wall and write its file pair."""
    endpoints = pick_endpoints(brick, rng, vertical=True)
    if endpoints is None:
        return None
    start, goal = endpoints

    try:
        main_path, details = sample_path(probability, geometry, start, goal, seed)
    except (AssertionError, ValueError):
        return None

    polylines = [main_path]
    branch, branch_details = (None, None)
    if rng.random() < preset['branch_probability'] and len(main_path) > 10:
        branch, branch_details = sample_branch(probability, geometry, main_path, rng, seed + 1)
        if branch is not None:
            polylines.append(branch)

    # Count the crack pixels brick by brick, in both label spaces at once. The
    # side of the crack is decided on the full wall raster, so that bricks
    # outside the cropped region are flagged too.
    wall_height, wall_width = grid.shape
    points_full = []
    counts_crop = {}
    counts_full = {}
    contacted_full = set()
    for points in polylines:
        for y_pix, x_pix in points:
            y_full = min(max(int(mapping['y0'] + (y_pix - mapping['pad_y']) * mapping['scale']), 0),
                         wall_height - 1)
            x_full = min(max(int(mapping['x0'] + (x_pix - mapping['pad_x']) * mapping['scale']), 0),
                         wall_width - 1)
            points_full.append((y_full, x_full))
            label = int(component_full[y_full, x_full])
            if label:
                contacted_full.add(label)
                if depth_full[y_full, x_full] >= min_interior_depth:
                    counts_full[label] = counts_full.get(label, 0) + 1
            label = int(component[y_pix, x_pix])
            if label and depth_crop[y_pix, x_pix] >= min_interior_depth:
                counts_crop[label] = counts_crop.get(label, 0) + 1

    crossed_full = crossed_from_counts(counts_full, min_crossing_pixels)
    crossed = crossed_from_counts(counts_crop, min_crossing_pixels)
    flags = brick_flags(bricks, component_full, points_full, crossed_full, extent, grid.shape)

    wall_file = out_dir / f'{prefix}_wall_{index}.txt'
    crack_file = out_dir / f'{prefix}_crack_{index}.txt'
    write_wall_file(wall_file, bricks, flags)
    write_crack_file(crack_file, polylines, mapping)

    if write_preview:
        marks = [start, goal] + ([tuple(branch[-1])] if branch is not None else [])
        render_preview(brick, component, polylines, crossed, marks).save(
            out_dir / f'{prefix}_preview_{index}.png')

    return dict(status='ok', prefix=prefix, index=index, wall_type=wall_type,
                wall_file=wall_file.name, crack_file=crack_file.name,
                bricks=len(bricks), crossed_bricks=len(crossed_full),
                contacted_bricks=len(contacted_full),
                entered_bricks=len(counts_full),
                crop_side_pixels=int(mapping['side']),
                crop_side_metres=round(mapping['side'] / mapping['resolution'], 4),
                crop_nominal_pixels=int(mapping['nominal']),
                crop_holds_all_courses=mapping['holds_all_courses'],
                wall_raster=list(grid.shape),
                main_path_points=int(len(main_path)),
                branch_points=int(len(branch)) if branch is not None else 0,
                main_success=bool(details['success']),
                branch_success=bool(branch_details['success']) if branch_details else False,
                flip_weights=details['flip_weights'],
                seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--walls', type=int, default=2,
                        help='How many walls to draw. Default: 2.')
    parser.add_argument('--paths-per-wall', type=int, default=3,
                        help='How many cracks to sample on each wall. The file pairs of '
                             'one wall share its prefix and its geometry, and differ only '
                             'in the crack and the damage flags. Default: 3.')
    parser.add_argument('--view-range', default='step', choices=sorted(VIEW_PRESETS),
                        help='View preset, which sets the wall size and the crop extent.')
    parser.add_argument('--branch-probability', type=float, default=0.3,
                        help='Probability that a scene carries a branch. Default: 0.3.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base seed. Scene i uses seed + 1000 * i.')
    parser.add_argument('--output', type=Path, default=None,
                        help='Where to write. Default: <workspace>/analysis/crack_coordinates.')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Crack-probability checkpoint. Default: the configured one.')
    parser.add_argument('--min-interior-depth', type=float, default=MIN_INTERIOR_DEPTH_PX,
                        help='How far inside a brick a crack pixel must lie, in pixels to '
                             'the nearest mortar or background, before it counts towards '
                             'breaking that brick. A path running along a joint grazes the '
                             'brick edge without entering it. Default: '
                             f'{MIN_INTERIOR_DEPTH_PX}.')
    parser.add_argument('--min-crossing-pixels', type=int, default=10,
                        help='A brick is flagged broken only when the crack occupies at '
                             'least this many pixels of it, measured on the 512-pixel '
                             'crop. Fewer pixels than this is a corner clip, not a '
                             'crossing. Default: 10. Raise it for a close view range, '
                             'where one brick covers more of the crop.')
    parser.add_argument('--no-preview', action='store_true',
                        help='Skip the preview images.')
    args = parser.parse_args()

    if not 0.0 <= args.branch_probability <= 1.0:
        raise SystemExit('--branch-probability must lie in [0, 1].')
    if args.min_crossing_pixels < 0:
        raise SystemExit('--min-crossing-pixels must not be negative.')
    if args.min_interior_depth < 0:
        raise SystemExit('--min-interior-depth must not be negative.')

    out_dir = args.output if args.output is not None else paths.analysis('crack_coordinates')
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint if args.checkpoint is not None else paths.checkpoint

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    preset = dict(VIEW_PRESETS[args.view_range])
    preset['branch_probability'] = args.branch_probability

    records = []
    walls_done = 0
    attempt = 0
    while walls_done < args.walls and attempt < args.walls * 10:
        seed = base_seed + 1000 * attempt
        rng = np.random.default_rng(seed)
        # procedural_layout draws from the `random` module.
        random.seed(seed)
        prefix = scene_prefix(rng)
        attempt += 1
        wall_records = generate_wall(model, device, preset, rng, seed, prefix, out_dir,
                                     not args.no_preview, args.min_crossing_pixels,
                                     args.min_interior_depth, args.paths_per_wall)
        if len(wall_records) != args.paths_per_wall:
            for record in wall_records:
                (out_dir / record['wall_file']).unlink(missing_ok=True)
                (out_dir / record['crack_file']).unlink(missing_ok=True)
                (out_dir / f'{record["prefix"]}_preview_{record["index"]}.png').unlink(missing_ok=True)
            continue
        walls_done += 1
        records.extend(wall_records)
        first = wall_records[0]
        print(f'WALL {walls_done}/{args.walls} {prefix} {first["wall_type"]} '
              f'bricks={first["bricks"]} raster={first["wall_raster"][0]}px '
              f'crop={first["crop_side_pixels"]}px '
              f'(nominal {first["crop_nominal_pixels"]}, all courses='
              f'{first["crop_holds_all_courses"]})', flush=True)
        for record in wall_records:
            print(f'   path {record["index"]}/{args.paths_per_wall} {record["crack_file"]} '
                  f'broken={record["crossed_bricks"]} '
                  f'entered={record["entered_bricks"]} '
                  f'touched={record["contacted_bricks"]} '
                  f'points={record["main_path_points"]}'
                  f'{" +branch" if record["branch_points"] else ""}', flush=True)

    if walls_done < args.walls:
        raise SystemExit(f'Only {walls_done} of {args.walls} walls completed in {attempt} attempts.')

    summary = dict(walls=args.walls, paths_per_wall=args.paths_per_wall,
                   file_pairs=len(records), view_range=args.view_range,
                   nominal_crop_metres=round(CROP_BASE_M * preset['scale'], 4),
                   actual_crop_metres=sorted({r['crop_side_metres'] for r in records}),
                   crop_holds_all_courses=all(r['crop_holds_all_courses'] for r in records),
                   crop_note='The wall raster height varies between draws, because the '
                             'joint widths are drawn per wall or per joint. The crop side '
                             'is the nominal extent, raised to the span the bricks occupy '
                             'if that is larger and lowered to the raster if that is '
                             'smaller, so the crop holds every course and never reaches '
                             'outside the raster. Nothing is padded in: a padded pixel '
                             'would read as mortar to the sampler.',
                   min_crossing_pixels=args.min_crossing_pixels,
                   min_interior_depth_pixels=args.min_interior_depth,
                   damage_rule='A brick is broken when the crack runs at least '
                               'min_interior_depth inside it for at least '
                               'min_crossing_pixels pixels. Contact along a joint is '
                               'not damage.',
                   anchor_xz=list(ANCHOR_XZ), anchor_is='lower-left corner',
                   image_size=IMAGE_SIZE, resolution=layout.RESOLUTION,
                   probability='four aligned flips fused by path_sampler.fuse_views',
                   sampler='path_sampler.generate, endpoint_mode=full_wall',
                   crossing_rule='0.7 to 1.4 brick heights, from crossing_rules',
                   config=SAMPLER_CONFIG, base_seed=base_seed,
                   checkpoint=str(checkpoint_path),
                   checkpoint_epoch=checkpoint.get('epoch'),
                   device=device, output=str(out_dir), records=records)
    (out_dir / 'generation_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
