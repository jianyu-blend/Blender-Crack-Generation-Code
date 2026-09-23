"""Prepare crack-free wall masks for crack-probability U-Net training.

Run the three commands in this order:

    python 01_crack_path_generation/preprocessing/mask_preprocessing.py draw-test-masks --partition train
    python 01_crack_path_generation/preprocessing/mask_preprocessing.py preview-masks --source <masks_image> --output <reconstruction>
    python 01_crack_path_generation/preprocessing/mask_preprocessing.py verify-masks --partition train --output <reconstruction> --accepted <preview>

The commands rasterise supplied YOLO polygons, reconstruct the wall beneath the
crack overlay, trace ordered crack polylines, and verify every saved product.
They do not train a model or perform statistical analysis.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import sys

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bcg_config import paths

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
heuristics, not calibrated experimental parameters. These settings affect preprocessing only.
If a class is absent, the available eligible class is used. Hidden labels cannot be uniquely recovered; wide
cracks, mortar continuity and brick corners need visual review. This is preprocessing,
not a validation result for any U-Net. Source count is {len(files)}, processed count is {len(selected)}.

Run from the repository root: `python 01_crack_path_generation/preprocessing/mask_preprocessing.py preview-masks --count {args.count} --output "{args.output.as_posix()}"`.
Dependencies: numpy, Pillow, opencv-python. Zhang-Suen thinning is implemented in NumPy.
''', encoding='utf-8')
    print(json.dumps(report, indent=2))

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

SOURCE_DRAW = paths.dataset_root
OUT_DRAW = paths.test_annotations
PALETTE_DRAW = np.array(
    [[0, 0, 0], [0, 255, 0], [255, 0, 0], [255, 255, 0]], dtype=np.uint8
)

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

Regenerate from the repository root with: python 01_crack_path_generation/preprocessing/mask_preprocessing.py draw-test-masks --partition {partition}
Dependencies: numpy, Pillow, opencv-python.
''', encoding='utf-8')
    print(json.dumps(report, indent=2))

COMMANDS = {
    "draw-test-masks": main_draw,
    "preview-masks": main_preview,
    "verify-masks": main_verify,
}


def main(argv=None):
    """Run one preprocessing command and forward the remaining arguments."""
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
