import random
import numpy as np
import cv2


def polygon(rows, cols, shape):
    """Return filled polygon indices without requiring scikit-image."""
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.rint(np.column_stack((cols, rows))).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return np.nonzero(mask)

# =========================================================
# Parameters
# =========================================================
NUM_ROWS = 10    
NUM_COLS = 10      

BRICK_LONG_W = 0.215
BRICK_SHORT_W = 0.102
BRICK_H = 0.065

MORTAR_MIN = 0.011
MORTAR_MAX = 0.021

RESOLUTION = 1000  # px/m; the wall rasterisation resolution used throughout
OUT_PNG = "brick_wall_10x10.png"

# =========================================================
# Wall-type draw: 40% regular, 40% semi-regular, 10% irregular, 10% chaotic
# =========================================================
def choose_wall_type():
    r = random.random()
    if r < 0.4:
        return "REGULAR"
    elif r < 0.8:
        return "SEMI_REGULAR"
    elif r < 0.9:
        return "IRREGULAR"
    else:
        return "CHAOTIC" 

def fixed_mortar_value():
    return random.uniform(MORTAR_MIN, MORTAR_MAX)

def rand_mortar_value():
    return random.uniform(MORTAR_MIN, MORTAR_MAX)

# Row index is 1-based: odd rows start with a short brick, even rows with a long one
def row_start_is_short(row_idx_1based: int) -> bool:
    return (row_idx_1based % 2 == 1)


def generate_chaotic_wall(num_rows=NUM_ROWS, num_cols=NUM_COLS, max_jitter=0.001, max_theta_deg=0.5):
    """
    Skyline packing for the chaotic wall type:

    1. Upright bricks may appear in the first row and the first column.
    2. Best-fit selection is used only in narrow gaps; wide gaps draw at random.
    3. A strict height cap: anything that would exceed ten courses is not placed.
    """
    bricks = []
    base_mortar = rand_mortar_value()
    
    # Strict height cap: ten standard courses plus nine mortar joints
    limit_H = num_rows * BRICK_H + (num_rows - 1) * base_mortar
    # Target wall width
    target_W = num_cols * BRICK_LONG_W + (num_cols - 1) * base_mortar

    # Initialise the skyline as segments of [left edge, right edge, current height]
    skyline = [{"x1": 0.0, "x2": target_W, "z": 0.0}]

    # Brick configurations: (nominal width, nominal height, span in X, span in Z, base angle)
    brick_configs = [
        (BRICK_LONG_W,  BRICK_H, BRICK_LONG_W,  BRICK_H,       0),  # long stretcher
        (BRICK_SHORT_W, BRICK_H, BRICK_SHORT_W, BRICK_H,       0),  # short stretcher
        (BRICK_LONG_W,  BRICK_H, BRICK_H,       BRICK_LONG_W, 90),  # long, stood upright
        (BRICK_SHORT_W, BRICK_H, BRICK_H,       BRICK_SHORT_W, 90)   # short, stood upright
    ]

    while True:
        # 1. Find the lowest plane in the skyline
        min_z = min(s["z"] for s in skyline)
        if min_z >= limit_H: 
            break
            
        # Index of the skyline segment at that height
        idx = next(i for i, s in enumerate(skyline) if abs(s["z"] - min_z) < 1e-5)
        seg = skyline[idx]
        sw = seg["x2"] - seg["x1"]  # width of the notch

        # 2. Keep only bricks that stay under the height cap
        valid_candidates = []
        for config in brick_configs:
            span_z = config[3]
            # Reject if current height + mortar + brick thickness exceeds the cap
            if seg["z"] + base_mortar + span_z > limit_H:
                continue
            # Reject if the width does not fit (mortar may compress to 0.005 m)
            if config[2] + 0.005 <= sw:
                valid_candidates.append(config)

        # 3. Choose a brick
        if not valid_candidates:
            # Nothing fits (too narrow or too tall): level the notch up to its neighbour
            left_z = skyline[idx-1]["z"] if idx > 0 else float('inf')
            right_z = skyline[idx+1]["z"] if idx < len(skyline)-1 else float('inf')
            seg["z"] = min(left_z, right_z)
            if seg["z"] == float('inf'): break  # cannot be levelled
            
            # Merge adjacent segments of equal height
            i = 0
            while i < len(skyline)-1:
                if abs(skyline[i]["z"] - skyline[i+1]["z"]) < 1e-5:
                    skyline[i]["x2"] = skyline[i+1]["x2"]
                    skyline.pop(i+1)
                else: i += 1
            continue

        # --- Keep the first row from coming out perfectly flat ---
        if sw > 0.3:
            # Plenty of room (first course or a large gap): draw at random so that
            # upright bricks have a chance of appearing
            chosen = random.choice(valid_candidates)
        else:
            # Narrow gap: take the brick that fills the width best, to avoid voids
            valid_candidates.sort(key=lambda c: abs(sw - (c[2] + base_mortar)))
            chosen = valid_candidates[0]

        w_orig, h_orig, span_x, span_z, theta_base = chosen
        
        # Adaptive mortar: shrink the joint when the space is tight
        actual_mortar = base_mortar
        if span_x + base_mortar > sw:
            actual_mortar = max(0.005, sw - span_x)

        # Geometric centre
        cx = seg["x1"] + actual_mortar/2.0 + span_x/2.0
        cz = seg["z"] + actual_mortar/2.0 + span_z/2.0
        
        # Random jitter
        cx += random.uniform(-max_jitter, max_jitter)
        cz += random.uniform(-max_jitter, max_jitter)
        theta = theta_base + random.uniform(-max_theta_deg, max_theta_deg)

        bricks.append((cx, cz, w_orig, h_orig, theta))

        # 4. Update the skyline
        used_w = span_x + actual_mortar
        new_seg_z = seg["z"] + span_z + actual_mortar
        
        # New segment at the height above the placed brick
        new_seg = {"x1": seg["x1"], "x2": seg["x1"] + used_w, "z": new_seg_z}
        
        if abs(used_w - sw) < 1e-4:
            # Exactly filled
            skyline[idx] = new_seg
        else:
            # Partly filled; the remaining width keeps its old height
            seg["x1"] += used_w
            skyline.insert(idx, new_seg)

    return bricks, base_mortar

# =========================================================
# Generate the wall: returns bricks = [(x_center, z_center, width, height, theta), ...]
# Even rows are shifted right by half a joint.
# Position jitter and rotation (theta in degrees) are applied per brick.
# =========================================================
def generate_wall_bricks(wall_type: str, num_rows=NUM_ROWS, num_cols=NUM_COLS,
                         max_jitter=0.002, max_theta_deg=0.5): 
    # CHAOTIC is handled separately
    if wall_type not in ("REGULAR", "SEMI_REGULAR", "IRREGULAR", "CHAOTIC"):
        raise ValueError("wall_type must be REGULAR / SEMI_REGULAR / IRREGULAR / CHAOTIC")

    # === The chaotic type is handed over to the skyline packer ===
    if wall_type == "CHAOTIC":
        return generate_chaotic_wall(num_rows, num_cols, max_jitter, max_theta_deg)
    # ==================================================

    bricks = []

    fixed_mortar = fixed_mortar_value() if wall_type in ("REGULAR", "SEMI_REGULAR") else None

    # Target row width measured in full-length bricks (excluding row_shift); used by
    # IRREGULAR to top up the end of a row
    mortar_mean = 0.5 * (MORTAR_MIN + MORTAR_MAX)
    target_row_width = num_cols * BRICK_LONG_W + (num_cols - 1) * mortar_mean

    z_top_edge = 0.0

    def _append_one(brick_list, x_right_edge, z_center, w, mortar_x):
        x_center = x_right_edge + mortar_x + w / 2.0
        # Position jitter
        jitter_x = random.uniform(-max_jitter, max_jitter)
        jitter_z = random.uniform(-max_jitter, max_jitter)
        x_center += jitter_x
        z_center += jitter_z
        # Rotation angle in degrees
        theta = random.uniform(-max_theta_deg, max_theta_deg)
        brick_list.append((x_center, z_center, w, BRICK_H, theta))
        return x_center + w / 2.0

    for row_idx in range(1, num_rows + 1):
        start_short = row_start_is_short(row_idx)

        # Vertical joint: no joint below the first row
        if row_idx == 1:
            mortar_z = 0.0
            z_center = 0.0
        else:
            mortar_z = fixed_mortar if fixed_mortar is not None else rand_mortar_value()
            z_center = z_top_edge + mortar_z + BRICK_H / 2.0

        # -------------------------------------------------
        # Row offset:
        # - SEMI_REGULAR: odd rows shift right by a fixed 0.0565 m, even rows do not move
        # - REGULAR / IRREGULAR: even rows shift right by half a joint, odd rows do not move
        # -------------------------------------------------
        if wall_type == "SEMI_REGULAR":
            row_shift = 0.0565 if (row_idx % 2 == 1) else 0.0
        else:
            if row_idx % 2 == 0:  # even row
                if fixed_mortar is not None:
                    row_shift = fixed_mortar / 2.0
                else:
                    row_shift = rand_mortar_value() / 2.0
            else:
                row_shift = 0.0

        # IRREGULAR: with probability 0.20 a row is all-long or all-short from the second
        # brick onwards; the first brick still follows the odd/even rule
        force_row_mode = None
        if wall_type == "IRREGULAR" and random.random() < 0.20:
            force_row_mode = "ALL_LONG" if random.random() < 0.5 else "ALL_SHORT"

        # Horizontal cursor within the row, starting at row_shift
        x_right_edge = row_shift

        # Lay the first num_cols bricks of the row
        row_bricks = []
        for col_idx in range(1, num_cols + 1):
            mortar_x = 0.0 if col_idx == 1 else (fixed_mortar if fixed_mortar is not None else rand_mortar_value())

            # Width selection
            if col_idx == 1:
                w = BRICK_SHORT_W if start_short else BRICK_LONG_W
            else:
                if wall_type == "REGULAR":
                    w = BRICK_LONG_W
                elif wall_type == "SEMI_REGULAR":
                    is_short = start_short if (col_idx % 2 == 1) else (not start_short)
                    w = BRICK_SHORT_W if is_short else BRICK_LONG_W
                else:  # IRREGULAR
                    if force_row_mode == "ALL_LONG":
                        w = BRICK_LONG_W
                    elif force_row_mode == "ALL_SHORT":
                        w = BRICK_SHORT_W
                    else:
                        w = BRICK_SHORT_W if random.random() < 0.5 else BRICK_LONG_W

            x_right_edge = _append_one(row_bricks, x_right_edge, z_center, w, mortar_x)

        # -------------------------------------------------
        # IRREGULAR row top-up: bring every row close to the same target width, so that a
        # run of short bricks does not leave a large void
        # -------------------------------------------------
        if wall_type == "IRREGULAR":
            target_right_edge = row_shift + target_row_width

            # Append bricks while the row falls short of the target; stop once the
            # remainder is too small for even a short brick
            max_extra = max(5, num_cols)  # safety bound against an endless loop
            extra = 0
            while extra < max_extra:
                remaining = target_right_edge - x_right_edge
                if remaining <= (MORTAR_MIN + BRICK_SHORT_W):
                    break  # not even the minimum joint plus a short brick fits

                # Try a random joint first; if it is too wide to fit, fall back to MORTAR_MIN
                mortar_x = rand_mortar_value()
                # Try the long brick first, then the short one
                placed = False
                for w_try in (BRICK_LONG_W, BRICK_SHORT_W):
                    if mortar_x + w_try <= remaining:
                        x_right_edge = _append_one(row_bricks, x_right_edge, z_center, w_try, mortar_x)
                        placed = True
                        break

                if not placed:
                    # The random joint was too wide: retry at the minimum joint
                    mortar_x = MORTAR_MIN
                    for w_try in (BRICK_LONG_W, BRICK_SHORT_W):
                        if mortar_x + w_try <= remaining:
                            x_right_edge = _append_one(row_bricks, x_right_edge, z_center, w_try, mortar_x)
                            placed = True
                            break

                if not placed:
                    break

                extra += 1

        bricks.extend(row_bricks)
        z_top_edge = z_center + BRICK_H / 2.0

    return bricks, fixed_mortar

# =========================================================
# Rectangles -> raster grid (brick=1, background=0)
# Rotated bricks are supported by filling their polygons.
# =========================================================
def build_wall_grid(bricks, resolution=RESOLUTION, margin=0.05):
    arr = np.array(bricks, dtype=float)
    xs, zs, ws, hs, thetas = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]  # theta is the per-brick rotation

    min_x = np.min(xs - ws / 2.0) - margin
    max_x = np.max(xs + ws / 2.0) + margin
    min_z = np.min(zs - hs / 2.0) - margin
    max_z = np.max(zs + hs / 2.0) + margin

    W = int((max_x - min_x) * resolution) + 1
    H = int((max_z - min_z) * resolution) + 1

    grid = np.zeros((H, W), dtype=np.float32)

    for x, z, w, h, theta in bricks:
        # Corner points relative to the centre
        half_w = w / 2.0
        half_h = h / 2.0
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])

        # Apply the rotation (degrees -> radians)
        theta_rad = np.deg2rad(theta)
        rot_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])
        corners = np.dot(corners, rot_matrix.T)  # rotate the corners

        # Convert to image coordinates (metres -> pixels)
        corners[:, 0] += x
        corners[:, 1] += z
        px_corners_x = (corners[:, 0] - min_x) * resolution
        px_corners_z = (corners[:, 1] - min_z) * resolution 
        
        # Fill the brick polygon
        rr, cc = polygon(px_corners_z, px_corners_x, shape=(H, W))  # rr: rows (z), cc: cols (x)
        grid[rr, cc] = 1.0

    return grid

def save_wall_image(grid, out_png=OUT_PNG, title=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, cmap="gray", origin="upper")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

def main():
    wall_type = choose_wall_type()
    # Jitter and rotation are passed through
    bricks, fixed_mortar = generate_wall_bricks(wall_type, max_jitter=0.002, max_theta_deg=0.5)
    grid = build_wall_grid(bricks)

    if wall_type in ("REGULAR", "SEMI_REGULAR"):
        title = f"{wall_type} (fixed mortar={fixed_mortar:.4f} m; even-row shift={fixed_mortar/2:.4f} m)"
    else:
        title = f"{wall_type} (random mortar per joint; even-row shift=per-row random/2)"

    save_wall_image(grid, OUT_PNG, title=title)

if __name__ == "__main__":
    main()
