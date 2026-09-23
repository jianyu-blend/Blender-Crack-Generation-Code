"""Blender scene construction, Boolean crack modelling and automatic label rendering.

This script implements the Blender stage of the BCG synthetic data generation
framework. It reads the wall and crack coordinate files produced by the
learning-guided crack-coordinate generator, builds the masonry scene, represents
the crack either as procedural material texture or as Boolean geometry, applies
domain randomisation and renders aligned RGB images and instance labels.

Workflow
--------
1.  Load the wall file and the crack file for one scene, together with the
    generation preset that defines the view range and the crack-width label.
2.  Construct the masonry geometry by placing the stored stretcher and header
    meshes at their recorded centres and rotations, and assign one procedural
    masonry material to the wall.
3.  For a crack width of at least 3 mm, convert the ordered crack coordinates
    into a bevelled curve and then into a voxel-remeshed cutting mesh. Cracks
    narrower than 3 mm carry no Boolean geometry and are produced by the
    procedural material textures instead.
4.  Subtract the cutting mesh from the bricks crossed by the path and from the
    mortar volume, separate the results and displace the two sides of the crack
    according to the selected crack-side motion.
5.  Randomise the environment map, the shadow-casting objects, the camera poses
    and the focal length, then render one RGB image per camera with Cycles.
6.  Render a second pass with the same scene state in which the materials emit
    class-specific flat colours, which gives a label image aligned with the RGB
    image.

The label images are converted into YOLO segmentation polygons by the separate
script ``masks_to_yolo_polygons.py``.

Usage
-----
    blender --background <asset_library.blend> --python wall_generator.py -- \
        --config config.json

All paths, preset selection and batch size are supplied through the
configuration file or the command line. Run with ``--write-default-config`` to
write a template configuration next to this script.

Coordinate convention
---------------------
The coordinate files are written in metres on the wall plane. The scene is built
at ``scale_factor`` scene units per metre, so every stored coordinate is
multiplied by that factor on import. The wall plane is the global XZ plane and
the cameras look along +Y.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import string
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESETS = SCRIPT_DIR / "presets.json"

# Damage flags stored in the fifth column of the wall file.
FLAG_BROKEN = 7          # brick crossed by the sampled crack path
FLAG_INTACT = 8          # intact brick
FLAG_SIDE_A = 9          # intact brick on the first side of the crack
FLAG_SIDE_B = 10         # intact brick on the second side of the crack

# Object pass indices used by the compositor.
PASS_BROKEN_BRICK = 1
PASS_INTACT_BRICK = 3
PASS_MORTAR = 4
PASS_CRACK_MESH = 5
PASS_CRACK_LABEL = 6

# Flat colours emitted during the label pass.
LABEL_COLOURS = {
    "intact_brick": (0.0, 1.0, 0.0),
    "broken_brick": (1.0, 0.0, 0.0),
    "mortar": (0.0, 0.0, 1.0),
    "crack": (1.0, 1.0, 0.0),
}

COLLECTION_BROKEN = "Broken Bricks"
COLLECTION_INTACT = "Unbroken Bricks"
COLLECTION_SIDE_A = "Side A Bricks"
COLLECTION_SIDE_B = "Side B Bricks"
COLLECTION_MOVED = "Broken Bricks Move"
COLLECTION_CRACKS = "Cracks"
COLLECTION_CRACK_BRICK = "Cracks_BrokenBrick"
COLLECTION_CRACK_MORTAR = "Cracks_Mortar"
COLLECTION_MASK_FIX = "Mask Improvements"

BRICK_TEMPLATES = {
    1: "Metric Half Bricks (Modern Standard)",   # header
    2: "Metric Bricks (Modern Standard)",        # stretcher
}
BRICK_VARIANT_COUNT = 4
BRICK_MATERIALS = [
    "Brick_01_ML", "Brick_02_ML", "Brick_03_ML",
    "Brick_01_MD", "Brick_02_MD", "Brick_03_MD",
]
LABEL_EMISSION_MATERIAL = "Emission_Yellow"
MASK_TEMPLATE_STANDARD = "Metric Bricks (Modern Standard) Mask"
MASK_TEMPLATE_HALF = "Metric Half Bricks (Modern Standard) Mask"


@dataclass
class Config:
    """Runtime configuration for one generation batch."""

    # Inputs
    asset_library: str = ""
    """Blender file holding the brick meshes, materials and geometry nodes."""

    hdri_directory: str = ""
    """Directory of equirectangular .exr environment maps."""

    coordinate_directory: str = ""
    """Directory holding the ``*_wall_*.txt`` and ``*_crack_*.txt`` file pairs."""

    output_directory: str = ""
    """Directory that receives the rendered images and labels."""

    presets_file: str = str(DEFAULT_PRESETS)

    # Preset selection
    view_range: str = "middle"
    """One of ``close``, ``middle`` or ``far``."""

    crack_width_label: str = "5to10mm"
    """One of the five crack-width labels defined in the presets file."""

    crack_side_motion: str = "translation"
    """One of ``stay``, ``translation`` or ``settling``."""

    # Batch
    max_scenes: int = 1
    seed: Optional[int] = None

    # Rendering
    resolution: int = 640
    render_samples: int = 256
    use_optix: bool = True
    stereo_offset_m: float = 0.10
    """Lateral offset of the paired camera, in metres on the wall plane."""

    shadow_caster_objects: List[str] = field(default_factory=lambda: ["Roofing"])
    """Rectangular objects that stay outside the camera view and cast shadows."""

    write_depth: bool = True

    # Appearance randomisation
    brick_variant_scene_probability: float = 0.67
    brick_variant_unit_probability: float = 0.70
    brick_variant_flip_probability: float = 0.50
    mortar_geometry_node_probability: float = 0.75
    hdri_strength_range: Tuple[float, float] = (0.2, 1.0)
    brick_displacement_scale_range: Tuple[float, float] = (0.05, 0.10)
    """One scene-wide scale sampled for NodeGroup_RoughB displacement."""

    label_render_samples: int = 4
    """Cycles samples for the flat-colour pass; Cycles keeps shader displacement aligned."""

    mask_shrinkwrap_offset: float = 0.03
    """Distance that keeps the yellow helper inside its broken-brick target."""

    def resolved(self, key: str) -> Path:
        value = getattr(self, key)
        if not value:
            raise ValueError(f"Configuration entry '{key}' is required but empty.")
        return Path(value).expanduser().resolve()


def load_config(argv: Sequence[str]) -> Tuple[Config, dict]:
    """Build the configuration from a JSON file and command-line overrides."""

    parser = argparse.ArgumentParser(
        prog="wall_generator.py",
        description="Generate cracked masonry scenes and aligned instance labels.",
    )
    parser.add_argument("--config", help="Path to a JSON configuration file.")
    parser.add_argument("--write-default-config", metavar="PATH",
                        help="Write a template configuration file and exit.")
    parser.add_argument("--asset-library")
    parser.add_argument("--hdri-directory")
    parser.add_argument("--coordinate-directory")
    parser.add_argument("--output-directory")
    parser.add_argument("--presets-file")
    parser.add_argument("--view-range", choices=["close", "middle", "far"])
    parser.add_argument("--crack-width-label")
    parser.add_argument("--crack-side-motion",
                        choices=["stay", "translation", "settling"])
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--render-samples", type=int)
    args = parser.parse_args(argv)

    if args.write_default_config:
        target = Path(args.write_default_config).expanduser().resolve()
        target.write_text(json.dumps(asdict(Config()), indent=2), encoding="utf-8")
        print(f"Wrote template configuration to {target}")
        raise SystemExit(0)

    data: dict = {}
    if args.config:
        data = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
        data.pop("_comment", None)

    for name, value in vars(args).items():
        if name in ("config", "write_default_config") or value is None:
            continue
        data[name] = value

    known = {f for f in Config.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown configuration entries: {sorted(unknown)}")

    config = Config(**data)
    presets = json.loads(Path(config.presets_file).expanduser().read_text(encoding="utf-8"))
    return config, presets


def resolve_preset(config: Config, presets: dict) -> dict:
    """Return the flat parameter set for the selected view range and crack width."""

    if config.view_range not in presets["view_ranges"]:
        raise ValueError(f"Unknown view range: {config.view_range}")
    if config.crack_width_label not in presets["crack_width"]:
        raise ValueError(f"Unknown crack-width label: {config.crack_width_label}")

    width = presets["crack_width"][config.crack_width_label]
    focal = presets["focal_length_mm"][config.view_range]
    motion = presets["crack_side_motion"][config.crack_side_motion]

    return {
        "scale_factor": presets["scale_factor"],
        "boolean_geometry": width["boolean_geometry"],
        "delta_c": width["delta_c"],
        "delta_b": width["delta_b"],
        "delta_m": width["delta_m"],
        "bevel": width["bevel"],
        "focal": focal,
        "motion": motion,
        "label": output_label(config),
    }


def output_label(config: Config) -> str:
    """Return the generation label encoded in every rendered filename."""

    view = {"close": "C", "middle": "M", "far": "F"}[config.view_range]
    width = {
        "lt3mm": "W0", "3to5mm": "W1", "5to10mm": "W2",
        "10to30mm": "W3", "gt30mm": "W4",
    }[config.crack_width_label]
    motion = {"stay": "ST", "translation": "TR", "settling": "SE"}[config.crack_side_motion]
    return f"{motion}{view}{width}"


# --------------------------------------------------------------------------- #
# Generic scene helpers
# --------------------------------------------------------------------------- #

def clear_scene() -> None:
    """Remove every object and orphaned datablock from the current scene."""

    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.select_set(True)
    bpy.ops.object.delete()

    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    crack_texture = bpy.data.textures.get("CrackClouds")
    if crack_texture:
        bpy.data.textures.remove(crack_texture)

    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras,
                  bpy.data.lights, bpy.data.images):
        for item in list(block):
            if not item.users:
                block.remove(item)

    for group in list(bpy.data.node_groups):
        if getattr(group, "type", "") == "GEOMETRY" or group.bl_idname == "GeometryNodeTree":
            bpy.data.node_groups.remove(group)

    for image in list(bpy.data.images):
        if (image.filepath or "").lower().endswith(".exr"):
            bpy.data.images.remove(image, do_unlink=True)


def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj: bpy.types.Object, name: str) -> None:
    target = ensure_collection(name)
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    target.objects.link(obj)


def delete_collection(name: str) -> None:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for child in list(collection.children):
        collection.children.unlink(child)
    for parent in list(bpy.data.collections):
        if parent.children.get(collection.name) is not None:
            parent.children.unlink(collection)
    for scene in bpy.data.scenes:
        if scene.collection.children.get(collection.name) is not None:
            scene.collection.children.unlink(collection)
    bpy.data.collections.remove(collection)


def set_collection_visibility(name: str, *, viewport: Optional[bool] = None,
                              render: Optional[bool] = None) -> None:
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for obj in collection.objects:
        if viewport is not None:
            obj.hide_viewport = viewport
        if render is not None:
            obj.hide_render = render


def select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj: bpy.types.Object, modifier: bpy.types.Modifier) -> None:
    select_only(obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def smart_uv_project(obj: bpy.types.Object) -> None:
    if obj is None or obj.type != "MESH":
        return
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project()
    except RuntimeError as error:
        print(f"Smart UV project skipped for {obj.name}: {error}")
    bpy.ops.object.mode_set(mode="OBJECT")


def unwrap_and_scale_uv(obj: bpy.types.Object, scale: float = 2.0) -> None:
    if obj.type != "MESH":
        return
    smart_uv_project(obj)
    if not obj.data.uv_layers:
        return
    for loop in obj.data.uv_layers.active.data:
        loop.uv.x *= scale
        loop.uv.y *= scale


def random_basename(length: int = 7) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def unique_basename(output_dir: Path, label: str, length: int = 7) -> str:
    while True:
        name = f"{label}_{random_basename(length)}"
        if not (output_dir / f"{name}_1_P.png").exists():
            return name


def append_from_library(library: Path, category: str, name: str) -> Optional[bpy.types.ID]:
    """Append one datablock from the asset library and return it."""

    bpy.ops.wm.append(
        filepath=str(library / category / name),
        directory=str(library / category),
        filename=name,
    )
    if category == "Object":
        return bpy.context.selected_objects[-1] if bpy.context.selected_objects else None
    if category == "Material":
        return bpy.data.materials.get(name)
    return None


def append_object_by_prefix(library: Path, prefix: str) -> Optional[bpy.types.Object]:
    """Append the first library object whose name matches or starts with ``prefix``."""

    with bpy.data.libraries.load(str(library), link=False) as (source, target):
        exact = [n for n in source.objects if n == prefix]
        matches = exact or [n for n in source.objects if n.startswith(prefix)]
        target.objects = matches[:1]

    for obj in target.objects:
        if obj is None:
            continue
        linked = any(obj.name in col.objects for col in bpy.data.collections)
        if not linked:
            bpy.context.scene.collection.objects.link(obj)
        return obj
    print(f"Object not found in asset library: {prefix}")
    return None


def randomise_brick_displacement(material: Optional[bpy.types.Material],
                                 bounds: Sequence[float]) -> Optional[float]:
    """Set one shared RoughB displacement scale for every brick in this scene.

    All placed bricks share ``material`` and its nested ``NodeGroup_RoughB``
    datablock. Sampling once here therefore gives every brick exactly the same
    displacement scale while allowing the next generated scene to differ.
    """

    if material is None or not material.use_nodes or not material.node_tree:
        print("Brick displacement randomisation skipped: material has no node tree.")
        return None
    if len(bounds) != 2:
        raise ValueError("brick_displacement_scale_range must contain two values.")

    low, high = sorted((float(bounds[0]), float(bounds[1])))
    if low < 0:
        raise ValueError("brick_displacement_scale_range cannot contain negative values.")
    scale = random.uniform(low, high)

    groups = []
    for node in material.node_tree.nodes:
        if node.type != "GROUP" or node.node_tree is None:
            continue
        base_name = re.sub(r"\.\d+$", "", node.node_tree.name)
        if base_name == "NodeGroup_RoughB" and node.node_tree not in groups:
            groups.append(node.node_tree)

    changed = 0
    for group in groups:
        for node in group.nodes:
            if node.bl_idname != "ShaderNodeDisplacement":
                continue
            scale_input = node.inputs.get("Scale")
            if scale_input is not None:
                scale_input.default_value = scale
                changed += 1

    if not changed:
        print(f"Brick displacement randomisation skipped: NodeGroup_RoughB not found in {material.name}.")
        return None

    bpy.context.scene["brick_displacement_scale"] = scale
    print(f"Brick displacement scale: {scale:.6f} (shared by all bricks)")
    return scale


# --------------------------------------------------------------------------- #
# Coordinate input
# --------------------------------------------------------------------------- #

COORDINATE_PATTERN = re.compile(r"([A-Z0-9]+)_(crack|wall)_(\d+)\.txt")


def discover_scene_inputs(directory: Path) -> List[Tuple[Path, Path]]:
    """Pair every ``*_crack_*.txt`` file with its matching ``*_wall_*.txt`` file."""

    cracks: Dict[Tuple[str, str], Path] = {}
    walls: Dict[Tuple[str, str], Path] = {}
    for entry in sorted(directory.iterdir()):
        match = COORDINATE_PATTERN.match(entry.name)
        if not match:
            continue
        prefix, kind, index = match.groups()
        (cracks if kind == "crack" else walls)[(prefix, index)] = entry

    keys = sorted(set(cracks) & set(walls))
    return [(cracks[k], walls[k]) for k in keys]


def read_wall_file(path: Path, scale: float) -> List[Tuple[float, float, int, float, int]]:
    """Read brick centres, types, rotations and damage flags from the wall file.

    Each line holds ``x z type rotation flag``. Coordinates are in metres and are
    multiplied by ``scale`` to give scene units. The stored rotation is negated
    so that the wall file convention matches the Blender Y rotation.
    """

    bricks: List[Tuple[float, float, int, float, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            bricks.append((
                float(parts[0]) * scale,
                float(parts[1]) * scale,
                int(parts[2]),
                -float(parts[3]),
                int(parts[4]),
            ))
        except ValueError:
            print(f"Skipping malformed brick line: {line.strip()}")
    return bricks


def read_crack_file(path: Path, scale: float) -> List[List[Tuple[float, float]]]:
    """Read the ordered main path and each branch from the crack file.

    Every line holds one polyline as a sequence of ``(x, z)`` pairs in metres.
    """

    polylines: List[List[Tuple[float, float]]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        points: List[Tuple[float, float]] = []
        for match in re.findall(r"\(([^)]+)\)", line):
            parts = match.split(",")
            if len(parts) != 2:
                continue
            try:
                points.append((float(parts[0]) * scale, float(parts[1]) * scale))
            except ValueError:
                continue
        if points:
            polylines.append(points)
    return polylines


def closest_index(point: Tuple[float, float],
                  points: Sequence[Tuple[float, float]]) -> int:
    if not points:
        return 0
    return min(range(len(points)),
               key=lambda i: (points[i][0] - point[0]) ** 2 + (points[i][1] - point[1]) ** 2)


# --------------------------------------------------------------------------- #
# Boolean crack geometry
# --------------------------------------------------------------------------- #

def add_crack_spline(curve: bpy.types.Curve, points: Sequence[Tuple[float, float]],
                     base_radius: float) -> None:
    """Add one polyline with a randomly varying local radius.

    The local radius multiplier is ``r_i = r_0 + e_i`` with ``e_i`` drawn from a
    uniform distribution on ``[-0.2, 0.2]``. This prevents the modelled crack
    from having a uniform width along its length. The brick cutter uses the base
    value ``r_0 = 1.0``, so ``r_i`` lies in [0.8, 1.2] and the nominal crack
    width is the one set by the sampled bevel depth alone.
    """

    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for index, (x, z) in enumerate(points):
        spline.points[index].co = (x, -0.05, z, 1)
        spline.points[index].radius = max(0.5, base_radius + random.uniform(-0.2, 0.2))


def build_crack_curve(polylines: List[List[Tuple[float, float]]], name: str,
                      bevel_depth: float, base_radius: float) -> Optional[bpy.types.Object]:
    """Convert the crack polylines into a single bevelled curve object.

    When three or more polylines are present they are kept independent. Otherwise
    the first polyline is the main path and every remaining polyline is a branch
    whose start point is moved onto the nearest point of the main path, so that
    the complete crack stays connected.
    """

    if not polylines:
        return None

    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2

    if len(polylines) >= 3:
        for points in polylines:
            if points:
                add_crack_spline(curve, points, base_radius)
    else:
        main = polylines[0]
        add_crack_spline(curve, main, base_radius)
        for branch in polylines[1:]:
            if not branch:
                continue
            branch = list(branch)
            branch[0] = main[closest_index(branch[0], main)]
            add_crack_spline(curve, branch, base_radius)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    curve.bevel_depth = bevel_depth
    curve.use_radius = True
    curve.bevel_resolution = 2
    return obj


def curve_to_mesh(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    select_only(obj)
    bpy.ops.object.convert(target="MESH")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.edge_face_add()
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.name = name
    obj.data.name = name
    return obj


def build_cutting_object(polylines: List[List[Tuple[float, float]]], name: str,
                         collection: str, bevel_depth: float, base_radius: float,
                         voxel_size: float) -> Optional[bpy.types.Object]:
    """Build one voxel-remeshed Boolean cutting mesh from the crack polylines."""

    curve = build_crack_curve(polylines, f"{name}_Curve", bevel_depth, base_radius)
    if curve is None:
        return None

    mesh = curve_to_mesh(curve, name)
    remesh = mesh.modifiers.new(name="Remesh", type="REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = voxel_size
    apply_modifier(mesh, remesh)
    move_to_collection(mesh, collection)
    return mesh


def stretch_cutters_through_wall(collection_name: str, bevel_depth: float) -> None:
    """Extend each cutting mesh along Y so that the Boolean cuts fully through."""

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        return
    scale_y = 5.0 / bevel_depth
    for obj in collection.objects:
        select_only(obj)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.transform.resize(value=(1, scale_y, 1))
        bpy.ops.object.mode_set(mode="OBJECT")


# --------------------------------------------------------------------------- #
# Masonry construction
# --------------------------------------------------------------------------- #

def merge_duplicate_materials(obj: bpy.types.Object) -> None:
    """Point duplicated material slots such as ``Brick_01_ML.001`` at the original."""

    for index, material in enumerate(obj.data.materials):
        if material is None:
            continue
        base_name = re.sub(r"\.\d+$", "", material.name)
        original = bpy.data.materials.get(base_name)
        if original is not None and original is not material:
            obj.data.materials[index] = original
            bpy.data.materials.remove(material)


def load_brick_templates(library: Path) -> Dict[int, dict]:
    """Append the base brick meshes and their shape variants from the library."""

    templates: Dict[int, dict] = {}
    for label, model in BRICK_TEMPLATES.items():
        base = append_from_library(library, "Object", model)
        if base is None:
            print(f"Failed to load brick template: {model}")
            continue
        base.data.materials.clear()

        variants = []
        for index in range(1, BRICK_VARIANT_COUNT + 1):
            variant = append_from_library(library, "Object", f"{model}.Type0{index}")
            if variant is not None:
                variant.data.materials.clear()
                variants.append(variant)
        templates[label] = {"base": base, "variants": variants}
    return templates


def place_bricks(bricks: Sequence[Tuple[float, float, int, float, int]],
                 templates: Dict[int, dict], material: Optional[bpy.types.Material],
                 config: Config) -> None:
    """Place one brick mesh at every stored centre, rotation and damage flag."""

    for name in (COLLECTION_BROKEN, COLLECTION_INTACT,
                 COLLECTION_SIDE_A, COLLECTION_SIDE_B):
        ensure_collection(name)

    flag_to_collection = {
        FLAG_BROKEN: COLLECTION_BROKEN,
        FLAG_INTACT: COLLECTION_INTACT,
        FLAG_SIDE_A: COLLECTION_SIDE_A,
        FLAG_SIDE_B: COLLECTION_SIDE_B,
    }

    use_variants = random.random() <= config.brick_variant_scene_probability

    for x, z, label, rotation, flag in bricks:
        template_set = templates.get(label)
        if template_set is None:
            print(f"Unknown brick type {label} at ({x:.3f}, {z:.3f}); skipped.")
            continue

        replaced = (use_variants
                    and template_set["variants"]
                    and random.random() <= config.brick_variant_unit_probability)
        template = random.choice(template_set["variants"]) if replaced else template_set["base"]

        select_only(template)
        bpy.ops.object.duplicate(linked=False)
        brick = bpy.context.active_object
        if brick is None:
            print(f"Failed to duplicate brick type {label} at ({x:.3f}, {z:.3f}).")
            continue

        brick.location = (x, 0.0, z)
        final_rotation = rotation
        if replaced and random.random() <= config.brick_variant_flip_probability:
            final_rotation += 180.0
        brick.rotation_euler[1] = math.radians(final_rotation)

        merge_duplicate_materials(brick)
        if material is not None:
            brick.data.materials.clear()
            brick.data.materials.append(material)
            brick.active_material = material

        collection_name = flag_to_collection.get(flag)
        if collection_name is None:
            print(f"Unknown damage flag {flag} at ({x:.3f}, {z:.3f}); treated as intact.")
            collection_name = COLLECTION_INTACT
        move_to_collection(brick, collection_name)

        unwrap_and_scale_uv(brick, scale=random.uniform(1.5, 2.5))

    for template_set in templates.values():
        if template_set["base"]:
            bpy.data.objects.remove(template_set["base"], do_unlink=True)
        for variant in template_set["variants"]:
            if variant:
                bpy.data.objects.remove(variant, do_unlink=True)


def build_mortar(bricks: Sequence[Tuple[float, float, int, float, int]],
                 library: Path, brick_material_name: str) -> Tuple[Optional[bpy.types.Object], dict]:
    """Create the mortar volume spanning the wall and assign a mortar material."""

    if not bricks:
        return None, {}

    min_x = min(b[0] for b in bricks)
    max_x = max(b[0] for b in bricks)
    min_z = min(b[1] for b in bricks)
    max_z = max(b[1] for b in bricks)

    extent = {
        "size_x": (max_x - min_x) + 1.2,
        "size_y": random.uniform(0.367, 0.487),
        "size_z": (max_z - min_z) + 0.4,
        "centre_x": (min_x + max_x) / 2.0,
        "centre_z": (min_z + max_z) / 2.0,
        "min_x": min_x, "max_x": max_x, "min_z": min_z, "max_z": max_z,
    }

    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(extent["centre_x"], 0, extent["centre_z"]))
    mortar = bpy.context.active_object
    mortar.name = "Mortar"
    mortar.scale = (extent["size_x"], extent["size_y"], extent["size_z"])
    bpy.ops.object.transform_apply(scale=True)
    smart_uv_project(mortar)

    if brick_material_name.endswith("_ML"):
        mortar_name = "Mortar_02"
    elif brick_material_name.endswith("_MD"):
        mortar_name = "Mortar_01"
    else:
        mortar_name = random.choice(["Mortar_01", "Mortar_02"])

    mortar_material = append_from_library(library, "Material", mortar_name)
    if mortar_material is not None:
        mortar.data.materials.clear()
        mortar.data.materials.append(mortar_material)
    else:
        print(f"Mortar material not found in asset library: {mortar_name}")

    return mortar, extent


# --------------------------------------------------------------------------- #
# Boolean subtraction and crack-side motion
# --------------------------------------------------------------------------- #

def boolean_and_split(target: bpy.types.Object,
                      cutters: bpy.types.Collection) -> List[bpy.types.Object]:
    """Subtract the cutting collection from ``target`` and separate loose parts."""

    select_only(target)
    modifier = target.modifiers.new(name="Crack_Boolean", type="BOOLEAN")
    modifier.operand_type = "COLLECTION"
    modifier.collection = cutters
    modifier.operation = "DIFFERENCE"
    modifier.solver = "FAST"
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")

    parts = list(bpy.context.selected_objects)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
    parts.sort(key=lambda o: o.location.x)
    return parts


def split_parts_by_side(parts: List[bpy.types.Object],
                        side_a: str, side_b: str) -> None:
    """Assign the lower-x half of the separated parts to side A and the rest to side B."""

    for index, part in enumerate(parts):
        move_to_collection(part, side_a if index < len(parts) / 2.0 else side_b)


def cut_broken_bricks(cutters: bpy.types.Collection) -> Tuple[List[bpy.types.Object], List[dict]]:
    """Apply the Boolean cut to every brick crossed by the sampled crack path."""

    broken = bpy.data.collections.get(COLLECTION_BROKEN)
    if broken is None or not broken.objects:
        return [], []

    halves: List[bpy.types.Object] = []
    records: List[dict] = []

    for brick in list(broken.objects):
        location = brick.location.copy()
        rotation = brick.rotation_euler.copy()
        mask_template = (MASK_TEMPLATE_HALF
                         if brick.name.startswith("Metric Half Bricks")
                         else MASK_TEMPLATE_STANDARD)
        parts = boolean_and_split(brick, cutters)
        halves.extend(parts)
        records.append({"position": location, "rotation": rotation,
                        "parts": parts, "joined": None,
                        "mask_template": mask_template})
        split_parts_by_side(parts, COLLECTION_BROKEN, COLLECTION_MOVED)

    return halves, records


def cut_mortar(mortar: bpy.types.Object, cutters: bpy.types.Collection,
               bevel_depth: float) -> None:
    """Apply the Boolean cut to the mortar volume and split it by crack side."""

    for obj in cutters.objects:
        select_only(obj)
        scale_y = 5.0 / bevel_depth
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.transform.resize(value=(1, scale_y, 1))
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    parts = boolean_and_split(mortar, cutters)
    split_parts_by_side(parts, COLLECTION_SIDE_A, COLLECTION_SIDE_B)


def displace_crack_side(motion: dict) -> None:
    """Displace the second side of the crack relative to the first.

    The rotation is applied about the common median point of the moved objects,
    first about Z and then about Y, after which the translation is added. The
    result is a relative movement between the two sides of the crack, which
    represents settlement or in-plane translation of part of the wall.
    """

    moved: List[bpy.types.Object] = []
    for name in (COLLECTION_SIDE_B, COLLECTION_MOVED):
        collection = bpy.data.collections.get(name)
        if collection:
            moved.extend(list(collection.objects))
    if not moved:
        return

    median = Vector((
        sum(o.location.x for o in moved) / len(moved),
        sum(o.location.y for o in moved) / len(moved),
        sum(o.location.z for o in moved) / len(moved),
    ))

    def sample_angle(bounds: Sequence[float]) -> float:
        if random.random() >= motion["rotation_probability"]:
            return 0.0
        return random.uniform(bounds[0], bounds[1])

    rot_z = math.radians(sample_angle(motion["rotation_z_deg"]))
    rot_y = math.radians(sample_angle(motion["rotation_y_deg"]))
    trans_x = random.uniform(*motion["translation_x"])
    trans_z = random.uniform(*motion["translation_z"])

    for obj in moved:
        dx = obj.location.x - median.x
        dy = obj.location.y - median.y
        dz = obj.location.z - median.z

        x1 = dx * math.cos(rot_z) - dy * math.sin(rot_z)
        y1 = dx * math.sin(rot_z) + dy * math.cos(rot_z)
        z1 = dz

        x2 = x1 * math.cos(rot_y) + z1 * math.sin(rot_y)
        y2 = y1
        z2 = -x1 * math.sin(rot_y) + z1 * math.cos(rot_y)

        obj.location = (median.x + x2 + trans_x, median.y + y2, median.z + z2 + trans_z)
        obj.rotation_euler[2] += rot_z
        obj.rotation_euler[1] += rot_y

    moved_collection = bpy.data.collections.get(COLLECTION_MOVED)
    if moved_collection:
        for obj in list(moved_collection.objects):
            move_to_collection(obj, COLLECTION_BROKEN)


def rebuild_mortar(delta_m: float) -> Optional[bpy.types.Object]:
    """Voxel-remesh each separated mortar piece and join them into one object."""

    pieces: List[bpy.types.Object] = []
    for name in (COLLECTION_SIDE_A, COLLECTION_SIDE_B):
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        for obj in list(collection.objects):
            if obj.name.startswith("Mortar"):
                pieces.append(obj)
                for current in list(obj.users_collection):
                    current.objects.unlink(obj)
                bpy.context.scene.collection.objects.link(obj)

    for piece in pieces:
        select_only(piece)
        remesh = piece.modifiers.new("RemeshFix", "REMESH")
        remesh.mode = "VOXEL"
        remesh.voxel_size = delta_m
        remesh.adaptivity = 0
        apply_modifier(piece, remesh)

    if not pieces:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for piece in pieces:
        piece.select_set(True)
    bpy.context.view_layer.objects.active = pieces[0]
    bpy.ops.object.join()
    mortar = bpy.context.active_object
    mortar.name = "Mortar"
    return mortar


def polish_bricks(halves: Sequence[bpy.types.Object], delta_b: float) -> None:
    """Remesh and simplify the cut brick halves, then bevel the intact bricks."""

    for brick in halves:
        select_only(brick)
        remesh = brick.modifiers.new("RemeshFix", "REMESH")
        remesh.mode = "VOXEL"
        remesh.voxel_size = delta_b
        remesh.adaptivity = 0
        apply_modifier(brick, remesh)

        decimate = brick.modifiers.new(name="Decimate", type="DECIMATE")
        decimate.decimate_type = "UNSUBDIV"
        decimate.iterations = 6
        apply_modifier(brick, decimate)

        for polygon in brick.data.polygons:
            polygon.use_smooth = True
        unwrap_and_scale_uv(brick, scale=2.0)

    for name in (COLLECTION_INTACT, COLLECTION_SIDE_A, COLLECTION_SIDE_B):
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        for brick in collection.objects:
            if brick.name.startswith("Mortar") or "Bevel" in brick.modifiers:
                continue
            bevel = brick.modifiers.new(name="Bevel", type="BEVEL")
            bevel.width = 0.001
            bevel.limit_method = "ANGLE"
            bevel.angle_limit = math.radians(30)
            bevel.segments = 1

            subsurf = brick.modifiers.new(name="Subdivision", type="SUBSURF")
            subsurf.subdivision_type = "CATMULL_CLARK"
            subsurf.levels = 2
            subsurf.render_levels = 2


def add_mortar_geometry_nodes(mortar: bpy.types.Object, library: Path,
                              probability: float) -> None:
    """Attach the geometry-node groups that move mortar vertices around the bricks."""

    if mortar is None or random.random() >= probability:
        return

    broken = bpy.data.collections.get(COLLECTION_BROKEN)
    has_broken = bool(broken and broken.objects)

    wanted = ["Mortar_Connection_Brick"]
    if has_broken:
        wanted.append("Mortar_Connection_BrokenBrick")

    try:
        with bpy.data.libraries.load(str(library), link=False) as (source, target):
            available = list(getattr(source, "node_groups", []))
            target.node_groups = [n for n in wanted
                                  if n in available and n not in bpy.data.node_groups]
    except Exception as error:
        print(f"Failed to load mortar geometry nodes: {error}")
        return

    brick_group = bpy.data.node_groups.get("Mortar_Connection_Brick")
    broken_group = bpy.data.node_groups.get("Mortar_Connection_BrokenBrick")

    if brick_group is not None:
        info = brick_group.nodes.get("Collection Info")
        intact = bpy.data.collections.get(COLLECTION_INTACT)
        if info is not None and intact is not None:
            info.inputs[0].default_value = intact
        modifier = mortar.modifiers.new(name="Mortar_Connection_Brick", type="NODES")
        modifier.node_group = brick_group

    if has_broken and broken_group is not None:
        info = broken_group.nodes.get("Collection Info")
        if info is not None:
            info.inputs[0].default_value = broken
        modifier = mortar.modifiers.new(name="Mortar_Connection_BrokenBrick", type="NODES")
        modifier.node_group = broken_group


def rejoin_broken_bricks(records: List[dict]) -> None:
    """Join the separated halves of each cut brick back into one labelled instance."""

    for record in records:
        parts = [p for p in record["parts"] if p and p.name in bpy.data.objects]
        if len(parts) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for part in parts:
                part.select_set(True)
            bpy.context.view_layer.objects.active = parts[0]
            bpy.ops.object.join()
            record["joined"] = bpy.context.active_object
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        elif len(parts) == 1:
            record["joined"] = parts[0]


def freeze_roughb_random_input(obj: bpy.types.Object, index: int) -> Optional[float]:
    """Replace Object Info Random with one explicit value on a broken brick.

    A shrink-wrap helper is a different object and therefore receives a
    different Object Info Random value.  Freezing only the RoughB group input on
    the target material lets its helper copy reproduce the same displacement.
    """

    source = obj.active_material
    if source is None or not source.use_nodes or not source.node_tree:
        return None
    material = source.copy()
    material.name = f"{source.name}_Broken_{index:03d}"
    value = random.random()
    changed = False
    for node in material.node_tree.nodes:
        if node.type != "GROUP" or node.node_tree is None:
            continue
        if re.sub(r"\.\d+$", "", node.node_tree.name) != "NodeGroup_RoughB":
            continue
        value_input = node.inputs.get("Value")
        if value_input is None:
            continue
        for link in list(material.node_tree.links):
            if link.to_socket == value_input:
                material.node_tree.links.remove(link)
        value_input.default_value = value
        changed = True
    if not changed:
        bpy.data.materials.remove(material)
        return None
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.active_material = material
    return value


def add_mask_support_objects(records: List[dict], library: Path,
                             shrinkwrap_offset: float) -> None:
    """Add shrink-wrapped helper meshes that stabilise the broken-brick label regions.

    The Boolean cut and the following voxel remesh can leave narrow gaps between
    the two halves of a broken brick. A copy of the brick template is shrunk onto
    each rejoined instance so that the label pass returns one continuous
    broken-brick region.
    """

    if not records:
        return

    if shrinkwrap_offset < 0:
        raise ValueError("mask_shrinkwrap_offset cannot be negative.")
    ensure_collection(COLLECTION_MASK_FIX)
    template_names = {
        record.get("mask_template", MASK_TEMPLATE_STANDARD)
        for record in records
    }
    templates = {
        name: append_object_by_prefix(library, name)
        for name in sorted(template_names)
    }
    for template in templates.values():
        if template is not None:
            template.hide_viewport = True
            template.hide_render = True
    created = 0
    created_by_template: Dict[str, int] = {}

    for index, record in enumerate(records, start=1):
        template_name = record.get("mask_template", MASK_TEMPLATE_STANDARD)
        template = templates.get(template_name)
        if template is None:
            print(f"Mask support skipped: template not found: {template_name}")
            continue
        target = record.get("joined")
        if target is None:
            continue
        live = bpy.data.objects.get(target.name)
        if live is None:
            continue
        frozen_random = freeze_roughb_random_input(live, index)

        helper = template.copy()
        if template.data:
            helper.data = template.data.copy()
        helper.name = f"BrokenBrickMaskSupport_{index:03d}"
        helper["mask_target_name"] = live.name
        if frozen_random is not None:
            helper["roughb_random_value"] = frozen_random
        helper.location = record["position"].copy()
        helper.rotation_euler = record["rotation"].copy()
        helper.scale = live.scale.copy()
        ensure_collection(COLLECTION_MASK_FIX).objects.link(helper)

        shrinkwrap = helper.modifiers.new(name="ShrinkwrapToBrokenBrick", type="SHRINKWRAP")
        shrinkwrap.target = live
        shrinkwrap.wrap_method = "NEAREST_SURFACEPOINT"
        shrinkwrap.wrap_mode = "INSIDE"
        shrinkwrap.offset = shrinkwrap_offset

        helper.hide_viewport = True
        helper.hide_render = True
        created += 1
        created_by_template[template_name] = created_by_template.get(template_name, 0) + 1

    for template in templates.values():
        if template is not None and template.name in bpy.data.objects:
            bpy.data.objects.remove(template, do_unlink=True)
    detail = ", ".join(
        f"{name}: {count}" for name, count in sorted(created_by_template.items()))
    print(f"Mask support objects created: {created} ({detail}); "
          f"shrink-wrap offset: {shrinkwrap_offset:.4f}")


def merge_side_collections() -> None:
    """Return the side collections to the intact-brick collection and clean up."""

    intact = ensure_collection(COLLECTION_INTACT)
    for name in (COLLECTION_SIDE_A, COLLECTION_SIDE_B):
        collection = bpy.data.collections.get(name)
        if collection is None:
            continue
        for obj in list(collection.objects):
            for current in list(obj.users_collection):
                current.objects.unlink(obj)
            intact.objects.link(obj)

    for name in (COLLECTION_CRACK_BRICK, COLLECTION_CRACK_MORTAR,
                 COLLECTION_SIDE_A, COLLECTION_SIDE_B, COLLECTION_MOVED):
        delete_collection(name)


# --------------------------------------------------------------------------- #
# Crack label volume and surrounding scene
# --------------------------------------------------------------------------- #

def build_crack_label_volume(extent: dict) -> bpy.types.Object:
    """Create the volume whose visible surface becomes the crack label region.

    The volume sits just inside the mortar block. Only the parts exposed by the
    Boolean cut are visible to the camera, so the label pass renders exactly the
    open crack.
    """

    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(extent["centre_x"], 0, extent["centre_z"]))
    label = bpy.context.active_object
    label.name = "Crack_Label"
    label.scale = (extent["size_x"] - 0.005,
                   extent["size_y"] - 0.005,
                   extent["size_z"] - 0.02)

    move_to_collection(label, COLLECTION_CRACKS)
    label.hide_viewport = True
    label.hide_render = True
    return label


def build_backing_volume(extent: dict) -> None:
    """Create the opaque backing behind the wall so that the crack has no see-through."""

    bpy.ops.mesh.primitive_cube_add(
        size=1, location=(extent["centre_x"], 0.45, extent["centre_z"]))
    backing = bpy.context.active_object
    backing.name = "Backing"
    backing.scale = (extent["size_x"], extent["size_y"], extent["size_z"])

    mortar = bpy.data.objects.get("Mortar")
    if mortar and mortar.data.materials:
        backing.data.materials.append(mortar.data.materials[0])
    else:
        print("Backing left untextured: mortar material not available.")


def place_shadow_casters(library: Path, names: Iterable[str]) -> None:
    """Place the rectangular shadow-casting objects outside the camera view."""

    for name in names:
        with bpy.data.libraries.load(str(library), link=False) as (source, target):
            target.objects = [name] if name in source.objects else []
        for obj in target.objects:
            if obj is None:
                continue
            bpy.context.collection.objects.link(obj)
            obj.location = (random.uniform(0.35, 1.55), -2, 0)


def randomise_world_environment(hdri_dir: Path, strength_range: Tuple[float, float]) -> None:
    """Select one environment map and randomise its rotation and strength."""

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    environment = nodes.new("ShaderNodeTexEnvironment")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")

    tex_coord.location = (-800, 0)
    mapping.location = (-600, 0)
    environment.location = (-400, 0)
    background.location = (-100, 0)
    output.location = (100, 0)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    if not hdri_dir.is_dir():
        raise RuntimeError(f"Environment map directory not found: {hdri_dir}")
    maps = sorted(p for p in hdri_dir.iterdir() if p.suffix.lower() == ".exr")
    if not maps:
        raise RuntimeError(f"No .exr environment map in {hdri_dir}")

    chosen = random.choice(maps)
    environment.image = bpy.data.images.load(str(chosen), check_existing=True)
    environment.projection = "EQUIRECTANGULAR"

    mapping.inputs["Rotation"].default_value[2] = math.radians(random.uniform(0, 360))
    background.inputs["Strength"].default_value = random.uniform(*strength_range)


def rerandomise_environment(shadow_casters: Iterable[str]) -> None:
    """Vary the environment rotation, strength and shadow-caster positions."""

    world = bpy.context.scene.world
    if world and world.node_tree:
        nodes = world.node_tree.nodes
        if "Mapping" in nodes:
            nodes["Mapping"].inputs["Rotation"].default_value.z = \
                math.radians(random.uniform(0, 180))
        if "Background" in nodes:
            nodes["Background"].inputs["Strength"].default_value = random.uniform(1, 2)
    for name in shadow_casters:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.location = (random.uniform(0.35, 1.75), -2, 0)


def set_background_strength(strength: float) -> None:
    world = bpy.context.scene.world
    if world is None or not world.node_tree:
        return
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            node.inputs["Strength"].default_value = strength
            return


def read_background_strength() -> Optional[float]:
    world = bpy.context.scene.world
    if world is None or not world.node_tree:
        return None
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            return node.inputs["Strength"].default_value
    return None


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def look_at(obj: bpy.types.Object, target: Sequence[float]) -> None:
    direction = Vector(target) - obj.location
    if direction.length == 0:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def apply_random_roll(camera: bpy.types.Object) -> None:
    draw = random.random()
    if draw < 0.30:
        roll = random.uniform(15.0, 25.0)
    elif draw < 0.60:
        roll = random.uniform(-25.0, -15.0)
    else:
        roll = random.uniform(-5.0, 5.0)
    camera.rotation_euler.rotate_axis("Z", math.radians(roll))


def duplicate_camera_laterally(camera: bpy.types.Object, offset: float) -> bpy.types.Object:
    """Copy a camera and shift it along its own local X axis."""

    paired = camera.copy()
    paired.data = camera.data.copy()
    bpy.context.collection.objects.link(paired)
    local_right = camera.rotation_euler.to_matrix() @ Vector((1, 0, 0))
    paired.location = camera.location + local_right * offset
    return paired


def crack_key_targets(polylines: Sequence[Sequence[Tuple[float, float]]]
                      ) -> Optional[Dict[str, Tuple[float, float]]]:
    """Return three points along the crack used as camera targets."""

    points = [(p[0], p[1]) for path in polylines for p in path
              if isinstance(p, (list, tuple)) and len(p) >= 2]
    if len(points) < 3:
        return None
    points.sort(key=lambda p: p[1])
    count = len(points)
    return {
        "bottom": points[max(0, int(count * 0.60))],
        "middle": points[int(count * 0.50)],
        "top": points[min(count - 1, int(count * 0.40))],
    }


def build_cameras(extent: dict, crack_polylines: Sequence[Sequence[Tuple[float, float]]],
                  focal_range: Sequence[float], stereo_offset: float,
                  scale_factor: float) -> List[bpy.types.Object]:
    """Create three primary cameras and their laterally shifted paired cameras.

    The first camera looks upwards along the crack, the second looks downwards
    and the third is a front view of the wall centre. Each primary camera is
    duplicated with a lateral offset, which gives six rendered views per wall.
    """

    cameras: List[bpy.types.Object] = []

    width_x = extent["max_x"] - extent["min_x"]
    height_z = extent["max_z"] - extent["min_z"]
    centre_x = (extent["min_x"] + extent["max_x"]) / 2.0
    centre_z = (extent["min_z"] + extent["max_z"]) / 2.0

    margin_x = max(0.08, width_x * 0.08)
    margin_z = max(0.08, height_z * 0.08)
    target_min_x = extent["min_x"] + margin_x
    target_max_x = extent["max_x"] - margin_x
    target_min_z = extent["min_z"] + margin_z
    target_max_z = extent["max_z"] - margin_z

    targets = crack_key_targets(crack_polylines) or {
        "bottom": (centre_x, extent["min_z"] + height_z * 0.15),
        "middle": (centre_x, centre_z),
        "top": (centre_x, extent["min_z"] + height_z * 0.85),
    }
    for key, point in targets.items():
        targets[key] = (clamp(point[0], target_min_x, target_max_x),
                        clamp(point[1], target_min_z, target_max_z))

    offset_scene_units = stereo_offset * scale_factor

    def add_pair(camera: bpy.types.Object) -> None:
        cameras.append(camera)
        cameras.append(duplicate_camera_laterally(camera, offset_scene_units))

    def add_crack_camera(name: str, cam_x: float, cam_z: float,
                         look_x: float, look_z: float,
                         yaw_jitter_deg: float) -> None:
        cam_y = -2.25
        pos_min_x = extent["min_x"] + width_x * 0.03
        pos_max_x = extent["max_x"] - width_x * 0.03
        pos_min_z = extent["min_z"] + height_z * 0.02
        pos_max_z = extent["max_z"] - height_z * 0.02

        cam_x = clamp(cam_x, pos_min_x, pos_max_x)
        cam_z = clamp(cam_z, pos_min_z, pos_max_z)
        look_x = clamp(look_x, target_min_x, target_max_x)
        look_z = clamp(look_z, target_min_z, target_max_z)

        jitter = math.tan(math.radians(
            random.uniform(-yaw_jitter_deg, yaw_jitter_deg))) * abs(cam_y)
        cam_x = clamp(cam_x + jitter, pos_min_x, pos_max_x)

        bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
        camera = bpy.context.object
        camera.name = name
        look_at(camera, (look_x, 0.0, look_z))
        apply_random_roll(camera)
        camera.data.lens = random.uniform(focal_range[0], focal_range[1])
        add_pair(camera)

    def add_front_camera(name: str, target_x: float, target_z: float) -> None:
        cam_y = -2.25
        bpy.ops.object.camera_add(
            location=(clamp(target_x, target_min_x, target_max_x),
                      cam_y,
                      clamp(target_z, target_min_z, target_max_z)))
        camera = bpy.context.object
        camera.name = name
        look_at(camera, (target_x, 0.0, target_z))
        camera.data.lens = focal_range[0]
        add_pair(camera)

    add_crack_camera("P1",
                     cam_x=targets["bottom"][0],
                     cam_z=extent["min_z"] + height_z * 0.18,
                     look_x=targets["middle"][0],
                     look_z=extent["min_z"] + height_z * 0.38,
                     yaw_jitter_deg=5.0)
    add_crack_camera("P2",
                     cam_x=targets["top"][0],
                     cam_z=extent["min_z"] + height_z * 0.82,
                     look_x=targets["middle"][0],
                     look_z=extent["min_z"] + height_z * 0.62,
                     yaw_jitter_deg=5.0)
    add_front_camera("P3",
                     target_x=clamp(centre_x, target_min_x, target_max_x),
                     target_z=clamp(centre_z, target_min_z, target_max_z))

    return cameras


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def assign_pass_indices() -> None:
    for name, index in ((COLLECTION_BROKEN, PASS_BROKEN_BRICK),
                        (COLLECTION_INTACT, PASS_INTACT_BRICK)):
        collection = bpy.data.collections.get(name)
        if collection:
            for obj in collection.objects:
                obj.pass_index = index

    for name, index in (("Mortar", PASS_MORTAR),
                        ("Crack_Mesh", PASS_CRACK_MESH),
                        ("Crack_Label", PASS_CRACK_LABEL)):
        obj = bpy.data.objects.get(name)
        if obj:
            obj.pass_index = index


def configure_render(config: Config) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = config.resolution
    scene.render.resolution_y = config.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"


def configure_cycles(config: Config) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config.render_samples
    if config.use_optix:
        preferences = bpy.context.preferences.addons.get("cycles")
        if preferences is not None:
            try:
                preferences.preferences.compute_device_type = "OPTIX"
                for device in preferences.preferences.devices:
                    device.use = True
                scene.cycles.device = "GPU"
            except Exception as error:
                print(f"OptiX not available, falling back to CPU: {error}")


def setup_file_output_nodes(output_dir: Path, write_depth: bool) -> None:
    """Configure the compositor so that one render writes the RGB and depth outputs."""

    scene = bpy.context.scene
    scene.use_nodes = True
    scene.render.use_compositing = True
    bpy.context.view_layer.use_pass_z = True
    tree = scene.node_tree

    for name in ("FileOutput_PNG", "FileOutput_EXR"):
        node = tree.nodes.get(name)
        if node:
            tree.nodes.remove(node)

    render_layers = tree.nodes.get("Render Layers") or tree.nodes.new("CompositorNodeRLayers")
    rendering_group = None
    depth_group = None
    for node in tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            if node.node_tree.name == "Rendering":
                rendering_group = node
            elif node.node_tree.name == "Depth Map":
                depth_group = node

    png_output = tree.nodes.new("CompositorNodeOutputFile")
    png_output.name = "FileOutput_PNG"
    png_output.location = (900, 300)
    png_output.base_path = str(output_dir)
    png_output.format.file_format = "PNG"
    png_output.format.color_mode = "RGBA"
    png_output.format.color_depth = "8"
    png_output.file_slots.new("P")
    if rendering_group:
        tree.links.new(rendering_group.outputs[0], png_output.inputs["P"])
    else:
        tree.links.new(render_layers.outputs["Image"], png_output.inputs["P"])

    if write_depth:
        png_output.file_slots.new("D")
        if depth_group:
            tree.links.new(depth_group.outputs[0], png_output.inputs["D"])

        exr_output = tree.nodes.new("CompositorNodeOutputFile")
        exr_output.name = "FileOutput_EXR"
        exr_output.location = (900, 0)
        exr_output.base_path = str(output_dir)
        exr_output.format.file_format = "OPEN_EXR"
        exr_output.format.color_mode = "RGB"
        exr_output.format.color_depth = "32"
        exr_output.file_slots.new("depth")
        tree.links.new(render_layers.outputs["Depth"], exr_output.inputs["depth"])


def render_rgb_view(base_name: str, camera_index: int, output_dir: Path,
                    write_depth: bool) -> None:
    """Render one RGB view and give the compositor outputs their final names."""

    scene = bpy.context.scene
    # File Output compositor nodes write P/D/depth themselves.  ``write_still``
    # would additionally save scene.render.filepath as a black/raw image such as
    # ``<base>_1.png``; that file is neither an RGB result nor an annotation.
    bpy.ops.render.render(write_still=False)
    frame = f"{scene.frame_current:04d}"

    renames = {f"P{frame}.png": f"{base_name}_{camera_index}_P.png"}
    if write_depth:
        renames[f"D{frame}.png"] = f"{base_name}_{camera_index}_D.png"
        renames[f"depth{frame}.exr"] = f"{base_name}_{camera_index}_depth.exr"

    for source, target in renames.items():
        source_path = output_dir / source
        if source_path.exists():
            os.replace(source_path, output_dir / target)


def create_emission_variant(base: Optional[bpy.types.Material], suffix: str,
                            colour: Tuple[float, float, float],
                            preserve_procedural_cracks: bool = True
                            ) -> Optional[bpy.types.Material]:
    """Copy a material and replace its surface with a flat emission of one colour.

    The copy keeps the displacement and procedural-crack nodes of the original,
    so the label follows the same surface as the RGB render. When the material
    exposes an ``Apply_Crack`` node, the emission is routed through it so that the
    procedural fine cracks remain part of the crack label.
    """

    if base is None:
        return None

    variant = base.copy()
    variant.name = f"{base.name}{suffix}"
    if not variant.use_nodes or not variant.node_tree:
        return variant

    tree = variant.node_tree
    output = next((n for n in tree.nodes
                   if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
    if output is None:
        return variant

    for link in list(tree.links):
        if link.to_node == output and link.to_socket.name == "Surface":
            tree.links.remove(link)

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = output.location + Vector((-300, 0))
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = 1.0

    apply_crack = tree.nodes.get("Apply_Crack")
    if preserve_procedural_cracks and apply_crack is not None:
        tree.links.new(emission.outputs["Emission"], apply_crack.inputs[1])
        tree.links.new(apply_crack.outputs[0], output.inputs["Surface"])
        mask_value = tree.nodes.get("Mask_Value")
        if mask_value is not None:
            mask_value.outputs[0].default_value = 1.0
    else:
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return variant


def bind_displacement_coordinates(material: Optional[bpy.types.Material],
                                  target: Optional[bpy.types.Object]) -> None:
    """Evaluate a helper's RoughB displacement in its target brick's object space."""

    if material is None or target is None or not material.node_tree:
        return
    for node in material.node_tree.nodes:
        if node.type != "GROUP" or node.node_tree is None:
            continue
        if re.sub(r"\.\d+$", "", node.node_tree.name) != "NodeGroup_RoughB":
            continue
        local_group = node.node_tree.copy()
        local_group.name = f"NodeGroup_RoughB_{target.name}"
        node.node_tree = local_group
        for inner in local_group.nodes:
            if inner.bl_idname == "ShaderNodeTexCoord":
                inner.object = target


def material_emission_colour(material: Optional[bpy.types.Material],
                             fallback: Tuple[float, float, float]
                             ) -> Tuple[float, float, float]:
    """Read the unlinked emission colour from an asset material."""

    if material is None or not material.use_nodes or not material.node_tree:
        return fallback
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeEmission":
            continue
        colour = node.inputs.get("Color")
        if colour is not None and not colour.is_linked:
            value = colour.default_value
            return float(value[0]), float(value[1]), float(value[2])
    return fallback


def prepare_label_materials() -> None:
    """Replace every appearance material with its flat-colour label counterpart."""

    brick_base = None
    for name in (COLLECTION_INTACT, COLLECTION_BROKEN):
        collection = bpy.data.collections.get(name)
        if not collection:
            continue
        for obj in collection.objects:
            if obj.type == "MESH" and obj.data.materials:
                brick_base = obj.data.materials[0]
                break
        if brick_base:
            break

    mortar = bpy.data.objects.get("Mortar")
    mortar_base = mortar.data.materials[0] if mortar and mortar.data.materials else None

    intact_material = create_emission_variant(
        brick_base, "_Label_Intact", LABEL_COLOURS["intact_brick"])
    yellow_source = bpy.data.materials.get(LABEL_EMISSION_MATERIAL)
    yellow_colour = material_emission_colour(
        yellow_source, LABEL_COLOURS["crack"])
    mortar_material = create_emission_variant(
        mortar_base, "_Label", LABEL_COLOURS["mortar"])

    crack_material = bpy.data.materials.get("CrackLabel")
    if crack_material is None:
        crack_material = bpy.data.materials.new("CrackLabel")
        crack_material.use_nodes = True
        tree = crack_material.node_tree
        tree.nodes.clear()
        emission = tree.nodes.new("ShaderNodeEmission")
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission.inputs["Color"].default_value = (*yellow_colour, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    def assign(collection_name: str, material: Optional[bpy.types.Material]) -> None:
        if material is None:
            return
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            return
        for obj in collection.objects:
            if obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(material)

    assign(COLLECTION_INTACT, intact_material)
    broken = bpy.data.collections.get(COLLECTION_BROKEN)
    if broken is not None:
        for obj in broken.objects:
            source = obj.active_material
            material = create_emission_variant(
                source, f"_Label_Broken_{obj.name}",
                LABEL_COLOURS["broken_brick"])
            if material is not None and obj.type == "MESH":
                obj.data.materials.clear()
                obj.data.materials.append(material)
                obj.active_material = material
    # Each helper receives its own pure Emission_Yellow surface on top of a copy
    # of the brick material.  The copy keeps displacement, while a private
    # RoughB group evaluates Object coordinates against the actual target brick.
    # Do not route this surface through Apply_Crack: Emission_Yellow is the
    # geometric crack helper, not the procedural fine-crack shader.
    support = bpy.data.collections.get(COLLECTION_MASK_FIX)
    if support is not None:
        for helper in support.objects:
            target = bpy.data.objects.get(helper.get("mask_target_name", ""))
            target_material = target.active_material if target is not None else brick_base
            helper_material = create_emission_variant(
                target_material,
                f"_Label_Emission_Yellow_{helper.name}",
                yellow_colour,
                preserve_procedural_cracks=False,
            )
            bind_displacement_coordinates(helper_material, target)
            if helper_material is not None and helper.type == "MESH":
                helper.data.materials.clear()
                helper.data.materials.append(helper_material)

    if mortar and mortar_material and mortar.type == "MESH":
        mortar.data.materials.clear()
        mortar.data.materials.append(mortar_material)

    crack_label = bpy.data.objects.get("Crack_Label")
    if crack_label and crack_label.type == "MESH":
        crack_label.data.materials.clear()
        crack_label.data.materials.append(crack_material)


def render_label_views(base_name: str, cameras: Sequence[bpy.types.Object],
                       output_dir: Path, samples: int) -> None:
    """Render the flat-colour label pass for every camera used in the RGB pass."""

    scene = bpy.context.scene
    crack_label = bpy.data.objects.get("Crack_Label")
    if crack_label:
        crack_label.hide_viewport = False
        crack_label.hide_render = False

    prepare_label_materials()
    set_collection_visibility(COLLECTION_MASK_FIX, viewport=False, render=False)
    set_background_strength(0.0)

    scene.use_nodes = False
    scene.render.use_compositing = False
    # Keep Cycles for the label pass.  The brick material uses true shader
    # displacement; Eevee and shrink-wrap helpers only see the undisplaced base
    # mesh, which can expose Crack_Label where the RGB surface is actually brick.
    # The copied emission materials retain the original displacement output, so
    # a low-sample Cycles pass gives aligned silhouettes and remains inexpensive.
    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(1, int(samples))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    for index, camera in enumerate(cameras, start=1):
        scene.camera = camera
        scene.render.filepath = str(output_dir / f"{base_name}_{index}_M.png")
        bpy.ops.render.render(write_still=True)
        print(f"Rendered label: {scene.render.filepath}")


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #

def generate_scene(crack_file: Path, wall_file: Path, config: Config,
                   preset: dict) -> Optional[str]:
    """Build, randomise and render one complete masonry scene."""

    library = config.resolved("asset_library")
    hdri_dir = config.resolved("hdri_directory")
    output_dir = config.resolved("output_directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = preset["scale_factor"]
    bricks = read_wall_file(wall_file, scale)
    if not bricks:
        print(f"No bricks read from {wall_file.name}; scene skipped.")
        return None
    crack_polylines = read_crack_file(crack_file, scale)

    clear_scene()

    displacement = bpy.data.textures.new("CrackClouds", type="CLOUDS")
    displacement.noise_scale = 0.0001

    # Stage (b): masonry geometry and procedural materials.
    with bpy.data.libraries.load(str(library), link=False) as (source, target):
        wanted_materials = [*BRICK_MATERIALS, LABEL_EMISSION_MATERIAL]
        target.materials = [m for m in wanted_materials if m in source.materials]
    brick_material_name = random.choice(BRICK_MATERIALS)
    brick_material = bpy.data.materials.get(brick_material_name)
    randomise_brick_displacement(brick_material,
                                 config.brick_displacement_scale_range)

    templates = load_brick_templates(library)
    place_bricks(bricks, templates, brick_material, config)
    mortar, extent = build_mortar(bricks, library, brick_material_name)

    # Stages (c) and (d): Boolean crack geometry and subtraction.
    use_boolean = preset["boolean_geometry"] and bool(crack_polylines)
    bevel_depth = 0.0
    brick_halves: List[bpy.types.Object] = []
    cut_records: List[dict] = []

    if use_boolean:
        bevel_depth = random.uniform(preset["bevel"][0], preset["bevel"][1]) * 0.01

        build_cutting_object(crack_polylines, "Crack_Brick", COLLECTION_CRACK_BRICK,
                             bevel_depth, base_radius=1.00,
                             voxel_size=preset["delta_c"])
        build_cutting_object(crack_polylines, "Crack_Mortar", COLLECTION_CRACK_MORTAR,
                             bevel_depth * 1.1, base_radius=1.05,
                             voxel_size=preset["delta_c"])

        brick_cutters = bpy.data.collections.get(COLLECTION_CRACK_BRICK)
        mortar_cutters = bpy.data.collections.get(COLLECTION_CRACK_MORTAR)

        if brick_cutters and brick_cutters.objects:
            stretch_cutters_through_wall(COLLECTION_CRACK_BRICK, bevel_depth)
            brick_halves, cut_records = cut_broken_bricks(brick_cutters)

        if mortar and mortar_cutters and mortar_cutters.objects:
            cut_mortar(mortar, mortar_cutters, bevel_depth)

        # Stage (d continued): relative movement of the two crack sides.
        displace_crack_side(preset["motion"])

        mortar = rebuild_mortar(preset["delta_m"]) or mortar
        polish_bricks(brick_halves, preset["delta_b"])
        add_mortar_geometry_nodes(mortar, library, config.mortar_geometry_node_probability)
        rejoin_broken_bricks(cut_records)
        add_mask_support_objects(cut_records, library,
                                 config.mask_shrinkwrap_offset)
    else:
        polish_bricks([], preset["delta_b"] or 0.009)
        add_mortar_geometry_nodes(mortar, library, config.mortar_geometry_node_probability)

    merge_side_collections()

    if extent:
        build_crack_label_volume(extent)
        build_backing_volume(extent)

    # Stage (e): domain randomisation.
    place_shadow_casters(library, config.shadow_caster_objects)
    randomise_world_environment(hdri_dir, config.hdri_strength_range)

    cameras = build_cameras(extent, crack_polylines, preset["focal"],
                            config.stereo_offset_m, scale) if extent else []
    if not cameras:
        print("No cameras created; scene skipped.")
        return None

    # Stages (f) and (g): RGB render followed by the aligned label render.
    assign_pass_indices()
    configure_render(config)
    configure_cycles(config)

    base_name = unique_basename(output_dir, preset["label"])
    original_strength = read_background_strength()

    set_collection_visibility(COLLECTION_MASK_FIX, viewport=True, render=True)
    setup_file_output_nodes(output_dir, config.write_depth)

    scene = bpy.context.scene
    for pair_start in range(0, len(cameras), 2):
        for eye, camera in enumerate(cameras[pair_start:pair_start + 2], start=1):
            scene.camera = camera
            render_rgb_view(base_name, pair_start + eye, output_dir, config.write_depth)
        # Vary the illumination between stereo pairs, not within a pair, so that
        # the two views of one pair remain consistent.
        if pair_start + 2 < len(cameras):
            rerandomise_environment(config.shadow_caster_objects)

    render_label_views(base_name, cameras, output_dir, config.label_render_samples)

    if original_strength is not None:
        set_background_strength(original_strength)

    print(f"Scene complete: {base_name} ({len(cameras)} views)")
    return base_name


def main(argv: Sequence[str]) -> int:
    config, presets = load_config(argv)
    preset = resolve_preset(config, presets)

    if config.seed is not None:
        random.seed(config.seed)

    coordinate_dir = config.resolved("coordinate_directory")
    scenes = discover_scene_inputs(coordinate_dir)
    if not scenes:
        print(f"No wall and crack file pairs found in {coordinate_dir}")
        return 1

    scenes = scenes[:config.max_scenes]
    print(f"Generating {len(scenes)} scene(s) with preset {preset['label']}")

    produced = 0
    for index, (crack_file, wall_file) in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] {wall_file.name}")
        if generate_scene(crack_file, wall_file, config, preset):
            produced += 1

    print(f"Finished: {produced} of {len(scenes)} scenes rendered.")
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(arguments))
