# Blender scene generation and automatic annotation

`wall_generator.py` builds the masonry scene from the coordinate files produced in
`01_crack_path_generation/`, represents the crack, randomises the scene and renders aligned
RGB images and instance labels. `masks_to_yolo_polygons.py` converts the rendered label images
into YOLO segmentation annotations.

## Requirements

Blender 4.5 LTS with Cycles. GPU rendering through OptiX is used for the reported production
rates but is not required. The script runs inside Blender, not in a standalone Python
interpreter.

A Blender asset library file supplies the reusable geometry, materials and geometry nodes:

| Datablock | Type | Role |
|---|---|---|
| `Metric Bricks (Modern Standard)` | Object | Stretcher mesh, 0.215 x 0.102 x 0.065 m |
| `Metric Half Bricks (Modern Standard)` | Object | Header mesh, 0.102 x 0.102 x 0.065 m |
| `... .Type01` to `... .Type04` | Object | Shape variants of each base mesh |
| `Metric Bricks (Modern Standard) Mask` | Object | Stretcher/standard-brick label support template |
| `Metric Half Bricks (Modern Standard) Mask` | Object | Header/half-brick label support template |
| `Brick_01_ML` to `Brick_03_MD` | Material | Fired clay, calcium silicate, Staffordshire blue and limestone appearances, including the procedural fine cracks |
| `Mortar_01`, `Mortar_02` | Material | Grey and yellowish mortar |
| `Mortar_Connection_Brick`, `Mortar_Connection_BrokenBrick` | Node group | Move the mortar vertices around the brick edges |
| `Rendering`, `Depth Map` | Node group | Compositor groups for the RGB and depth outputs |
| `Roofing` | Object | Shadow caster kept outside the camera view. It is one object built
from two rectangular blocks that move together, which the manuscript describes as two
rectangular shadow-casting objects. |

`Dataset_Generator.blend` is the render template and `Asset_Library.blend` supplies the reusable
objects, materials and node groups. Both files are included in this directory.

The environment maps are equirectangular `.exr` files in a separate directory and are not
included in Git. Download suitable HDRIs from
[BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree%3Ahdr-outdoor) or
[Poly Haven](https://polyhaven.com/hdris).

## Running

Write a template configuration, edit the paths, then run the generator:

    blender --background --python wall_generator.py -- --write-default-config config.json

    blender --background Dataset_Generator.blend --python wall_generator.py -- \
        --config config.json

Every configuration entry can also be given on the command line, for example
`--view-range far --crack-width-label 10to30mm --max-scenes 20`.

Convert the rendered labels afterwards:

    python masks_to_yolo_polygons.py --masks <output_dir> --output <labels_dir>

For the external [CSG2 v1](https://github.com/DavidHidde/cracked-surface-generation/tree/v1)
binary crack masks:

    python masks_to_yolo_polygons.py --masks <csg2_masks> --output <labels_dir> \
        --binary --class-id 2 --pattern "*.png"

## Inputs

Each scene reads one pair of coordinate files, named `<PREFIX>_wall_<N>.txt` and
`<PREFIX>_crack_<N>.txt`.

The wall file holds one brick per line as `x z type rotation flag`, with coordinates in metres
on the wall plane. The damage flag records whether the brick is crossed by the crack path and,
for intact bricks, which side of the crack it lies on:

| Flag | Meaning |
|---:|---|
| 7 | brick crossed by the sampled crack path |
| 8 | intact brick |
| 9 | intact brick on the first side of the crack |
| 10 | intact brick on the second side of the crack |

The crack file holds one polyline per line as a sequence of `(x, z)` pairs in metres. The first
line is the main path and every following line is a branch.

## Generation presets

`presets.json` combines one of three view ranges with one of five crack-width labels. Each
preset stores the voxel-remeshing resolutions for the crack mesh, the affected bricks and the
mortar, the bevel-depth range of the crack curve and the focal-length range of the camera.

Cracks narrower than 3 mm carry no Boolean geometry; they are produced by the procedural
material textures and their voxel and bevel entries are null. Cracks of at least 3 mm use
Boolean modelling based on the sampled coordinates. Narrower categories use a smaller voxel
size, because a voxel suited to a wide crack removes fine detail from a narrow one and can
break it into discontinuous segments.

The scene is built at `scale_factor` scene units per metre of wall. A sampled bevel value `v`
gives a cutting radius of `v * 0.01` scene units, and with the local radius multiplier
`r = 1.0 + e`, `e ~ U(-0.2, 0.2)`, the modelled crack width is `4 * v * r` millimetres.

`crack_side_motion` controls how the two sides of the crack move relative to each other:
`stay` leaves them in place, `translation` applies a small in-plane offset, and `settling`
adds a rotation as well as a larger offset.

## Output

Each scene produces six views from three primary cameras and their laterally shifted paired
cameras.

| File | Content |
|---|---|
| `<base>_<n>_P.png` | RGB render, Cycles |
| `<base>_<n>_M.png` | Flat-colour label render, green intact brick, red broken brick, blue mortar, yellow crack |
| `<base>_<n>_D.png` | Depth visualisation |
| `<base>_<n>_depth.exr` | Linear depth, 32-bit |

`<base>` is the generation label followed by a random seven-character identifier. The label
encodes the crack-side motion, the view range and the crack-width category, so the generation
condition of every image can be recovered from its filename.

The compositor writes the named RGB/depth products directly; the scene's raw render is not
written, so no extra black `<base>_<n>.png` files are produced. The label pass reuses the scene
state, camera and procedural crack patterns of the RGB pass and only replaces the surface
shaders. It uses a low-sample Cycles render so `NodeGroup_RoughB` material displacement is
evaluated for the mask silhouette as well as the RGB image. This prevents the yellow crack
volume from leaking through the undisplaced broken-brick base mesh. The background strength is
set to zero during this pass, which makes the label independent of lighting and shadows.
The shrink-wrapped broken-brick helpers remain in use. Their surface colour is read from the
asset-library material `Emission_Yellow`, while the generated helper material is based on the
active brick material so it inherits the identical `NodeGroup_RoughB` displacement chain and
the scene-wide displacement scale. Each helper evaluates that displacement in its target
broken brick's object-coordinate space; its surface stays a pure `Emission_Yellow` shader and
is not passed through the procedural `Apply_Crack` branch. Because Blender's `Object Info`
Random output differs between the target and helper objects, each broken brick's RoughB random
input is frozen to an explicit per-brick value before RGB rendering and copied to its helper.
The helper shape is selected from the standard- or half-brick mask template to match the target.
`mask_shrinkwrap_offset` controls how far the helper stays inside the target and defaults to
0.03 scene units (increased from the former fixed value of 0.01).

For each generated scene, `brick_displacement_scale_range` samples one displacement Scale value
(default 0.05 to 0.10) and applies that same value to `NodeGroup_RoughB` for every brick in the
scene. `label_render_samples` controls the low-cost flat-colour Cycles pass and defaults to 4.

## Differences from the internal production script

This release consolidates the internal generator without changing the modelled geometry:

- Paths, preset selection and batch size come from a configuration file or the command line
  instead of module-level constants.
- The presets are keyed on view range and crack-width label, and are held in `presets.json`.
- A duplicated `add_spline` definition, in which the second shadowed the first, is replaced by
  one function.
- The RGB pass no longer renders every camera twice. The internal script ran a full render loop
  and then repeated it as stereo pairs, which doubled the render time without changing the
  output.
- The truncated final statement that restored the world background strength is completed.
- Output resolution is set explicitly to 640 x 640 rather than inherited from the .blend file.
