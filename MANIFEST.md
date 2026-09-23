# Release manifest

Inventory of this repository and its relationship to the manuscript. 36 files, approximately
9.41 MiB.

| Directory | Files | Manuscript section |
|---|---:|---|
| `01_crack_path_generation/` | 8 + 2 READMEs | Mask preprocessing and learning-guided crack-coordinate generation |
| `02_blender_generation/` | 5 + README | Blender scene generation, two required `.blend` files and automatic annotation |
| `03_prior_evaluation/` | 10 + README | Evaluation of the learned spatial prior; prior-path results |
| `04_downstream_training/` | 1 + README | Training and evaluation protocol; sample selection |
| `05_datasets/` | README | Reserved for the released datasets |

---

## 1. `01_crack_path_generation/`

| File | Source in the working repository |
|---|---|
| `procedural_layout.py` | `Crack On Wall/Placing_Brick_V3.py` |
| `refine_crack_path.py` | `Crack On Wall/Refine_Carck.py` |
| `generate_crack_coordinates.py` | Rebuilt for release from `Crack On Wall/infer_crack_unet.py` |
| `preprocessing/mask_preprocessing.py` | The three retained mask-rasterisation, reconstruction and verification commands |
| `unet/prepare.py` | the U-Net training code |
| `unet/models.py` | the U-Net training code |
| `unet/train.py` | the U-Net training code |
| `unet/predict.py` | the U-Net training code |

The four U-Net files are the current training code. No trained weights are distributed; the
training run writes its checkpoint into the configured workspace.

`preprocessing/mask_preprocessing.py` contains only the three commands required to create and
validate the reconstructed masks read by `unet/prepare.py`. The paper-statistics,
parameter-calibration and manuscript-figure commands from the former analysis module are not
part of this release.

`generate_crack_coordinates.py` connects this stage to the Blender stage. The earlier inventory
had no file that wrote the `*_wall_*.txt` and `*_crack_*.txt` pairs that
`02_blender_generation/wall_generator.py` reads, so the two stages could not be run one after
the other. The rewrite keeps the geometry of the working script and changes four things:

| Change | Detail |
|---|---|
| Probability map | Four aligned flip predictions fused by `03_prior_evaluation/path_sampler.py`, as Section 3.2.3 specifies. The working script averages eight Monte Carlo dropout passes instead, so the released BCG images were generated from that map, not this one |
| Sampler | `path_sampler.generate`, the released module, in place of the working script's own walk. The in-brick rule is then the 0.7 to 1.4 interval in `crossing_rules.py`; the working script uses 0.8 to 1.4 |
| Damage flag | A brick is flagged broken when the crack runs at least `--min-interior-depth` pixels inside it for at least `--min-crossing-pixels` pixels. The working script flags every brick the path touches, including the ones it only grazes while travelling along a joint |
| Configuration | Every path comes from `config.yaml`; the view presets, the branch probability and both damage thresholds are command-line options. No absolute path remains in the file |

The crop is unchanged in placement and changed only in extent: it is anchored by its lower-left
corner at the same wall coordinate, and its side is now bounded below by the span the bricks
occupy, so that no draw can leave a course outside it. Measured over 300 draws of the step
preset the wall raster is 778 to 942 pixels tall and the crop 779 to 921, and every draw holds
all ten courses.

## 2. `02_blender_generation/`

| File | Origin |
|---|---|
| `wall_generator.py` | Rebuilt for release from the internal production script |
| `presets.json` | New; the preset table on the manuscript's view-range and crack-width axes |
| `masks_to_yolo_polygons.py` | New; label-image to YOLO polygon conversion |
| `Dataset_Generator.blend` | Render template with the production render and compositor settings |
| `Asset_Library.blend` | Reusable masonry geometry, materials, mask helpers and node groups |

`wall_generator.py` keeps the modelled geometry of the internal script and changes only its
structure and configuration. The changes are listed at the end of
`02_blender_generation/README.md`. Two of them fix defects: a duplicated `add_spline`
definition in which the second shadowed the first, and a render loop that rendered every camera
twice. A truncated final statement is also completed.

`masks_to_yolo_polygons.py` closes a gap in the earlier inventory. The manuscript states that
the label renders and the external CSG2 binary masks are converted to YOLO polygons by the same
procedure, but no converter was present in the working repository. This script implements that
conversion for both inputs.

## 3. `03_prior_evaluation/`

| File | Role |
|---|---|
| `workspace.py` | Locations from `config.yaml` and small IO helpers |
| `crossing_rules.py` | The 0.7 to 1.4 in-brick traversal rule and the 8-neighbourhood |
| `task_data.py` | Layout, reference centreline and brick geometry for one image |
| `metrics.py` | Fixed-tolerance and width-adaptive centreline scoring, and path diversity |
| `path_sampler.py` | The sampler, as five composed stages |
| `evaluate.py` | The four pipeline stages |
| `evaluation_loop.py` | The paired test loop |
| `report.py` | Summary tables and `REPORT.md` |
| `sweep_gamma_lambda.py` | The goal-bias and inertia grid |
| `build_prior_ablation.py` | The eighteen-setting table |

The reported method was developed over several rounds. The working repository held that history
as a snapshot of 35 modules carrying version suffixes, in which each round's driver read the
previous round's frozen outputs from disk, and four of the referenced rounds had no driver code
at all. That snapshot could not be run by anyone who did not already hold every round's outputs.

This release contains the final method as one self-contained pipeline. The rewrite:

| Change | Detail |
|---|---|
| Sampler merged | The five composition stages, which all execute on every call, are one module with descriptive names. Verified: on synthetic layouts the merged sampler returns paths and metadata bit-identical to the original for all five seeds |
| Metrics merged | Each helper was taken from the single module that defined its final version; the function bodies are copied verbatim |
| Driver rewritten | `prepare` builds the protocol from the data manifests instead of inheriting it from an earlier round. `calibrate` screens the three fusion schemes on their own calibration scores; the original round's retention constraints, which compared against a round that is not released, are not reproduced |
| History removed | The superseded round drivers, the reports for those rounds, and the five modules unreachable from the final pipeline are not included |

No file name or module carries a version suffix. The complete pipeline was run end to end on a
synthetic workspace, from `prepare` through `report`.

## 4. `04_downstream_training/`

The downstream experiments use unmodified public implementations, so the training settings are
documented in `README.md` rather than shipped as one script per condition. That README holds
the two training schedules, the augmentation configurations, the three uncertainty scores, the
candidate-scoring settings and the composition of the selection-strategy groups.

`example_yolo_active_learning.py` is one complete acquisition loop, from
`NCC/YOLO_AL_Addition8_Group1_R200_R400_R600.py`. Every path and training setting comes from the
`downstream:` block of `config.yaml`, with a `BCG_DOWNSTREAM_<NAME>` environment override for
each one. The script stops with a message naming the missing entry if `experiment_root` or
`synthetic_pool` is unset. No cluster path remains in the file.

Not included: the other 60-epoch YOLO, Mask R-CNN, Mask2Former and augmentation-control scripts,
which differ from the example only in the uncertainty score, the real-image base sizes, the
acquisition points and whether a generation-label quota is applied.

The 400-epoch data-source and hybrid-augmentation runs were executed on a different cluster and
no script for them is present in the working repository. Their settings are documented in
`README.md`, but the repository holds no code for that schedule.

## 5. `05_datasets/`

Holds no data, and points at the dataset repository instead. The BCG images are released as a
separate repository, so that the code and the data can be downloaded independently and the
dataset can be versioned without enlarging every clone of this one. The dataset repository
carries the directory layout, the class order, what the filenames encode, the checksums and a
verification script.

## 6. Trained weights

None are distributed. `01_crack_path_generation/unet/` trains the crack-probability U-Net and
writes `<workspace_root>/runs/crack_unet/best.pt`, which every later stage reads through
`unet_checkpoint` in `config.yaml`.

Downstream segmentation weights are not included. A model-only checkpoint is approximately
140 MB for YOLOv8x-seg, 430 MB for Mask R-CNN X-101-FPN, or 270 MB for Mask2Former Swin-S, and
optimizer-state checkpoints are larger. Retaining checkpoints from several hundred runs would
add tens of gigabytes of binary files. The repository provides the training protocols and the
runnable acquisition-loop example instead.

## 7. Release checks

- No Chinese text remains anywhere in the repository.
- No file name, module or identifier carries a development version suffix.
- No absolute path and no cluster account name appears in any file. Every location comes from
  `config.yaml` through `bcg_config.py`.
- Every cross-file import resolves within the released layout.
- Required mask preprocessing is isolated in `01_crack_path_generation/preprocessing/`; the
  paper-only analysis code is not included.
