# Learning-guided crack-coordinate generation

This stage produces the wall and crack coordinate files that the Blender stage renders. It
combines a procedural masonry layout, a crack-probability model learned from real masks, and a
probability-guided path sampler.

## Contents

| File | Role |
|---|---|
| `preprocessing/mask_preprocessing.py` | Rasterises YOLO labels, reconstructs crack-free layouts and verifies the saved products |
| `procedural_layout.py` | Builds the 2D masonry layout and rasterises it |
| `generate_crack_coordinates.py` | Writes the wall and crack coordinate files for the Blender stage |
| `unet/prepare.py` | Builds the optimisation and internal-validation split from the reconstructed masks |
| `unet/models.py` | The crack-probability U-Net |
| `unet/train.py` | Training loop and checkpoint selection |
| `unet/predict.py` | Single-view inference on one layout, writing a probability array and a display image |
| `refine_crack_path.py` | Median and Savitzky-Golay smoothing of the exported coordinates |

The reconstruction is the `preview-masks` command in
`preprocessing/mask_preprocessing.py`. The four-view fusion used for the reported evaluation is
the `predict` stage of `../03_prior_evaluation/evaluate.py`.

The path sampler itself lives in `../03_prior_evaluation/path_sampler.py`, which is the module
used by both the production generator and the reported evaluation.

## Procedural masonry layouts

Parametric walls are built from stretcher and header bricks in four layout types:

| Layout | Share | Composition |
|---|---:|---|
| Regular | 0.40 | Stretchers throughout the row, after the first brick |
| Semi-regular | 0.40 | Stretchers and headers alternating on a fixed pattern |
| Irregular | 0.10 | Stretcher or header drawn at random within each row |
| Chaotic | 0.10 | Skyline packing; orientation and course height vary freely |

In every type the first brick of a row follows the same parity rule: odd rows begin with a
header and even rows with a stretcher, which is what staggers the vertical joints. Regular and
semi-regular walls draw one joint width for the whole wall, while irregular and chaotic walls
draw a new joint width at each joint. Semi-regular rows shift by a fixed 0.0565 m on odd rows;
regular and irregular rows shift even rows by half a joint. Irregular rows are topped up at the
end so that every row reaches a common target width.

For each brick the generator stores the centre in the wall plane, the brick type and the
rotation angle. The vector layout is rasterised at 1000 pixels per metre by polygon filling,
which keeps rotated bricks correctly represented, and the raster is flipped vertically to match
the image coordinate system. This produces a binary brick mask, a binary mortar mask and a
connected-component map with a separate identifier for each brick.

## Crack-probability model

A lightweight U-Net learns the empirical spatial relationship between the masonry layout and the
occurrence of cracks in the real training masks.

The colour-coded masks hold green intact bricks, red broken bricks, yellow cracks and a black
mortar and background region. The yellow pixels are stored as the binary crack target. A
separate copy is reconstructed as a crack-free layout by joining broken-brick fragments near the
annotated crack with a repair band adapted to brick area and crack width. This removes the crack
shape from the input while the original yellow annotation remains the target. After
reconstruction, the red and green pixels form the binary brick-occupancy channel and the
reconstructed black pixels form the complementary mortar and background channel.

| Setting | Value |
|---|---|
| Encoder | three levels with 32, 64 and 128 channels |
| Bottleneck | 256 channels |
| Decoder | symmetric |
| Dropout | 0.15 in the convolutional blocks |
| Input | 512 x 512 masks, two channels |
| Batch size | 8 |
| Optimiser | AdamW, initial learning rate 0.001, weight decay 1e-4 |
| Loss | 0.7 BCE + 0.3 Dice |
| Training augmentation | none |
| Epochs | 100 |
| Checkpoint | the epoch with the lowest internal-validation loss |

All masks used at this stage come from the 1000-image training partition of
[MCrack1300](https://universe.roboflow.com/acsalab/masonry-zqhaw). Filename
families and exact duplicate records are kept together in the fixed 900-image optimisation
subset and the 100-image internal-validation subset, with random seed 42.

### Producing the training data

`prepare.py` reads a finished reconstruction from `<workspace_root>/masks/train_reconstructed/`.
Three commands in `preprocessing/mask_preprocessing.py` produce it from the annotated dataset,
and none of them has to be repeated once it has run:

Run these commands from the repository root:

    python 01_crack_path_generation/preprocessing/mask_preprocessing.py draw-test-masks --partition train --expect-images 1000
    python 01_crack_path_generation/preprocessing/mask_preprocessing.py preview-masks \
        --source <workspace>/masks/train_annotations/masks_image \
        --output <workspace>/masks/train_reconstructed --count 1000
    python 01_crack_path_generation/preprocessing/mask_preprocessing.py preview-masks \
        --source <workspace>/masks/train_annotations/masks_image \
        --output <workspace>/analysis/mask_preview_train --count 50
    python 01_crack_path_generation/preprocessing/mask_preprocessing.py verify-masks --partition train \
        --output <workspace>/masks/train_reconstructed \
        --accepted <workspace>/analysis/mask_preview_train

The first rasterises the annotations into colour-coded masks. The second reconstructs all of
them. The third repeats the reconstruction on the 50-image subset that `verify-masks` requires,
which is the determinism check: the subset must come out byte-identical to the corresponding
images of the full run. The fourth validates the result and writes the `validation_report.json`
that `prepare.py` refuses to start without. Then:

    python unet/prepare.py
    python unet/train.py --run <workspace_root>/runs/crack_unet

Training writes `<workspace_root>/runs/crack_unet/best.pt`, which every later stage reads
through `unet_checkpoint` in `config.yaml`.

Which epoch wins is not stable between runs. The validation-loss curve is nearly flat near
convergence, while cuDNN's non-deterministic convolution backward and mixed-precision training
perturb it at around 1e-7, so the argmin can move by several epochs between otherwise identical
runs, and will almost certainly differ on another GPU or software stack. The selected weights
perform equivalently; only the epoch number moves. Treat the epoch as a record of one run, not
as a target to reproduce.

## Generation geometry

For synthetic generation a 0.922 x 0.922 m region is cropped from the procedural wall raster,
centred at the fixed wall coordinate (0.3, -0.02) m, which contains approximately ten brick
courses. At 1000 pixels per metre this is about 922 x 922 pixels and is resized to 512 x 512
using nearest-neighbour interpolation. The crop origin, padding and resize scale are stored so
that the sampled path can be mapped back to wall coordinates in metres.

## Principal sampler parameters

| Parameter | Value | Purpose |
|---|---|---|
| Wall rasterisation | 1000 pixels/m | Pixel-to-world mapping |
| Probability-map resolution | 512 x 512 | Inference and path sampling |
| Probability-map crop | 0.922 x 0.922 m | Fixed wall extent for crack-coordinate generation |
| Goal bias, gamma | 0.7 | Attraction towards the sampled endpoint |
| Inertia coefficient, lambda | 0.8 | Penalise abrupt changes from the previous step |
| Revisit factor | 0.01 | Loop suppression and numerical stability |
| In-brick run length | 0.7 to 1.4 brick heights | Reject implausible traversals |
| Branch probability | 0.20 or 0.30 | Compensate for camera cropping and vary branch frequency |

The goal bias and inertia coefficient were selected with the eighteen-setting grid in
`../03_prior_evaluation/`. The fixed traversal and branching controls are recorded with every
generated run in `generation_summary.json`.

## Producing the coordinate files

`generate_crack_coordinates.py` runs the whole stage: it draws a wall, crops the generation
region, predicts and fuses the four flips, samples the paths and writes the file pairs that
`../02_blender_generation/wall_generator.py` reads. It needs a trained checkpoint, which it
takes from `unet_checkpoint` in `config.yaml`.

    python generate_crack_coordinates.py --walls 20 --paths-per-wall 3

One wall carries several cracks, so the file pairs of a wall share its prefix and its geometry
and differ only in the crack and in the damage flags. The output goes to
`<workspace_root>/analysis/crack_coordinates`, which is also where `refine_crack_path.py` looks
by default, and `generation_summary.json` beside it records the seed, the sampler settings and
the checkpoint for every scene.

| Option | Meaning |
|---|---|
| `--walls`, `--paths-per-wall` | How many walls, and how many cracks on each |
| `--view-range` | `close`, `middle`, `far` or `step`; sets the wall size and the crop extent |
| `--branch-probability` | How often a scene carries a branch |
| `--min-interior-depth` | How far inside a brick a crack pixel must lie to count as damage |
| `--min-crossing-pixels` | How many such pixels a brick needs before it is flagged broken |
| `--seed` | Base seed; every scene's seed is derived from it and recorded |

The two damage thresholds decide the brick flags. A path travelling along a joint clips the
outermost row of the bricks beside it without entering them, and a path that does enter a brick
may only clip its corner; neither is damage. The defaults, 2 pixels of depth and 10 such pixels,
keep the complete traversals and drop both kinds of contact. The summary reports `touched`,
`entered` and `broken` per scene, so the effect of a threshold can be seen without opening the
files.

## Output

The sampler outputs a binary crack mask and the ordered coordinates of the main path and any
branches on the 512 x 512 grid. These are mapped back to wall coordinates in metres by
reversing the resize, adding the crop offset and dividing by the raster resolution.

The mapped path is then placed on the connected-component brick map. A brick is marked as
broken when the crack runs inside it, by the two thresholds above; a brick the path only
touches while travelling along a joint beside it is not damaged. The wall file records which
side of the crack every remaining brick lies on, which allows the two sides to be moved
differently during scene construction. Two
coordinate files are written for the Blender stage: the wall file with the bricks and their
damage labels, and the crack file with the main path and its branches.
