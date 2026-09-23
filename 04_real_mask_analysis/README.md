# Analysis of the real masonry masks

`real_mask_analysis.py` holds every analysis that produces the evidence behind the generator
parameters. Each analysis is a subcommand and keeps the arguments it had as a separate script.

    python real_mask_analysis.py                  # list the commands
    python real_mask_analysis.py <command> --help # arguments for one command

## Commands

| Command | Role |
|---|---|
| `crack-features` | Crack-location and crack-path statistics from the colour-coded masks |
| `high-recall-traversals` | The 104 mortar-to-broken-brick-to-mortar traversals behind the 0.7 to 1.4 in-brick rule |
| `branch-traversal` | Branching rate, 101 of 1000 images, and traversal measurement |
| `compare-paths` | Generated smoothed crack coordinates against real crack paths |
| `calibrate` | Goal-bias and inertia calibration with the learned term held constant |
| `calibrate-with-maps` | The same calibration on fixed learned probability maps |
| `far-layouts` | Clean 512 px procedural wall layouts for far-range calibration |
| `preview-masks` | Deterministic mask-reconstruction previews and ordered crack polylines |
| `verify-masks` | Integrity check on the saved mask products |
| `draw-test-masks` | Rasterise the YOLO annotations of one partition into colour-coded masks |
| `paper-assets` | Turn the frozen evaluation outputs into manuscript statistics and a figure |

`preview-masks`, `verify-masks` and `draw-test-masks` produce the reconstructed masks that
`01_crack_path_generation/unet/prepare.py` reads. Run them in that order; the sequence is in
`01_crack_path_generation/README.md`.

`draw-test-masks` rasterises one partition, named by `--partition`. It defaults to `test`, which
is the partition the path evaluation uses, and writes to `<workspace>/masks/test_annotations`.
Any other partition writes to `<workspace>/masks/<partition>_annotations` unless `--output`
says otherwise. The crack-probability U-Net is trained on the `train` partition, so that
partition has to be rasterised before `prepare.py` can run. `--expect-images` asserts an image
count, which is how the original script asserted its 150.

`verify-masks` validates one reconstruction against the colour masks it came from. `--partition`
selects those colour masks, and `--source`, `--output` and `--accepted` override the three
locations individually. `--expect-accepted` sets how many visually reviewed samples must be
present and byte-identical, which the original script fixed at 50.

Every command writes under `workspace_root`, and every default input path is derived from
`config.yaml` rather than from the current working directory.

## Dependencies

The common path comes from `numpy`, `opencv-python` and `Pillow`. Two commands need more:
`calibrate-with-maps` imports `torch` for the crack-probability model, and `paper-assets` imports
`matplotlib`. Both imports are made inside the functions that use them, so the other nine
commands run without those packages installed.

## About the merge

This file was assembled from twelve separate scripts. Function and class bodies were copied
without modification, and the merge was checked by comparing all 106 top-level definitions
against their sources.

Ten names were defined in more than one script with different implementations. They were kept
apart rather than merged, because the versions are not interchangeable:

| Name | Kept as |
|---|---|
| `main` | `main_<command>`, eleven versions |
| `parse_args` | `parse_args_<command>`, seven versions |
| `write_csv` | `write_csv_branch`, `write_csv_traversals`, `write_csv_calib`; the calibration version takes the union of the keys of every row, the others take the keys of the first row |
| `PALETTE` | `PALETTE_DRAW` as `uint8`, `PALETTE_FEATURES` as `int32` |
| `NEIGHBOURS` | `NEIGHBOURS_CALIB` as an array, `NEIGHBOURS_FEATURES` as a tuple |
| `resample_path`, `turning_angles` | `*_calib` and `*_features` |
| `ROOT`, `OUT`, `SOURCE` | `*_DRAW` and `*_ASSETS`, since each script pointed at its own directories |

The three scripts that imported one another now call the corresponding functions directly, so
the cross-import chain is gone.
