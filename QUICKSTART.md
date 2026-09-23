# Quick start: one scene, six RGB images and six masks

These steps are written for Windows PowerShell. They take the repository from a fresh clone to
one Blender scene. A scene produces six aligned `*_P.png` RGB images and six `*_M.png` colour
masks.

MCrack1300, trained weights and HDRI/EXR files are not stored in this repository.

## 1. Clone the repository

```powershell
git clone https://github.com/jianyu-blend/Blender-Crack-Generation-Code.git
cd Blender-Crack-Generation-Code
```

## 2. Create the Python environment

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 3. Download and configure the real dataset

Download a YOLO-segmentation export of
[MCrack1300 from Roboflow](https://universe.roboflow.com/acsalab/masonry-zqhaw). Do not copy the
dataset into this Git repository. Its extracted directory must contain `train`, `valid`, `test`
and `data.yaml`.

Create the local configuration:

```powershell
Copy-Item .\config.example.yaml .\config.yaml
notepad .\config.yaml
```

Set these two entries and save the file:

```yaml
dataset_root: ../MCrack1300
workspace_root: ../bcg_workspace
```

Check the resolved paths:

```powershell
python .\bcg_config.py
```

## 4. Build the U-Net training masks

Replace the value below with the same `workspace_root` used in `config.yaml`:

```powershell
$BCG_WORKSPACE = (Resolve-Path ..\bcg_workspace).Path
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py draw-test-masks --partition train --expect-images 1000
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py preview-masks --source "$BCG_WORKSPACE\masks\train_annotations\masks_image" --output "$BCG_WORKSPACE\masks\train_reconstructed" --count 1000
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py preview-masks --source "$BCG_WORKSPACE\masks\train_annotations\masks_image" --output "$BCG_WORKSPACE\analysis\mask_preview_train" --count 50
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py verify-masks --partition train --output "$BCG_WORKSPACE\masks\train_reconstructed" --accepted "$BCG_WORKSPACE\analysis\mask_preview_train"
```

## 5. Train the crack-probability U-Net

```powershell
python .\01_crack_path_generation\unet\prepare.py
python .\01_crack_path_generation\unet\train.py --run "$BCG_WORKSPACE\runs\crack_unet"
```

The released training configuration uses 100 epochs and saves the lowest-validation-loss model
as `$BCG_WORKSPACE\runs\crack_unet\best.pt`. The checkpoint is not uploaded to this repository.

## 6. Generate and refine one coordinate pair

```powershell
$COORDINATES = "$BCG_WORKSPACE\analysis\crack_coordinates"
python .\01_crack_path_generation\generate_crack_coordinates.py --walls 1 --paths-per-wall 1 --view-range middle --output "$COORDINATES"
python .\01_crack_path_generation\refine_crack_path.py "$COORDINATES"
```

The refine command is important: it removes small duplicate jumps and smooths the exported crack
polyline before Blender constructs the crack geometry.

To generate one wall with three alternative cracks instead, use `--paths-per-wall 3`. That writes
three wall/crack file pairs sharing the same masonry layout.

## 7. Add HDRIs

Create an HDRI directory:

```powershell
New-Item -ItemType Directory -Force .\HDRI
```

Download one or more equirectangular `.exr` HDRIs from
[BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree%3Ahdr-outdoor) or
[Poly Haven](https://polyhaven.com/hdris), and place the files directly in `HDRI`.

## 8. Create the Blender render configuration

Blender 4.5 LTS is recommended. If `blender` is not on `PATH`, replace it in the commands below
with the full path to `blender.exe`.

```powershell
blender --background .\02_blender_generation\Dataset_Generator.blend --python .\02_blender_generation\wall_generator.py -- --write-default-config .\render_config.json
notepad .\render_config.json
```

Set at least these values in `render_config.json`:

```json
{
  "asset_library": "02_blender_generation/Asset_Library.blend",
  "hdri_directory": "HDRI",
  "coordinate_directory": "../bcg_workspace/analysis/crack_coordinates",
  "output_directory": "outputs/renders",
  "presets_file": "02_blender_generation/presets.json",
  "view_range": "middle",
  "crack_width_label": "5to10mm",
  "crack_side_motion": "translation",
  "max_scenes": 1,
  "resolution": 640,
  "render_samples": 64,
  "write_depth": false,
  "brick_displacement_scale_range": [0.05, 0.10],
  "label_render_samples": 4,
  "mask_shrinkwrap_offset": 0.03
}
```

Keep the other fields written by `--write-default-config`; only change their values as needed.
`Dataset_Generator.blend` supplies the compositor and the saved Cycles/colour-management setup.

## 9. Render

```powershell
blender --background .\02_blender_generation\Dataset_Generator.blend --python .\02_blender_generation\wall_generator.py -- --config .\render_config.json
```

The output directory will contain:

- six `*_P.png` RGB images;
- six aligned `*_M.png` colour masks;
- no raw black `*.png` render and, with `write_depth` set to `false`, no depth files.

Mask colours are green intact brick, red broken brick, blue mortar and yellow crack.

## 10. Optional: convert masks to YOLO polygons

```powershell
python .\02_blender_generation\masks_to_yolo_polygons.py --masks .\outputs\renders --output .\outputs\labels
```

For the learned-prior evaluation and downstream active-learning experiments, continue with the
README files in `03_prior_evaluation` and `05_downstream_training`.
