# Mask preprocessing

This directory contains the three preprocessing commands required to turn an annotated
YOLO-segmentation masonry dataset into the crack-free wall masks and crack targets used to
train the crack-probability U-Net.

Run the commands from the repository root:

```powershell
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py draw-test-masks --partition train --expect-images 1000
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py preview-masks --source "$BCG_WORKSPACE\masks\train_annotations\masks_image" --output "$BCG_WORKSPACE\masks\train_reconstructed" --count 1000
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py preview-masks --source "$BCG_WORKSPACE\masks\train_annotations\masks_image" --output "$BCG_WORKSPACE\analysis\mask_preview_train" --count 50
python .\01_crack_path_generation\preprocessing\mask_preprocessing.py verify-masks --partition train --output "$BCG_WORKSPACE\masks\train_reconstructed" --accepted "$BCG_WORKSPACE\analysis\mask_preview_train"
```

The first command rasterises the supplied YOLO polygons into colour-coded masks. The second
reconstructs the full training set. The third repeats a deterministic 50-image subset, and the
fourth verifies that this subset is byte-identical to the corresponding full-run outputs and
writes `validation_report.json`. `unet/prepare.py` refuses to start unless that report passes.

The module contains only these mask-preprocessing operations.
