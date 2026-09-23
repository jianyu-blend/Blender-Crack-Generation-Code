# Figures

Illustrative images used by the repository READMEs. These are a handful of examples, not the
dataset; the dataset is released separately.

Expected files, referenced from the top-level `README.md`:

| File | Content |
|---|---|
| `sample_close_rgb.png` | A close-range render |
| `sample_close_mask.png` | Its label render, same camera and scene state |
| `sample_middle_rgb.png` | A middle-range render |
| `sample_middle_mask.png` | Its label render |
| `sample_far_rgb.png` | A far-range render |
| `sample_far_mask.png` | Its label render |

In a label render, green is an intact brick, red a broken brick, yellow a crack, and blue
mortar. The RGB pass and the label pass share the scene state and the camera, so a pair is
aligned pixel for pixel.

Keep these small. A few hundred kilobytes each is enough for a README; they are illustrations,
not data. If a figure is only needed once, prefer a downscaled copy over the full render.
