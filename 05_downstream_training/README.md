# Downstream training

The downstream experiments train standard instance-segmentation models on combinations of
real and BCG synthetic images. They use unmodified public implementations, so this directory
documents the training settings rather than shipping one script per experimental condition.
`example_yolo_active_learning.py` is a complete, runnable example of the acquisition loop that
every selection and data-budget run follows.

The task is three-class instance segmentation of brick, broken brick and crack. Every model is
evaluated on the fixed 150-image [MCrack1300](https://doi.org/10.1016/j.aei.2024.102826)
validation partition, which is excluded from
training and from the synthetic candidate pools.

## Notation

| Symbol | Meaning |
|---|---|
| `R_n` | `n` real images supplied directly to instance-segmentation training |
| `S_k` | `k` BCG synthetic images |
| `A_k` | `k` additional sampling slots drawn from the same real subset, used by the augmentation control |

The real subsets are nested, so every smaller subset is contained in the larger ones.

## Training protocols

Two schedules are used. The single-stage experiments follow the 400-epoch schedule, and every
experiment with repeated acquisition stages follows the 60-epoch schedule.

| Setting | 400-epoch schedule | 60-epoch schedule |
|---|---|---|
| Experiments | Data source, hybrid augmentation | Synthetic dataset, architecture, selection strategy, data budget |
| Model | YOLOv8x-seg | YOLOv8x-seg, Mask R-CNN with ResNeXt-101-FPN, Mask2Former with Swin-S |
| Initial weights | Official pretrained weights | Official pretrained weights, COCO-pretrained for the two non-YOLO models |
| Epochs | 400 | 60 |
| Batch size | 16 | 8 |
| Initial learning rate | 0.01 | 0.005, except the real-only augmentation control at 0.05 |
| Image size | 640 x 640 | 640 x 640 |
| Checkpoint | `best.pt`, highest validation mask mAP within the run | Best validation checkpoint within the budget |
| Hardware | NVIDIA A100 80 GB (BlueBEAR) | NVIDIA 1g.10gb GPU partition (NCC) |

Every run starts independently from the same pretrained weights. Mask R-CNN and Mask2Former
use their own model-specific optimisation settings from their reference implementations.

## Augmentation

All YOLOv8x-seg runs except the real-only augmentation control use the same Ultralytics
augmentation configuration.

| Parameter | Standard runs | Real-only augmentation control |
|---|---:|---:|
| `hsv_h` | 0.015 | 0.030 |
| `hsv_s` | 0.7 | 0.900 |
| `hsv_v` | 0.4 | 0.600 |
| `degrees` | 0 | 10 |
| `translate` | 0.1 | 0.200 |
| `scale` | 0.5 | 0.750 |
| `shear` | 0 | 5 |
| `perspective` | 0 | 5e-4 |
| `fliplr` | 0.5 | 0.5 |
| `flipud` | 0 | 0 |
| `mosaic` | 1.0 | 1.0 |
| `mixup` | 0 | 0.15 |
| `copy_paste` | 0 | 0.10 |
| `close_mosaic` | 10 | 10 |

In the control condition `R_n + A_k` holds `n + k` training entries per epoch. The `k`
additional entries come from balanced repeated sampling of the same `n` real pairs, so each
source image appears either `floor((n+k)/n)` or `ceil((n+k)/n)` times, and a new online
transformation is sampled whenever an entry is read.

The control shares the 60-epoch schedule and the batch size of the `R_n + S_k` runs it is
compared against, but it uses an initial learning rate of 0.05 rather than 0.005. The two
conditions are therefore matched on the number of training entries, not on the optimiser
setting.

## Synthetic sample selection

Selection scores complete images, so the confidence scores include every retained instance
class. For image `x_i` with `M_i` predicted instances and ordered confidences
`c_i1 ... c_iMi`, three image-level uncertainty scores are evaluated.

| Score | Definition | Value when `M_i = 0` |
|---|---|---|
| `u_max` | `1 - max_j c_ij` | 1 |
| `u_mean` | `1 - sum_j c_ij / max(1, M_i)` | 1 |
| `u_top3` | `1 - sum_{j<=K} c_i(j) / max(1, K)`, `K = min(3, M_i)` | 1 |

Candidate scoring uses a fixed configuration across all three measures.

| Parameter | Value |
|---|---:|
| Input resolution | 640 x 640 |
| Minimum prediction confidence | 0.001 |
| Non-maximum-suppression IoU | 0.7 |

Images are sorted from the highest to the lowest uncertainty score and the required number is
taken from the top of the ranked list. The scoring model changes after each acquisition round,
while every new training run starts from the same pretrained weights.

The selection-strategy experiment compares three groups on the 8000-image BCG pool:

| Group | Runs | Composition |
|---|---:|---|
| Unconstrained uncertainty | 6 | three scores, two independent repetitions each |
| Random | 10 | ten independent random orders |
| Generation-label-constrained uncertainty | 6 | three scores, two repetitions each, with an approximately even acquisition quota across the 18 generation-label groups |

A global uncertainty ranking fills any unallocated positions when a generation-label group has
too few remaining images. Within every run, larger budgets extend the earlier selected subset.

## Example script

The example covers the 60-epoch schedule. The 400-epoch data-source and hybrid-augmentation runs
were executed on a different cluster and no script for them is released here; their settings are
in the table above.

`example_yolo_active_learning.py` runs one complete selection curve: it trains a model on the
current real and synthetic subset, scores the unselected pool with the saved best weights,
ranks the images, extends the cumulative subset and retrains. The acquisition points are
200, 400, 1000, 2000, 3000 and 4000 synthetic images.

Paths and training settings come from the `downstream:` block of `config.yaml` in the
repository root:

| Entry | Purpose |
|---|---|
| `experiment_root` | Working directory for the acquisition runs. Required. |
| `synthetic_pool` | The BCG synthetic candidate pool. Required. |
| `valid_images` | The fixed 150-image validation partition. Defaults to `dataset_root/valid/images`. |
| `model_weights` | Pretrained weights, default `yolov8x-seg.pt` |
| `epochs`, `batch`, `learning_rate`, `image_size` | Training settings, default 60, 8, 0.005 and 640 |
| `device`, `workers` | CUDA device index and dataloader workers |

Each entry can also be overridden by the matching `BCG_DOWNSTREAM_<NAME>` environment
variable, which is convenient inside a job script.

The script stops with a message naming the missing entry if `experiment_root` or
`synthetic_pool` is unset, so nothing runs against a wrong directory by accident.

Run it with:

    python -u example_yolo_active_learning.py

The other conditions differ from this example only in the uncertainty score
(`UNCERTAINTY_SCORE_TYPE`), the real-image base sizes, the acquisition points and whether a
generation-label quota is applied.

## Reported metrics

Performance is reported separately for boxes and masks using mean average precision, at an IoU
of 0.50 and averaged over IoU thresholds from 0.50 to 0.95. Crack-specific performance is
reported alongside the three-class mean, because a class-averaged value can hide differences in
crack segmentation. Relative gains are calculated against the paired real-only baseline from
the same experiment group.
