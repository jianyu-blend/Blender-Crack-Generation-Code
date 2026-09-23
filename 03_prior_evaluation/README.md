# Evaluation of the learned spatial prior

This directory compares the stochastic path sampler under a learned crack-probability map and
under a uniform map, and produces the eighteen-setting grid reported in the manuscript.

## Contents

| File | Role |
|---|---|
| `workspace.py` | Locations from `config.yaml` and small IO helpers |
| `crossing_rules.py` | The 0.7 to 1.4 in-brick traversal rule and the 8-neighbourhood |
| `task_data.py` | Layout, reference centreline and brick geometry for one image |
| `metrics.py` | Fixed-tolerance and width-adaptive centreline scoring, and path diversity |
| `path_sampler.py` | The sampler: four-view fusion, prior tempering, material field, remaining-cost field and the look-ahead walk |
| `evaluate.py` | The four pipeline stages |
| `evaluation_loop.py` | The paired test loop |
| `report.py` | Summary tables and `REPORT.md` |
| `sweep_gamma_lambda.py` | Sweeps the goal bias and the inertia coefficient over the grid |
| `build_prior_ablation.py` | Turns the sweep outputs into the eighteen-setting table |

## Protocol

Both methods receive the same reconstructed layout, the same supplied start and end points, the
same five seeds and the same sampling budget. The uniform method assigns a probability of one
throughout the raster and keeps the geometric costs, the endpoint attraction and the
brick-crossing rules, so it is a geometry-guided baseline rather than a second model.

Reference main paths come from two geodesic sweeps on the largest connected skeleton component.
Their endpoints are supplied to both samplers at their original coordinates. The sampler never
reads the reference centreline; it is used only for the endpoints and for scoring.

| Setting | Value |
|---|---|
| Goal bias, gamma | 0.3 to 1.3 in steps of 0.2 |
| Inertia coefficient, lambda | 0.7 to 0.9 in steps of 0.1 |
| Seeds | 41001 to 41005 |
| Attempts per seed | up to 8, of 16384 steps each |
| Scoring tolerance | 4 pixels at 512 x 512 |
| Bootstrap | 10000 resamples over whole filename families |

## Running

Set `config.yaml` first, then run the stages in order:

    python evaluate.py prepare
    python evaluate.py predict calibration
    python evaluate.py predict test
    python evaluate.py calibrate
    python evaluate.py evaluate
    python report.py

`prepare` writes `protocol.json`, including the image lists and the frozen rules. `predict`
writes the four aligned flip probability maps. `calibrate` screens the three fusion schemes on
the calibration images and freezes the winner in `locked_parameters.json` before any test image
is scored. `evaluate` scores the test partition, and `report.py` writes the tables and
`REPORT.md`. Everything is written under `<workspace_root>/evaluation/`.

The eighteen-setting grid reuses the probability maps and varies only the sampler:

    python sweep_gamma_lambda.py \
        --grid "0.3,0.7;0.3,0.8;0.3,0.9;0.5,0.7;0.5,0.8;0.5,0.9;0.7,0.7;0.7,0.8;0.7,0.9;\
0.9,0.7;0.9,0.8;0.9,0.9;1.1,0.7;1.1,0.8;1.1,0.9;1.3,0.7;1.3,0.8;1.3,0.9"

    python build_prior_ablation.py <per_image.csv> <summary.csv> <output.csv>

`build_prior_ablation.py` reads the sweep outputs only. It generates no new path and no new
model prediction. Its `--expect-settings`, `--expect-images` and `--expect-families` options
assert the manuscript's 18, 149 and 95; leave them out for another dataset.

## The sampler

`path_sampler.generate` composes five stages in one call:

| Stage | Role |
|---|---|
| `fuse_views` | Blend the four aligned flip predictions, with seed-specific Dirichlet weights |
| `_temper_prior` | Raise the image-relative map to a seed-specific exponent, so the strength of the learned guidance varies between samples |
| `_prepare_material` | Add the material-preference random field |
| `_prepare_field` | Build the reverse-Dijkstra remaining-cost field |
| `_walk_lookahead` | Draw each pixel from the locally normalised weights, with the look-ahead regret factor |

Every candidate pixel must lower the remaining cost and pass the in-brick crossing rule. Strict
cost descent makes a retained path acyclic, so the 0.01 revisit factor is inactive on it.

## Relationship to the development history

The reported method was reached over several development rounds. This release contains the
final method as one self-contained pipeline: the superseded round drivers, and the protocol
chain in which each round read the previous round's frozen outputs, are not included. The
sampler stages are the composition layers of the final method and all execute on every call.

One consequence is that `calibrate` screens the three fusion schemes on their own calibration
scores. The additional retention constraints of the original round, which compared against an
earlier round's results, are not reproduced here, because that round's outputs are not part of
the release.
