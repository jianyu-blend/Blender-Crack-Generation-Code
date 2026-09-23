"""Paired comparison of a learned and a uniform spatial prior over crack paths.

Four stages, run in order:

    python evaluate.py prepare              write protocol.json
    python evaluate.py predict calibration  four-view probability maps
    python evaluate.py predict test
    python evaluate.py calibrate            freeze the fusion scheme
    python evaluate.py evaluate             score the test partition

Both methods receive the same layouts, the same supplied endpoints, the same
five seeds and the same sampling budget. The uniform method replaces the
predicted probability with a constant map and keeps every geometric rule, so it
is a geometry-guided baseline rather than a second model.

The sampler never reads the reference centreline. The reference is used only to
supply the two endpoints and to score the result afterwards.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from workspace import (TRAIN, TEST, RUN, OUT, SEEDS,
                       read_json, write_json, read_csv, write_csv, digest, load_mask, paths)
from task_data import task_data
from metrics import geometry, fixed_metrics, morphology, aggregate, spread
from path_sampler import generate

# The three candidate schemes for fusing the four aligned flip predictions.
# Everything else is frozen at the reported values.
CONFIGS = [dict(id=f'flip_{fusion}', gamma=.9, inertia=.9, heading_steps=1, brick_penalty=1.,
                route_sigma=.35, field_sigma=.15, prior_power=1., prior_sigma=1.,
                lookahead=1., fusion=fusion)
           for fusion in ['half_mean', 'mean', 'random']]

PRIMARY_ATTEMPTS = 8
ATTEMPT_BUDGET = 16384

# The manuscript grid: six goal-bias and three inertia values.
GAMMA_GRID = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
LAMBDA_GRID = [0.7, 0.8, 0.9]
PRODUCTION_GAMMA, PRODUCTION_LAMBDA = 0.7, 0.8


# ----------------------------------------------------------------- prepare
def prepare():
    """Record the protocol: the image lists, the grid and the fixed rules."""
    if (OUT / 'protocol.json').exists():
        raise RuntimeError(f'Protocol already exists; reuse the recorded run: {OUT}')
    sys.path.insert(0, str(paths.unet_code_dir()))
    from prepare import family

    train = read_csv(RUN / 'data_manifest.csv')
    test = read_csv(TEST / 'manifest.csv')
    split = read_json(RUN / 'split.json')

    trainfamilies = {r['source_family'] for r in train}
    testfamilies = {family(r['filename']) for r in test}
    modeltrainfamilies = {r['source_family'] for r in train if r['split'] == 'train'}

    # Calibration images are internal-validation images whose filename family
    # does not also appear in the test partition.
    allowed = [r for r in train
               if r['filename'] in split['validation'] and r['source_family'] not in testfamilies]
    # Screen on a subset drawn across the brick-area scale, never on test scores.
    nonempty = [r for r in allowed if int(r['skeleton_pixels']) >= 20]
    nonempty.sort(key=lambda r: (float(r['reference_area_px2']), r['filename']))
    chosen = [nonempty[i]['filename']
              for i in np.linspace(0, len(nonempty) - 1, min(30, len(nonempty)), dtype=int)]

    sourcehashes = {r['source_sha256'] for r in train}
    pairhashes = {r['brick_sha256'] + r['target_sha256'] for r in train}
    overlaps = []
    for r in test:
        name = r['filename']
        fam = family(name)
        overlaps.append(dict(
            filename=name, family=fam,
            possible_family_overlap_with_training=fam in trainfamilies,
            possible_family_overlap_with_model_training=fam in modeltrainfamilies,
            exact_source_mask_overlap=r['source_sha256'] in sourcehashes,
            exact_input_target_pair_overlap=(digest(TEST / 'brick_binary' / name)
                                             + digest(TEST / 'crack_binary' / name)) in pairhashes))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'data_overlap_audit.csv', overlaps)
    protocol = dict(
        checkpoint=str(paths.checkpoint), checkpoint_sha256=digest(paths.checkpoint),
        calibration_names=[r['filename'] for r in allowed],
        screening_names=chosen,
        test_names=[r['filename'] for r in test],
        test_no_family_overlap_names=[r['filename'] for r in overlaps
                                      if not r['possible_family_overlap_with_training']],
        seeds=SEEDS, configs=CONFIGS,
        primary_attempts=PRIMARY_ATTEMPTS, attempt_budget=ATTEMPT_BUDGET,
        gamma_grid=GAMMA_GRID, lambda_grid=LAMBDA_GRID,
        production_gamma=PRODUCTION_GAMMA, production_lambda=PRODUCTION_LAMBDA,
        uniform_prior='Constant map, so only the goal, direction, revisit and traversal rules '
                      'affect transitions.',
        inference='Four views of the frozen network in eval mode and float32: identity, '
                  'horizontal flip, vertical flip and both. Predictions are mapped back before '
                  'stacking. No target pixel is read at inference.',
        fusion='half_mean weights [0.625,0.125,0.125,0.125]; mean weights four 0.25; random '
               'draws Dirichlet([0.5]*4) seeded with [seed,15015]. Four constant baseline maps '
               'stay constant under the same fusion.',
        selection='Screen the three fusion schemes on the screening images by mean best-of-five '
                  'centreline F1 with the learned prior, then confirm the winner on all valid '
                  'calibration images and freeze it before any test image is scored.',
        fixed=dict(revisit_factor=0.01, in_brick_min=0.7, in_brick_max=1.4,
                   probability_floor=1e-6, transverse_alignment_min=math.sqrt(0.5),
                   attempt_budget=ATTEMPT_BUDGET, path_match_tolerance_px=4,
                   secondary_tolerance_px=2, resolution=[512, 512]),
        scope='Main-path spatial-prior comparison. Branch initiation is not tuned or validated '
              'here. Metrics are centreline path metrics, not the downstream detector mask mAP.',
        endpoint_conditioning='Two-sweep geodesic path on the dominant skeleton component. Its '
                              'endpoints are supplied to both methods unchanged and may lie '
                              'inside a brick.',
        brick_geometry='Connected binary brick regions with minimum-area rectangles estimate '
                       'brick identity and short-axis height; touching bricks may merge.',
        test_scores_used_for_selection=False)
    write_json(OUT / 'protocol.json', protocol)
    print(f'PREPARED {len(protocol["calibration_names"])} calibration and '
          f'{len(protocol["test_names"])} test images -> {OUT}', flush=True)


# ----------------------------------------------------------------- predict
def predict(partition):
    """Write the four aligned flip probability maps for one partition."""
    import torch
    sys.path.insert(0, str(paths.unet_code_dir()))
    from models import UNet

    p = read_json(OUT / 'protocol.json')
    checkpoint_path = paths.checkpoint
    if digest(checkpoint_path) != p['checkpoint_sha256']:
        raise RuntimeError('The checkpoint changed since prepare; rerun prepare.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_ch=2, base=32, p_drop=.15).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    torch.set_num_threads(4)

    names = p['calibration_names'] if partition == 'calibration' else p['test_names']
    data = TRAIN if partition == 'calibration' else TEST
    folder = OUT / 'probabilities' / partition
    folder.mkdir(parents=True, exist_ok=True)
    dims = [(), (-1,), (-2,), (-2, -1)]
    for i, name in enumerate(names):
        file = folder / (Path(name).stem + '.npy')
        if file.exists():
            continue
        brick = load_mask(data, 'brick_binary', name).astype(np.float32)
        x = torch.from_numpy(np.stack([brick, 1 - brick])).to(device)
        batch = torch.stack([x if not d else torch.flip(x, d) for d in dims])
        with torch.inference_mode():
            predictions = model(batch).sigmoid()[:, 0]
        aligned = np.stack([(predictions[j] if not d else torch.flip(predictions[j], d)).cpu().numpy()
                            for j, d in enumerate(dims)])
        if aligned.shape[0] != 4 or not np.isfinite(aligned).all():
            raise RuntimeError(f'Bad prediction for {name}')
        np.save(file, aligned)
        if (i + 1) % 10 == 0:
            print(f'PREDICT {partition} {i + 1}/{len(names)}', flush=True)
    write_json(OUT / (partition + '_inference.json'),
               dict(images=len(names), checkpoint_sha256=p['checkpoint_sha256'],
                    checkpoint_epoch=checkpoint.get('epoch'), device=device,
                    torch=str(torch.__version__),
                    view_order=['identity', 'horizontal', 'vertical', 'both'],
                    targets_read=False))


# --------------------------------------------------------------- calibrate
def calibration_run(names, configs, stage):
    """Sample every named calibration image under each config and both methods."""
    p = read_json(OUT / 'protocol.json')
    rows, excluded, started = [], [], time.perf_counter()
    for i, name in enumerate(names):
        task = task_data(TRAIN, name)
        if task is None:
            excluded.append(name)
            continue
        geo = geometry(task)
        probability = np.load(OUT / 'probabilities/calibration' / (Path(name).stem + '.npy'))
        for config in configs:
            for method in ['unet', 'no_unet']:
                prob = probability if method == 'unet' else np.ones_like(probability)
                local, sampled = [], []
                for seed in SEEDS:
                    cache = (OUT / 'calibration_paths' / config['id'] / method / str(seed)
                             / (Path(name).stem + '.json'))
                    if cache.exists():
                        saved = read_json(cache)
                        path = np.array(saved['points_xy'], np.int32)[:, ::-1].copy()
                        details, m = saved['generation'], saved['metrics']
                    else:
                        path, details = generate(prob, geo, seed, config,
                                                 p['primary_attempts'], p['attempt_budget'])
                        m = fixed_metrics(path, task, details['success'])
                        m.update(morphology(path, task))
                        write_json(cache, dict(points_xy=path[:, ::-1].tolist(),
                                               generation=details, metrics=m))
                    sampled.append(path)
                    local.append(dict(config=config['id'], filename=name, method=method, seed=seed,
                                      **{k: v for k, v in details.items() if k != 'success'}, **m))
                unique = len({a.tobytes() for a in sampled})
                rows.extend(dict(**r, unique_paths=unique) for r in local)
        if (i + 1) % 5 == 0:
            print(f'{stage}: {i + 1}/{len(names)} images, '
                  f'{time.perf_counter() - started:.1f}s', flush=True)
    write_csv(OUT / (stage + '_runs.csv'), rows)
    per, summary = aggregate(rows)
    write_csv(OUT / (stage + '_per_image.csv'), per)
    write_csv(OUT / (stage + '_summary.csv'), summary)
    write_json(OUT / (stage + '_excluded.json'), excluded)
    return summary


def diversity_summary(config, names):
    """Pairwise separation between the five sampled paths of each image."""
    rows = []
    for name in names:
        task = task_data(TRAIN, name)
        if task is None:
            continue
        for method in ['unet', 'no_unet']:
            sampled = []
            for seed in SEEDS:
                item = read_json(OUT / 'calibration_paths' / config['id'] / method / str(seed)
                                 / (name.rsplit('.', 1)[0] + '.json'))
                sampled.append(np.array(item['points_xy'], np.int32)[:, ::-1].copy())
            rows.append(dict(config=config['id'], filename=name, method=method,
                             **spread(sampled, task['brick'].shape, task['height'])))
    return rows


def calibrate():
    """Screen the three fusion schemes, then freeze the best one."""
    if (OUT / 'locked_parameters.json').exists():
        raise RuntimeError('Parameters are already frozen.')
    p = read_json(OUT / 'protocol.json')

    summary = calibration_run(p['screening_names'], CONFIGS, 'screening')
    divs = sum([diversity_summary(c, p['screening_names']) for c in CONFIGS], [])
    ranks = [dict(config=cid,
                  unet_best_f1=next(r['mean_max_f1'] for r in summary
                                    if r['config'] == cid and r['method'] == 'unet'),
                  unet_mean_f1=next(r['mean_f1'] for r in summary
                                    if r['config'] == cid and r['method'] == 'unet'),
                  unet_completion=next(r['completion'] for r in summary
                                       if r['config'] == cid and r['method'] == 'unet'))
             for cid in dict.fromkeys(r['config'] for r in summary)]
    chosen = max(ranks, key=lambda r: (r['unet_best_f1'], r['unet_mean_f1']))
    write_csv(OUT / 'screening_diversity.csv', divs)
    write_json(OUT / 'screening_selection.json', dict(selected=chosen, ranks=ranks))
    print('SCREENED', chosen, flush=True)

    config = next(c for c in CONFIGS if c['id'] == chosen['config'])
    summary = calibration_run(p['calibration_names'], [config], 'calibration')
    divs = diversity_summary(config, p['calibration_names'])
    write_csv(OUT / 'calibration_diversity.csv', divs)
    write_json(OUT / 'locked_parameters.json',
               dict(config=config, calibration=summary, screening_ranks=ranks,
                    protocol_sha256=digest(OUT / 'protocol.json'),
                    generator_sha256=digest(Path(__file__).with_name('path_sampler.py')),
                    test_scores_used_for_selection=False))
    print('LOCKED', summary, flush=True)


# ---------------------------------------------------------------- evaluate
def evaluate():
    from evaluation_loop import evaluate as run
    run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['prepare', 'predict', 'calibrate', 'evaluate'])
    parser.add_argument('partition', nargs='?', choices=['calibration', 'test'],
                        help='required for the predict stage')
    args = parser.parse_args()
    if args.stage == 'predict':
        if not args.partition:
            parser.error('predict needs a partition: calibration or test')
        predict(args.partition)
    else:
        globals()[args.stage]()
