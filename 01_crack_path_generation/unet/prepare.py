"""Prepare an auditable training run from the reconstructed mask products."""
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from bcg_config import paths

DATA = paths.train_masks
RUN = paths.unet_run

# How many images are held out for internal validation. The rest optimise the
# network. The manuscript uses 100 out of the 1000 MCrack1300 training masks;
# another dataset only has to supply enough whole filename families to reach
# this number exactly.
VALIDATION_IMAGES = int(paths.get('validation_images', 100))
SPLIT_SEED = int(paths.get('split_seed', 42))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(name):
    stem = re.split(r'_(?:jpg|jpeg|png)(?:[._]|$)', name, flags=re.I)[0]
    stem = re.sub(r'\.rf\..*$', '', stem)
    if re.fullmatch(r'\d+(?:-\d+)?', stem):
        return 'numeric_' + str(int(stem.split('-')[0]))
    if re.fullmatch(r'a_\d+_\d+', stem):
        return stem.rsplit('_', 1)[0]
    if re.fullmatch(r'l\d+_\d+', stem):
        return stem.rsplit('_', 1)[0]
    return stem


def main():
    if RUN.exists():
        raise RuntimeError(f'Run already exists; do not overwrite: {RUN}')
    validation = json.loads((DATA / 'validation_report.json').read_text())
    if validation['status'] != 'passed':
        raise RuntimeError(f"Mask preprocessing did not pass: {validation['status']}")
    rows = list(csv.DictReader((DATA / 'manifest.csv').open(encoding='utf-8-sig')))
    if validation['completed_images'] != len(rows):
        raise RuntimeError(
            f"validation_report.json records {validation['completed_images']} images "
            f"but manifest.csv holds {len(rows)}."
        )
    if len(rows) <= VALIDATION_IMAGES:
        raise RuntimeError(
            f'{len(rows)} reconstructed masks is not more than the '
            f'{VALIDATION_IMAGES} requested for validation.'
        )
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        row['source_family'] = family(row['filename'])
        for kind, value in [('family', row['source_family']), ('source', row['source_sha256'])]:
            if (kind, value) in seen:
                parent[find(i)] = find(seen[(kind, value)])
            else:
                seen[(kind, value)] = i
        row['brick_sha256'] = digest(DATA / 'brick_binary' / row['filename'])
        row['target_sha256'] = digest(DATA / 'crack_binary' / row['filename'])
        pair = ('pair', row['brick_sha256'] + row['target_sha256'])
        if pair in seen:
            parent[find(i)] = find(seen[pair])
        else:
            seen[pair] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    grouped = sorted(groups.values(), key=lambda g: rows[g[0]]['filename'])
    random.Random(SPLIT_SEED).shuffle(grouped)
    reachable = {0: []}
    for gi, group in enumerate(grouped):
        for size, choice in list(reachable.items()):
            new = size + len(group)
            if new <= VALIDATION_IMAGES and new not in reachable:
                reachable[new] = choice + [gi]
        if VALIDATION_IMAGES in reachable:
            break
    if VALIDATION_IMAGES not in reachable:
        raise RuntimeError(
            f'Cannot obtain exactly {VALIDATION_IMAGES} validation images without '
            f'splitting a filename family. Choose a different validation_images value.'
        )
    val_indices = {i for gi in reachable[VALIDATION_IMAGES] for i in grouped[gi]}
    for i, row in enumerate(rows):
        row['split'] = 'validation' if i in val_indices else 'train'
        row['group_id'] = f'g{find(i):04d}'
    train = [r['filename'] for r in rows if r['split'] == 'train']
    val = [r['filename'] for r in rows if r['split'] == 'validation']
    assert len(val) == VALIDATION_IMAGES and len(train) == len(rows) - VALIDATION_IMAGES
    assert not {r['group_id'] for r in rows if r['split'] == 'train'} & {r['group_id'] for r in rows if r['split'] == 'validation'}
    RUN.mkdir(parents=True)
    split = dict(seed=SPLIT_SEED, train=train, validation=val,
                 method='Seeded source-family grouping; exact 100-image subset via subset sum. Exact source/pair duplicates unioned.',
                 limitation='Filename families are a conservative grouping heuristic, not verified physical-wall identities.',
                 groups=len(groups))
    (RUN / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    with (RUN / 'data_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    config = dict(run_dir=str(RUN), data_dir=str(DATA), seed=SPLIT_SEED, epochs=100,
                  batch_size=8, image_size=512, learning_rate=0.001, weight_decay=0.0001,
                  base_channels=32, dropout=0.15, in_channels=2, use_brick_type=False,
                  bce_weight=0.7, dice_weight=0.3, amp=True, num_workers=0,
                  augmentation='none', optimiser='AdamW', scheduler='none',
                  checkpoint_selection='Lowest sample-mean internal validation loss',
                  stopping_rule='Fixed 100-epoch training run; convergence assessed from saved trajectories, not assumed.',
                  mask_encoding='adaptive_brick_and_complement',
                  input_channels=['brick_binary/255', '1-brick_binary/255'],
                  target='crack_binary/255, including the recorded dark-yellow palette exception',
                  train_images=len(train), validation_images=len(val), total_images=len(rows),
                  source_description=f'{len(rows)} reconstructed masks from the configured dataset. '
                                     f'The manuscript used the 1000-image MCrack1300 training partition.',
                  evaluation_scope='New internal validation on held-out reconstructed masks. Filename grouping does not verify wall-level independence; preprocessing was visually tuned on 50 masks before this split.',
                  learned_quantity='Empirical crack-pixel likelihood conditional on reconstructed brick/mortar layout; no structural-mechanics validation.',
                  source_manifest_sha256=digest(DATA / 'manifest.csv'),
                  split_sha256=digest(RUN / 'split.json'),
                  initialisation='random; no archive checkpoint loaded')
    (RUN / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    code = RUN / 'code'
    code.mkdir()
    for filename in ['models.py', 'prepare.py', 'train.py', 'predict.py']:
        shutil.copy2(Path(__file__).parent / filename, code / filename)
    shutil.copy2(DATA / 'run_summary.json', RUN / 'preprocessing_summary.json')
    summary = dict(train=len(train), validation=len(val), source_groups=len(groups), run_dir=str(RUN),
                   train_empty_targets=sum(int(r['crack_pixels']) == 0 for r in rows if r['split'] == 'train'),
                   validation_empty_targets=sum(int(r['crack_pixels']) == 0 for r in rows if r['split'] == 'validation'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
