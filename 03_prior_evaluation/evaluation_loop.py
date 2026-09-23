"""Score the frozen configuration on the test partition, both methods paired.

For every test layout the loop samples five paths with the learned prior and
five with the uniform prior, from the same endpoints, seeds and budget, and
writes the per-seed metrics, the pixel coordinates and the centreline masks.
Paths that never reach the endpoint are kept and scored.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from workspace import TEST, OUT, SEEDS, read_json, write_json, write_csv, digest, load_mask
from task_data import task_data
from metrics import geometry, reference_geometry, score, morphology, diversity, aggregate
from crossing_rules import crossing_records
from path_sampler import generate


def evaluate():
    if (OUT / 'test_runs.csv').exists():
        raise RuntimeError('Test results already exist; remove them to rerun.')
    p = read_json(OUT / 'protocol.json')
    lock = read_json(OUT / 'locked_parameters.json')
    if digest(OUT / 'protocol.json') != lock['protocol_sha256']:
        raise RuntimeError('protocol.json changed after the parameters were frozen.')
    if digest(Path(__file__).with_name('path_sampler.py')) != lock['generator_sha256']:
        raise RuntimeError('path_sampler.py changed after the parameters were frozen.')

    config = lock['config']
    rows, excluded, divs = [], [], []
    for i, name in enumerate(p['test_names']):
        task = task_data(TEST, name)
        if task is None:
            excluded.append(name)
            continue
        geo = geometry(task)
        width = reference_geometry(load_mask(TEST, 'crack_binary', name), task)
        probability = np.load(OUT / 'probabilities/test' / (Path(name).stem + '.npy'))
        records = []
        for method in ['unet', 'no_unet']:
            prob = probability if method == 'unet' else np.ones_like(probability)
            local, sampled = [], []
            for seed in SEEDS:
                path, details = generate(prob, geo, seed, config,
                                         p['primary_attempts'], p['attempt_budget'])
                assert np.array_equal(path[0], task['start'])
                assert bool(details['success']) == np.array_equal(path[-1], task['goal'])
                m = score(path, task, details['success'], width)
                m.update(morphology(path, task))
                local.append(dict(config=config['id'], filename=name, method=method, seed=seed,
                                  **{k: v for k, v in details.items() if k != 'success'}, **m))
                sampled.append(path)
                records.append(dict(method=method, seed=seed, points_xy=path[:, ::-1].tolist(),
                                    generation=details, metrics=m,
                                    crossings=crossing_records(path, task)))
                folder = OUT / 'centreline_masks' / method / str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                mask = np.zeros(geo['brick'].shape, np.uint8)
                mask[path[:, 0], path[:, 1]] = 255
                Image.fromarray(mask).save(folder / name)
            d = diversity(sampled, geo['brick'].shape, task['height'])
            divs.append(dict(filename=name, method=method, **d))
            rows.extend(dict(**r, unique_paths=d['unique_seed_paths']) for r in local)
        write_json(OUT / 'paths' / (Path(name).stem + '.json'),
                   dict(filename=name,
                        known_start_xy=task['start'][::-1].tolist(),
                        known_goal_xy=task['goal'][::-1].tolist(),
                        reference_main_path_xy=task['gt'][:, ::-1].tolist(),
                        paths=records))
        if (i + 1) % 5 == 0:
            print(f'Test {i + 1}/{len(p["test_names"])}', flush=True)

    write_csv(OUT / 'test_runs.csv', rows)
    write_csv(OUT / 'seed_diversity.csv', divs)
    write_json(OUT / 'excluded.json', excluded)
    per, summary = aggregate(rows)
    write_csv(OUT / 'per_image.csv', per)
    write_csv(OUT / 'summary.csv', summary)
    write_json(OUT / 'evaluation_complete.json',
               dict(status='complete', images=len({r['filename'] for r in rows}),
                    excluded=len(excluded), config=config,
                    protocol_sha256=digest(OUT / 'protocol.json'),
                    locked_sha256=digest(OUT / 'locked_parameters.json')))
    print('TEST COMPLETE', summary, flush=True)


if __name__ == '__main__':
    evaluate()
