"""Summarise the paired test results and write the report tables.

The paired intervals are clustered by filename family. A filename family is a
grouping proxy taken from the export's naming; it is not a verified physical
wall identity, so the intervals are descriptive.
"""
from pathlib import Path

import numpy as np

from workspace import OUT, read_json, read_csv, write_csv, write_json


def family(name):
    """The filename-family proxy used to cluster the bootstrap."""
    import re
    stem = re.split(r'_(?:jpg|jpeg|png)(?:[._]|$)', name, flags=re.I)[0]
    stem = re.sub(r'\.rf\..*$', '', stem)
    if re.fullmatch(r'\d+(?:[-_]\d+)*', stem):
        return 'numeric_' + str(int(re.split('[-_]', stem)[0]))
    if re.fullmatch(r'(?:a_\d+|l\d+)_\d+', stem):
        return stem.rsplit('_', 1)[0]
    return stem


def paired_interval(values, names, families, draws=10000, seed=20260917):
    """A 95% interval for a paired difference, resampling whole families."""
    unique = list(dict.fromkeys(families[n] for n in names))
    sums = np.array([sum(v for n, v in zip(names, values) if families[n] == f) for f in unique])
    counts = np.array([sum(families[n] == f for n in names) for f in unique])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(draws, len(unique)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high), len(unique)


def main():
    protocol = read_json(OUT / 'protocol.json')
    lock = read_json(OUT / 'locked_parameters.json')
    per = read_csv(OUT / 'per_image.csv')
    summary = read_csv(OUT / 'summary.csv')

    names = sorted({r['filename'] for r in per})
    families = {n: family(n) for n in names}
    subsets = {
        'all_test': set(protocol['test_names']),
        'no_filename_family_overlap': set(protocol['test_no_family_overlap_names']),
    }

    rows, paired = [], []
    for label, keep in subsets.items():
        selected = [n for n in names if n in keep]
        for method in ['unet', 'no_unet']:
            values = {key: [float(r[key]) for r in per
                            if r['filename'] in keep and r['method'] == method]
                      for key in ['max_f1', 'mean_f1', 'completion']}
            rows.append(dict(subset=label, method=method, images=len(selected),
                             **{k: float(np.mean(v)) for k, v in values.items()}))
        for key in ['max_f1', 'mean_f1', 'completion']:
            lookup = {(r['method'], r['filename']): float(r[key]) for r in per}
            differences = [lookup[('unet', n)] - lookup[('no_unet', n)]
                           for n in selected if ('unet', n) in lookup and ('no_unet', n) in lookup]
            paired_names = [n for n in selected if ('unet', n) in lookup and ('no_unet', n) in lookup]
            low, high, clusters = paired_interval(differences, paired_names, families)
            paired.append(dict(subset=label, metric=key,
                               difference_unet_minus_uniform=float(np.mean(differences)),
                               ci95_low=low, ci95_high=high,
                               images=len(paired_names), filename_family_clusters=clusters))

    write_csv(OUT / 'comparison_summary.csv', rows)
    write_csv(OUT / 'paired_comparisons.csv', paired)
    write_json(OUT / 'results.json',
               dict(locked_parameters=lock, summary=rows, paired=paired))

    lines = ['# Learned against uniform spatial prior', '',
             f'Frozen configuration: `{lock["config"]["id"]}`, '
             f'gamma={lock["config"]["gamma"]}, lambda={lock["config"]["inertia"]}.', '',
             'Both methods use the same layouts, supplied endpoints, five seeds and sampling',
             'budget. The uniform method replaces the predicted probability with a constant map',
             'and keeps every geometric rule.', '',
             '## Results', '',
             '| Subset | Method | Images | Best-of-five F1 | Mean sampled F1 | Completion |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["subset"]} | {r["method"]} | {r["images"]} | '
                     f'{r["max_f1"]:.3f} | {r["mean_f1"]:.3f} | {r["completion"]:.3f} |')
    lines += ['', '## Paired differences, learned minus uniform', '',
              'Intervals come from 10000 bootstrap draws over whole filename families.', '',
              '| Subset | Metric | Difference | 95% interval | Images | Families |',
              '|---|---|---:|---|---:|---:|']
    for r in paired:
        lines.append(f'| {r["subset"]} | {r["metric"]} | '
                     f'{r["difference_unet_minus_uniform"]:.3f} | '
                     f'[{r["ci95_low"]:.3f}, {r["ci95_high"]:.3f}] | '
                     f'{r["images"]} | {r["filename_family_clusters"]} |')
    lines += ['', '## Scope', '',
              '- The comparison is conditional on supplied reference endpoints. It does not',
              '  measure crack detection from an RGB image.',
              '- The layouts are reconstructed from the annotations, so they may retain shape',
              '  information related to the target.',
              '- A filename family is a naming proxy, not a verified physical wall, so the',
              '  intervals are descriptive rather than a test of wall-level independence.',
              '- These are centreline path metrics, not the downstream detector mask mAP.',
              '- The eighteen-setting grid is produced separately by `sweep_gamma_lambda.py`',
              '  and `build_prior_ablation.py`.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('REPORT COMPLETE', OUT / 'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
