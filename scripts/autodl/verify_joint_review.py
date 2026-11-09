"""Verify delivered evidence; this is not another generation experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--seal', action='store_true', help='Create manifest once, then verify it')
    args = ap.parse_args()
    repo = args.repo.resolve()
    root = repo / 'runs/autodl/20261110_joint_representation'
    manifest_path = root / 'manifest.json'
    scripts = [repo / 'scripts/autodl' / x for x in (
        'measure_eligible_joint_pool.py', 'analyze_joint_domains.py',
        'build_joint_review.py', 'verify_joint_review.py')]
    if args.seal:
        if manifest_path.exists():
            raise ValueError('Manifest already exists; do not silently reseal changed evidence')
        files = [p for p in root.rglob('*') if p.is_file()
                 and p.name not in ('manifest.json', 'verification_receipt.json')]
        files += scripts
        entries = {str(p.relative_to(repo)): sha(p) for p in sorted(files)}
        manifest_path.write_text(json.dumps({'files': entries}, indent=2) + '\n')
    manifest = read(manifest_path)['files']
    for relative, expected in manifest.items():
        if sha(repo / relative) != expected:
            raise ValueError('Delivery hash mismatch: ' + relative)
    summary = read(root / 'measured/summary.json')
    analysis = read(root / 'analysis/result.json')
    rows = analysis['rows']
    queries = read(root / 'measured/frozen_queries.json')
    if [q['id'] for q in queries] != [r['candidate'] for r in rows]:
        raise ValueError('Candidate order changed')
    labels = []
    pair_total = 0
    for row in rows:
        m = read(root / 'measured' / (row['candidate'] + '.json'))
        archive = root / 'measured' / m['domain_file']
        if sha(archive) != m['domain_file_sha256']:
            raise ValueError('Domain archive changed')
        with np.load(archive, allow_pickle=False) as data:
            a, b = data['work'], data['dining']
        if any(x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all() for x in (a, b)):
            raise ValueError('Invalid rectangle domain')
        if any(not ((x[:, 2:] - x[:, :2]) > 0).all() for x in (a, b)):
            raise ValueError('Degenerate rectangles')
        # Full pair enumeration, using the negation of overlap on both axes.
        # It does not call the compact query or old strong_solve implementation.
        count = 0
        for w in a:
            overlap_x = (b[:, 0] < w[2]) & (w[0] < b[:, 2])
            overlap_y = (b[:, 1] < w[3]) & (w[1] < b[:, 3])
            count += int((~(overlap_x & overlap_y)).sum())
        if count != m['exhaustive']['compatible_pairs']:
            raise ValueError('Full pair count differs')
        found = count > 0
        if found != row['joint_witness'] or found != row['exact_compressed']['found']:
            raise ValueError('Scoped labels disagree')
        if found:
            i, j = row['exact_compressed']['indices']
            if not (np.array_equal(a[i], row['exact_compressed']['work_bounds'])
                    and np.array_equal(b[j], row['exact_compressed']['dining_bounds'])):
                raise ValueError('Witness is not from measured domains')
            overlap = ((a[i, 0] < b[j, 2]) and (b[j, 0] < a[i, 2]) and
                       (a[i, 1] < b[j, 3]) and (b[j, 1] < a[i, 3]))
            if overlap:
                raise ValueError('Returned witness overlaps')
        labels.append((bool(len(a) and len(b)), found))
        pair_total += len(a) * len(b)
    if (len(rows), sum(x for x, _ in labels), sum(y for _, y in labels)) != (21, 21, 9):
        raise ValueError('Unexpected aggregate labels')
    table = list(csv.DictReader((root / 'merged_label_table.csv').open()))
    if len(table) != 98 or sum(r['new_measurement_available'] == 'True' for r in table) != 11:
        raise ValueError('Invalid coverage update')
    for path, expected in summary['source_hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Historical input changed: ' + path)
    official = read(root / 'source_index.json')
    for source in official['files']:
        if sha(Path(source['path'])) != source['sha256']:
            raise ValueError('Read official source changed')
    receipt = {'delivery_hashes_match': len(manifest), 'historical_sources_unchanged': len(summary['source_hashes']),
               'official_sources_unchanged': len(official['files']), 'candidate_domains_verified': len(rows),
               'rectangle_pairs_recounted': pair_total, 'labels': {'separate': 21, 'joint': 9},
               'new_primitive_labels': 11, 'merged_rows': 98,
               'scope': 'Artifact integrity and finite-domain numerical recount only; no new meshes, FCL, generation, inference or training.'}
    (root / 'verification_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
