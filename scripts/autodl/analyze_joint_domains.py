"""Exact ordinary compression of two axis-aligned rectangle domains.

This is computational geometry, not a learned method. The global extrema
reduce the EXISTENCE query to four separation certificates after linear input
reduction. They cannot in general solve three or more activity assignments.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_extremes(a, b):
    if len(a) == 0 or len(b) == 0:
        return {'found': False, 'indices': None, 'separation': None}
    # (work index, dining index, exact separation inequality)
    choices = [
        (int(a[:, 0].argmax()), int(b[:, 2].argmin()), 'dining_left'),
        (int(a[:, 2].argmin()), int(b[:, 0].argmax()), 'dining_right'),
        (int(a[:, 1].argmax()), int(b[:, 3].argmin()), 'dining_below'),
        (int(a[:, 3].argmin()), int(b[:, 1].argmax()), 'dining_above'),
    ]
    for i, j, relation in choices:
        x, y = a[i], b[j]
        ok = {'dining_left': y[2] <= x[0], 'dining_right': y[0] >= x[2],
              'dining_below': y[3] <= x[1], 'dining_above': y[1] >= x[3]}[relation]
        if ok:
            return {'found': True, 'indices': [i, j], 'separation': relation,
                    'work_bounds': x.tolist(), 'dining_bounds': y.tolist()}
    return {'found': False, 'indices': None, 'separation': None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--measured', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.measured / 'frozen_queries.json').read_text())
    rows = []
    for query in frozen:
        r = json.loads((args.measured / (query['id'] + '.json')).read_text())
        f = args.measured / r['domain_file']
        if sha(f) != r['domain_file_sha256']:
            raise ValueError('Changed measured domain')
        t = time.perf_counter()
        with np.load(f, allow_pickle=False) as data:
            a, b = data['work'], data['dining']
        load_time = time.perf_counter() - t
        t = time.perf_counter()
        result = exact_extremes(a, b)
        query_time = time.perf_counter() - t
        if result['found'] != r['finite_joint_witness']:
            raise ValueError('Compressed query and measured Cartesian result disagree')
        supports = {task: {s['id'] for s in r['surface_evidence'] if s['domain_sizes'][task] > 0}
                    for task in ('work', 'dining')}
        distinct = any(x != y for x in supports['work'] for y in supports['dining'])
        rows.append({'candidate': query['id'], 'separate_witness': r['finite_separate_witness'],
                     'joint_witness': r['finite_joint_witness'],
                     'unary_only_joint_prediction': r['finite_separate_witness'],
                     'distinct_support_posthoc_prediction': distinct,
                     'exact_compressed': result,
                     'domain_sizes': r['domain_sizes'],
                     'movement_m': math.hypot(*query['offset_xy_m']),
                     'domain_load_seconds': load_time, 'extrema_query_seconds': query_time,
                     'measured_costs': r['timings_seconds']})
    n = len(rows)
    k = sum(r['joint_witness'] for r in rows)
    entropy = -sum(p * math.log2(p) for p in (k / n, 1-k/n) if p)
    summary = {'states': n, 'parent_scenes': 1, 'new_independent_families': 0,
               'unary_joint_predictor_correct': sum(r['unary_only_joint_prediction'] == r['joint_witness'] for r in rows),
               'best_constant_joint_predictor_correct': max(k, n-k),
               'ordinary_compressed_correct': n,
               'distinct_support_posthoc_correct': sum(r['distinct_support_posthoc_prediction'] == r['joint_witness'] for r in rows),
               'posthoc_rule_is_not_prospective_method_evidence': True,
               'entropy_of_joint_label_given_constant_unary_vector_bits': entropy,
               'condition_blind_candidate_id_oracle_correct': 2*n-sum(r['separate_witness'] != r['joint_witness'] for r in rows),
               'condition_blind_candidate_id_oracle_denominator': 2*n,
               'oracle_uses_labels_not_a_policy': True,
               'cheapest_scoped_actions': {c: min((r for r in rows if r[field]), key=lambda r: (r['movement_m'], r['candidate']))['candidate']
                   for c, field in [('A_separate', 'separate_witness'), ('B_joint', 'joint_witness')]},
               'input_output_boundaries': 'All methods see the same exact measured domains. Predicted geometry, mesh generation, human use and whole-room semantics are not tested.'}
    summary['timings_seconds'] = {}
    for label, ts in {
        'domain_construction': [r['measured_costs']['domain_construction'] for r in rows],
        'cartesian_count': [r['measured_costs']['cartesian_count'] for r in rows],
        'old_ordinary_solver': [r['measured_costs']['ordinary_extreme_solver'] for r in rows],
        'new_extrema_query': [r['extrema_query_seconds'] for r in rows],
        'compressed_domain_load': [r['domain_load_seconds'] for r in rows],
    }.items():
        summary['timings_seconds'][label] = {'sum': sum(ts), 'median': statistics.median(ts), 'maximum': max(ts)}
    (args.output / 'result.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate', 'separate_witness', 'joint_witness', 'unary_only_joint_prediction', 'distinct_support_posthoc_prediction', 'movement_m', 'extrema_query_seconds'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
