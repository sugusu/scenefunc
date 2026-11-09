"""Audit real archived supervision and calculate declared mathematical controls.

No optimizer, model, new scene, or invented preference label is used. Gradient
examples below are finite mathematical distributions, NOT training results.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

from consequence_contract import exact_admission


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def material_audit(manifest, plan):
    groups = {arm: [r for r in manifest['records'] if r['arm'] == arm]
              for arm in ('separate', 'paired')}
    groups['action_only'] = [r for r in groups['separate'] if r['component'] == 'action']
    totals = {}
    for arm, rows in groups.items():
        families = sorted({r['family_id'] for r in rows})
        counts = {f: {c: sum(r[c + '_tokens'] for r in rows if r['family_id'] == f)
                      for c in ('action', 'decision')} for f in families}
        coefficients = {}
        for c in ('action', 'decision'):
            # Dataset weights len(ds)/(F*N_fc), then Trainer accumulation /len(ds).
            coefficients[c] = {f: (1 / len(families) if counts[f][c] else 0.) for f in families}
        totals[arm] = {'components': len(rows), 'families': len(families),
                       'action_tokens': sum(r['action_tokens'] for r in rows),
                       'decision_tokens': sum(r['decision_tokens'] for r in rows),
                       'input_tokens': sum(r['input_tokens'] for r in rows),
                       'family_token_counts': counts,
                       'family_mean_nll_coefficients': coefficients,
                       'action_coefficient_total': sum(coefficients['action'].values()),
                       'decision_coefficient_total': sum(coefficients['decision'].values())}
    by_arm = {a: {r['id']: r for r in rows} for a, rows in groups.items()}
    if set(by_arm['paired']) != set(by_arm['separate']):
        raise ValueError('Material IDs differ')
    pairs = []
    for key, left in by_arm['separate'].items():
        right = by_arm['paired'][key]
        equal_target = all(left[k] == right[k] for k in
                           ('family_id', 'target_sha256', 'action_tokens', 'decision_tokens', 'prompt_sha256'))
        equal_images = all(left['transformed_sha256'][k] == right['transformed_sha256'][k]
                           for k in ('pixel_values', 'image_grid_thw'))
        if not equal_target or not equal_images:
            raise ValueError('Supervision/observation mismatch: ' + key)
        pairs.append({'id': key, 'component': left['component'], 'equal_target_and_prompt': equal_target,
                      'equal_image_tensor_hashes': equal_images,
                      'equal_input_tensor_hash': left['transformed_sha256']['input_ids'] == right['transformed_sha256']['input_ids'],
                      'input_token_difference': right['input_tokens'] - left['input_tokens']})
    return {'arms': totals, 'paired_vs_separate': pairs, 'planned_arms': plan['arm_order'],
            'paired_extra_action_condition_tokens': sum(r['input_token_difference'] for r in pairs),
            'paired_changed_action_inputs': sum(r['component'] == 'action' and not r['equal_input_tensor_hash'] for r in pairs),
            'actual_tensor_files_loaded': False,
            'scope': 'Published manifest hashes/counts and source-derived weights; not a forward loss or proof of equal FLOPs.'}


def set_likelihood(logits, labels):
    """Ordinary partial-label set likelihood, normalized over OBSERVED candidates.

    Labels are True/False/None. The gradient is analytical, not a model update.
    All-positive groups give zero signal. A group without a positive is skipped.
    """
    if len(logits) != len(labels) or any(not math.isfinite(x) for x in logits):
        raise ValueError('Bad logits')
    if any(y is not None and type(y) is not bool for y in labels):
        raise ValueError('Bad labels')
    known = [i for i, y in enumerate(labels) if y is not None]
    positive = [i for i, y in enumerate(labels) if y is True]
    if not positive:
        return {'loss': None, 'gradient': [0.] * len(labels), 'reason': 'no observed positive'}
    def distribution(indices):
        maximum = max(logits[i] for i in indices)
        weights = {i: math.exp(logits[i] - maximum) for i in indices}
        total = sum(weights.values())
        return maximum + math.log(total), {i: x / total for i, x in weights.items()}
    denominator, q = distribution(known)
    numerator, r = distribution(positive)
    return {'loss': denominator - numerator,
            'gradient': [q.get(i, 0.) - r.get(i, 0.) for i in range(len(labels))],
            'reason': 'observed-subset likelihood; not an unbiased all-candidate objective'}


def scope_audit(pool):
    conditions = {'A': 'finite_separate_witness', 'B': 'finite_joint_witness'}
    rows = []
    for candidate in pool:
        for condition, functional_key in conditions.items():
            value = candidate['scope_values']
            admission = exact_admission(value, ['within_room', 'no_new_mesh_intersection', functional_key])
            target = {'pass': True, 'fail': False, 'unknown': None}[admission['scoped_status']]
            rows.append({'candidate': candidate['id'], 'condition': condition,
                         'primitive_function_target': value[functional_key],
                         'conjunction_target': target,
                         'unknown_primitive_is_not_function_failure': value[functional_key] is None,
                         'geometry_only_prediction': value['within_room'] is True and value['no_new_mesh_intersection'] is True,
                         'whole_room_target': None})
    summaries = {}
    for condition in conditions:
        group = [r for r in rows if r['condition'] == condition]
        measured = [r for r in group if r['conjunction_target'] is not None]
        eligible = [r for r in measured if r['geometry_only_prediction']]
        summaries[condition] = {
            'positive': sum(r['conjunction_target'] is True for r in group),
            'negative': sum(r['conjunction_target'] is False for r in group),
            'unknown': sum(r['conjunction_target'] is None for r in group),
            'geometry_only_correct': sum(r['geometry_only_prediction'] == r['conjunction_target'] for r in measured),
            'measured_denominator': len(measured),
            'geometry_eligible_correct': sum(r['conjunction_target'] for r in eligible),
            'geometry_eligible_denominator': len(eligible)}
    by_candidate = defaultdict(list)
    for r in rows:
        if r['conjunction_target'] is not None:
            by_candidate[r['candidate']].append(r['conjunction_target'])
    contradictions = [key for key, ys in by_candidate.items() if len(set(ys)) > 1]
    oracle_correct = sum(max(Counter(ys).values()) for ys in by_candidate.values())
    observations = sum(len(ys) for ys in by_candidate.values())
    ordinary_correct = sum(r['geometry_only_prediction'] == r['conjunction_target'] for r in rows if r['conjunction_target'] is not None)
    losses = {}
    for condition in conditions:
        for field in ('primitive_function_target', 'conjunction_target'):
            group = [r for r in rows if r['condition'] == condition]
            labels = [r[field] for r in group]
            result = set_likelihood([0.] * len(group), labels)
            # A numerical derivative check of the declared math, not training.
            errors = []
            for i in range(len(group)):
                lo = [0.] * len(group); hi = lo.copy(); lo[i] = -1e-5; hi[i] = 1e-5
                left = set_likelihood(lo, labels)['loss']; right = set_likelihood(hi, labels)['loss']
                if left is not None:
                    errors.append(abs((right-left)/2e-5 - result['gradient'][i]))
            losses[condition + '__' + field] = {**result, 'finite_difference_max_error': max(errors, default=0.)}
            if max(errors, default=0.) > 1e-7:
                raise ValueError('Set-likelihood gradient check failed')
    return {'rows': rows, 'per_condition': summaries,
            'geometry_only': {'correct': ordinary_correct, 'denominator': observations, 'accuracy': ordinary_correct/observations,
                              'is_learned_predictor': False},
            'context_blind_candidate_identity_oracle': {'correct': oracle_correct, 'denominator': observations,
                'accuracy': oracle_correct/observations, 'conflicting_candidate_ids': contradictions,
                'uses_answer_labels': True, 'is_executable_test_policy': False},
            'uniform_logit_math': losses,
            'scope': 'One archived Paris parent only; no full-room preference or generalization claim.'}


def gradient_expectation(p, k):
    """Exact enumeration for iid Bernoulli actions and random continued branch.

    Reward is the chosen action, a pure math distribution. It is not an indoor
    scene, simulation, learning run, or measurement of the upstream model.
    """
    values = {key: [] for key in ('chosen', 'broadcast', 'rescaled_broadcast')}
    for actions in itertools.product((0, 1), repeat=k):
        prob = math.prod(p if a else 1-p for a in actions) / k
        for j in range(k):
            reward = actions[j]
            chosen = reward * (actions[j] - p)
            broadcast = reward * sum(a-p for a in actions) / k
            for key, value in [('chosen', chosen), ('broadcast', broadcast), ('rescaled_broadcast', k*broadcast)]:
                values[key].append((prob, value))
    moments = {}
    for key, entries in values.items():
        mean = sum(prob*v for prob, v in entries)
        moments[key] = {'mean': mean, 'variance': sum(prob*(v-mean)**2 for prob, v in entries)}
    if not math.isclose(moments['chosen']['mean'], p*(1-p), abs_tol=1e-12):
        raise ValueError('Chosen-branch expectation mismatch')
    if not math.isclose(moments['broadcast']['mean'], p*(1-p)/k, abs_tol=1e-12):
        raise ValueError('Broadcast expectation mismatch')
    return {'probability': p, 'branches': k, 'moments': moments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = {str(p): sha(p) for p in (args.material, args.plan, args.pool)}
    material = material_audit(read(args.material), read(args.plan))
    scope = scope_audit(read(args.pool))
    credit = {'examples': [gradient_expectation(p, k) for p in (.2, .5, .8) for k in (2, 4)],
              'scope': 'Exhaustive mathematical expectations only. Requires iid on-policy samples, uniform independent branch choice and chosen-branch reward; actual greedy probe/normalized rewards do not satisfy all assumptions.',
              'no_claim_of_measured_training_bias_or_performance': True}
    dump(args.output / 'material_objective.json', material)
    dump(args.output / 'target_scope.json', scope)
    dump(args.output / 'branch_credit_math.json', credit)
    after = {str(p): sha(p) for p in (args.material, args.plan, args.pool)}
    if before != after:
        raise ValueError('Sources changed')
    dump(args.output / 'receipt.json', {'status': 'completed', 'source_hashes': before, 'sources_unchanged': True,
         'model_calls': 0, 'optimizer_updates': 0, 'scene_executions': 0, 'geometry_rechecks': 0})
    print(json.dumps({'scope_counts': scope['per_condition'], 'geometry_only': scope['geometry_only'],
                      'context_blind_upper_reference': scope['context_blind_candidate_identity_oracle'],
                      'changed_action_inputs': material['paired_changed_action_inputs'],
                      'added_prefix_tokens': material['paired_extra_action_condition_tokens']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
