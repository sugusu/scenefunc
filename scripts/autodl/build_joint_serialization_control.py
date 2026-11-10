"""Prepare same-material ordinary joint autoregressive SFT control.

Reuses every recorded D/A target token and coefficient exactly once. Missing D
stays missing. This is data/protocol preparation, not training or a new method.
The resulting one-call prompt removes only our exact phase instruction suffix.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import time

from phase_objective import PHASE_MODE
from routed_multimodal import frame_completed_decision


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def raw_messages(component):
    messages = copy.deepcopy(component['messages'])
    suffix = PHASE_MODE[component['stage']]
    text = messages[0]['content']
    if not text.endswith(suffix):
        raise ValueError('Exact phase suffix absent; do not guess original prompt')
    messages[0]['content'] = text[:-len(suffix)]
    return messages


def build(a):
    start = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.processor, local_files_only=True,
                                             trust_remote_code=False)
    data = a.material.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines()]
    by = {(r['unit_id'], r['arm'], r['stage']): r for r in rows}
    actions = [r for r in rows if r['arm'] == 'paired_phase_v3' and r['stage'] == 'action']
    output = []
    comparisons = []
    source_tokens = 0
    source_weight = 0.
    for action in actions:
        unit = action['unit_id']
        decision = by.get((unit, 'paired_phase_v3', 'decision'))
        components = ([decision] if decision else []) + [action]
        original = raw_messages(action)
        for c in components:
            if raw_messages(c) != original:
                raise ValueError('D and A observations differ: ' + unit)
            other = by[(unit, 'separate_phase_v3', c['stage'])]
            for key in ('target_token_ids', 'token_objective_weights', 'source_token_origins'):
                if c[key] != other[key]:
                    raise ValueError('Arm supervision differs: ' + unit + ':' + key)
        text = ''.join(c['target_text'] for c in components)
        tokens = [t for c in components for t in c['target_token_ids']]
        weights = [w for c in components for w in c['token_objective_weights']]
        if tokenizer.encode(text, add_special_tokens=False) != tokens:
            raise ValueError('Joining changed target tokens: ' + unit)
        if len(tokens) != len(weights):
            raise ValueError('Target weight length mismatch')
        if decision and action['assistant_prefix'] != frame_completed_decision(decision['target_text']):
            raise ValueError('Conditioned action prefix differs from documented framing: ' + unit)
        if not decision and action['assistant_prefix']:
            raise ValueError('Unrecorded decision prefix')
        source_tokens += len(tokens)
        source_weight += sum(weights)
        d_len = len(decision['target_token_ids']) if decision else 0
        # A text-token witness of the protocol difference. These are NOT
        # multimodal prefill costs or inference measurements.
        raw_system_ids = tokenizer.encode(original[0]['content'], add_special_tokens=False)
        a_system_ids = tokenizer.encode(action['messages'][0]['content'], add_special_tokens=False)
        comparisons.append(dict(unit_id=unit, raw_system_tokens=len(raw_system_ids),
            action_system_tokens=len(a_system_ids), current_action_system_equals_joint=raw_system_ids == a_system_ids,
            split_host_prefix_equals_joint_decision=(action['assistant_prefix'] ==
                (decision['target_text'] if decision else '')),
            split_host_separator_tokens=action['host_separator_tokens_added'],
            recorded_decision=bool(decision), decision_target_tokens=d_len,
            action_target_tokens=len(action['target_token_ids'])))
        output.append(dict(unit_id=unit, family_id=action['family_id'], arm='joint_same_material_v1',
            messages=original, assistant_prefix='', target_text=text,
            target_token_ids=tokens, token_objective_weights=weights,
            decision_target_mask=[True]*d_len+[False]*(len(tokens)-d_len),
            action_target_mask=[False]*d_len+[True]*(len(tokens)-d_len),
            source_token_origins=[v for c in components for v in c['source_token_origins']],
            source_request_sha256=action['source_request_sha256'],
            target_sha256=hashlib.sha256(text.encode()).hexdigest(),
            input_sha256=hashlib.sha256(canonical(original).encode()).hexdigest(),
            note='Native-style joint serialization; no added decisions, no optimizer updates'))
    if len(output) != 24 or sum(r['recorded_decision'] for r in comparisons) != 10:
        raise ValueError('Frozen source coverage changed')
    (a.out/'components.jsonl').write_text(''.join(canonical(x)+'\n' for x in output))
    (a.out/'protocol_comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    summary = dict(status='same_material_joint_targets_prepared', units=len(output),
        families=len({x['family_id'] for x in output}), recorded_decisions=10,
        missing_decisions_retained=14, source_tokens_once=source_tokens,
        coefficient_sum=source_weight, new_labels=0, optimizer_updates=0,
        actual_model_calls=0, paid_api_calls=0, full_scene_runs=0,
        source_material_sha256=hashlib.sha256(data).hexdigest(),
        actual_tokenizer=True, every_joined_target_token_verified=True,
        text_protocol_changed_in_current_split_arm=sum(not x['current_action_system_equals_joint'] for x in comparisons),
        split_prefix_differs_from_joint=sum(not x['split_host_prefix_equals_joint_decision'] for x in comparisons),
        full_multimodal_encoding_executed=False, training_ready=False,
        seconds=time.perf_counter()-start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Ordinary joint-SFT control data. No loss/gradient equality claim '
        'for the current split implementation, whose system instructions differ. '
        'Full multimodal encoding, training, native joint host and scenes remain unrun.')
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert hashlib.sha256(a.material.read_bytes()).hexdigest() == summary['source_material_sha256']
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('material', 'processor', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    build(ap.parse_args())
