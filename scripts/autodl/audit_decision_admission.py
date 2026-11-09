"""Separate knowledge, action fidelity and outcome evidence without editing data.

All corpus entries remain represented. Outcome links are evaluation-only and
cannot be exported as policy observations. This does not authorize training or
invent independent professional labels from successful tool receipts.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

from audit_material_routing import GitInputs, CORPUS, write


def source_entry(knowledge, decision_id):
    if isinstance(knowledge, dict):
        if knowledge.get('decision_id') == decision_id:
            return knowledge
        for value in knowledge.values():
            found = source_entry(value, decision_id)
            if found is not None:
                return found
    elif isinstance(knowledge, list):
        for value in knowledge:
            found = source_entry(value, decision_id)
            if found is not None:
                return found
    return None


def compression_opportunities(fragments, transitions, module):
    """Run only the pinned upstream candidate extractor, never its reviewer/API.

    Failed/unparsed attempts are barriers. The adaptation carries no imagery
    and is not an input accepted by the full four-stage production compressor.
    """
    groups = {}
    for f in fragments:
        groups.setdefault(transitions[f['id']]['cell'], []).append(f)
    result = []
    for cell, fs in groups.items():
        turns, mapping = [], {}
        for f in fs:
            t = transitions[f['id']]
            if not t['native_called']:
                turns.append({'role': 'user', 'content': 'Recorded validation failure: context barrier.'})
                continue
            mapping[len(turns)] = f['id']
            turns.append({'role': 'assistant', 'tool_calls': [t['call']]})
            turns.append({'role': 'tool_results', 'results': f['execution']['outcome']['receipt']['results']})
        episode = {'stage': 'furniture', 'turns': turns}
        segments = module.candidates('furniture', episode)
        offered = []
        for segment in segments:
            allowed = sorted(module.deletable_turns(segment))
            offered.append({'segment_id': segment['segment_id'],
                'recorded_action_ids': [mapping[op['turn_index']] for op in segment['operations']],
                'deletion_candidate_ids': [mapping[i] for i in allowed],
                'repair_confirmed': segment['repair_confirmed']})
        result.append({'cell': cell, 'recorded_decisions': len(fs),
                       'candidate_segments': offered,
                       'deleted_actions': 0, 'quality_verified': False})
    return result


def pose_state(objects):
    """Convert the published pre-action object table to the upstream pose schema."""
    result = {}
    for oid, obj in objects.items():
        q = obj['transform']['rotation_wxyz']
        norm = math.sqrt(sum(x*x for x in q)); w,x,y,z = [v/norm for v in q]
        rpy = [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
               math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
               math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
        result[oid] = {'position': obj['transform']['translation'],
            'rotation': [math.degrees(a) for a in rpy],
            'scale': [1.,1.,1.], 'dimensions': [b-a for a,b in zip(obj['bbox_min'],obj['bbox_max'])],
            'immutable': obj.get('immutable',False)}
    return result


def compression_pose_trials(fragments, transitions, candidates_module, checker):
    groups = {}
    for f in fragments:
        groups.setdefault(transitions[f['id']]['cell'], []).append(f)
    rows=[]
    for cell, fs in groups.items():
        turns=[]; mapping={}; index={f['id']:i for i,f in enumerate(fs)}
        for f in fs:
            t=transitions[f['id']]
            if not t['native_called']:
                turns.append({'role':'user','content':'Recorded rejection barrier.'});continue
            mapping[len(turns)]=f
            turns.append({'role':'assistant','tool_calls':[t['call']]})
            turns.append({'role':'tool_results','results':f['execution']['outcome']['receipt']['results']})
        ep={'stage':'furniture','turns':turns}
        for segment in candidates_module.candidates('furniture',ep):
            allowed=sorted(candidates_module.deletable_turns(segment))
            if not allowed:continue
            plans=[allowed]+[[i] for i in allowed] if len(allowed)>1 else [allowed]
            first=mapping[segment['start']]
            last=segment['operations'][-1]['turn_index']
            end=mapping[last]
            successor=index[end['id']]+1
            if successor>=len(fs):raise ValueError('Expected saved measured next-state input')
            if transitions[end['id']]['after_state_sha256']!=transitions[fs[successor]['id']]['before_state_sha256']:
                raise ValueError('Recorded boundary discontinuity')
            initial=pose_state(first['parent']['objects'])
            reference=pose_state(fs[successor]['parent']['objects'])
            for drops in plans:
                report=checker.check_segment(ep,segment,drops,initial=initial,reference_after=reference)
                # For each retained action after a deletion, compare the recorded
                # endpoint of the last retained action with that action's old input.
                # Equality is evidence about saved states, not new native execution.
                gaps=[]; previous=None; removed_since=False
                for op in segment['operations']:
                    ti=op['turn_index']
                    if ti in drops:
                        removed_since=True;continue
                    current=mapping[ti]
                    if removed_since and previous is not None:
                        gaps.append({'preceding_retained_action':previous['id'],
                            'next_retained_action':current['id'],
                            'recorded_boundary_hash_equal':transitions[previous['id']]['after_state_sha256']==transitions[current['id']]['before_state_sha256']})
                    previous=current;removed_since=False
                report.update(cell=cell,removed_candidate_ids=[mapping[i]['id'] for i in drops],
                    end_id=end['id'],measured_endpoint_requirements=transitions[end['id']]['after'],
                    archived_boundary_comparisons=gaps,
                    full_compression_executed=False,
                    observation_rebinding_performed=False,
                    no_new_label_or_training_admission=True)
                rows.append(report)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--upstream-source', type=Path, required=True)
    ap.add_argument('--upstream-index', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    src = GitInputs(Path('.'))
    corpus = src.json(CORPUS)
    coverage_path = Path('runs/autodl/20261110_conditional_evidence/actual/condition_coverage.json')
    coverage = json.loads(coverage_path.read_text())
    cov = {r['unit_id']: r for r in coverage['rows']}
    full_ledger, reasons, outcomes = [], [], {}
    for unit in corpus['rows']:
        inp, input_path = src.reference(unit['input'])
        ep, episode_path = src.reference(unit['documents']['decision_action'])
        t = ep['turns'][0]
        reason = json.loads(t['content']) if unit['target_kind'] == 'tool_action' and t.get('content') else None
        link = cov[unit['id']].get('source_consequence')
        if link:
            data = src.json(link['path'])
            if src.manifest[link['path']] != link['sha256']:
                raise ValueError('Outcome source mismatch')
            # Preserve original evidence; never use one episode outcome as a label for every action.
            outcomes[link['path']] = data
        calls = t.get('tool_calls', [])
        local_receipt = ep.get('evaluation_only', {}).get('actual_reply', {}).get('results', [])
        source_decision = None
        if reason:
            did = reason['principle_applicability']['source_decision_id']
            source_decision = source_entry(inp['context']['available_source_knowledge'], did)
            if source_decision is None:
                raise ValueError('No same-input source principle: '+unit['id'])
            wanted = reason['principle_applicability']['sourced_principle']
            if wanted not in json.dumps(source_decision, ensure_ascii=False):
                raise ValueError('Attributed principle differs')
            target_ids = [c['arguments'].get('object_id', c['arguments'].get('asset_id')) for c in calls]
            named = reason['proposed_decision']['object_or_asset_references']
            if not all(oid and oid in named for oid in target_ids):
                raise ValueError('Reason/action identity mismatch')
            reasons.append({'id': unit['id'], 'family_id': unit['family_id'],
                'source_decision_id': did, 'source_knowledge': source_decision,
                'input_path': input_path, 'episode_path': episode_path,
                'decision': reason, 'calls': calls,
                'authorship': 'Only the sourced principle is attributed; state application, plan and costs are researcher annotations',
                'evidence_for_action_quality': link,
                'reason_is_not_a_positive_quality_label': True})
        full_ledger.append({'id': unit['id'], 'family_id': unit['family_id'],
            'target_kind': unit['target_kind'], 'input_path': input_path, 'episode_path': episode_path,
            'has_attributed_principle': reason is not None,
            'local_recorded_success_receipt': bool(local_receipt) and all(r['output'].get('success') is True for r in local_receipt),
            'execution_evidence_reference_present': bool(unit.get('source', {}).get('execution_evidence')),
            'action_target': calls,
            'outcome_reference': link,
            'scoped_terminal_target': unit.get('target') if unit['target_kind']=='terminal_control' else None,
            'supervision_roles': {'source_knowledge': 'attributed_paraphrase' if reason else 'no_reason_target',
                'decision_application': 'researcher_reconstruction' if reason else 'not_supervised',
                'action_imitation': 'recorded_target' if calls else 'not_an_edit',
                'outcome_value': 'not_automatically_labelled',
                'whole_room_professional_quality': 'unknown'},
            'existing_training_authorization_reinterpreted': False})
    upstream = args.upstream_source/'trajectory_compression/candidates.py'
    index = json.loads(args.upstream_index.read_text())
    source_hash = hashlib.sha256(upstream.read_bytes()).hexdigest()
    assert next(x['sha256'] for x in index if x['path']=='trajectory_compression/candidates.py') == source_hash
    package = types.ModuleType('pinned_trajectory_compression')
    package.__path__ = [str(upstream.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__+'.candidates', upstream)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # This file has only stdlib imports, definitions and constants; no model client.
    effects = Path('runs/autodl/20261110_executed_effects/actual_v2')
    fs = [json.loads(l) for l in (effects/'source_fragments.jsonl').read_text().splitlines()]
    ts = {r['id']: r for r in map(json.loads, (effects/'transitions.jsonl').read_text().splitlines())}
    opportunities = compression_opportunities(fs, ts, module)
    checker_path=upstream.parent/'offline_check.py'
    assert hashlib.sha256(checker_path.read_bytes()).hexdigest()==next(x['sha256'] for x in index if x['path']=='trajectory_compression/offline_check.py')
    cspec=importlib.util.spec_from_file_location(package.__name__+'.offline_check',checker_path)
    checker=importlib.util.module_from_spec(cspec);sys.modules[cspec.name]=checker;cspec.loader.exec_module(checker)
    trials=compression_pose_trials(fs,ts,module,checker)
    summary = {'status': 'completed_offline_evidence_and_candidate_audit',
        'training_units': len(full_ledger), 'families': len({r['family_id'] for r in full_ledger}),
        'target_kinds': dict(Counter(r['target_kind'] for r in full_ledger)),
        'reason_units': len(reasons), 'source_principles': len({r['source_decision_id'] for r in reasons}),
        'linked_episode_outcome_records': len(outcomes),
        'local_recorded_success_receipts': sum(r['local_recorded_success_receipt'] for r in full_ledger),
        'quality_labels_generated_from_success': 0,
        'archived_policy_trajectories': len(opportunities),
        'archived_policy_decisions': len(fs),
        'upstream_candidate_segments': sum(len(r['candidate_segments']) for r in opportunities),
        'upstream_candidate_deletions': sum(len(s['deletion_candidate_ids']) for r in opportunities for s in r['candidate_segments']),
        'actual_compressed_trajectories': 0,
        'offline_pose_deletion_trials':len(trials),
        'offline_pose_trial_status':dict(Counter(t['status'] for t in trials)),
        'stale_observation_trials':sum(bool(t['changed_inputs']) for t in trials),
        'source_sha256': src.manifest,
        'local_source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [coverage_path,effects/'source_fragments.jsonl',effects/'transitions.jsonl']},
        'upstream_candidate_source_sha256': source_hash,
        'wall_seconds': time.perf_counter()-started,
        'model_calls': 0, 'training_updates': 0, 'native_scene_calls': 0,
        'scope': 'Source-preserving role manifest plus actual official candidate extraction on adapted furniture-only records; not complete upstream compression or a quality filter.'}
    write(args.output/'unit_roles.json', full_ledger)
    write(args.output/'reason_evidence.json', reasons)
    write(args.output/'linked_outcomes.json', outcomes)
    write(args.output/'compression_candidates.json', opportunities)
    write(args.output/'compression_pose_trials.json',trials)
    write(args.output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if 'sha256' not in k}, ensure_ascii=False))


if __name__ == '__main__':
    main()
