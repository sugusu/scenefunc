"""Recover observed one-step effects from a frozen, complete policy batch.

The 920 development family is evaluation-only. No policy, preference label,
counterfactual action, geometry simulation, or training example is invented.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import time


RUN = 'runs/920/P920-EVALFIX-v1_20261107T130916Z'
STATE_MARKER = 'CURRENT NATIVE STATE:'
FEEDBACK_MARKER = 'CURRENT MEASURED FEEDBACK (declared proxies; no reference solution):'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    # Same exact representation as the frozen native_checkpoint.canonical_hash.
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def extract(messages):
    texts = []
    for m in messages:
        if m['role'] != 'user':
            continue
        texts.append(m['content'] if isinstance(m['content'], str) else '\n'.join(
            c['text'] for c in m['content'] if c.get('type') == 'text'))
    full = '\n'.join(texts)
    values = []
    for marker in (STATE_MARKER, FEEDBACK_MARKER):
        if full.count(marker) != 1:
            raise ValueError('Ambiguous observation marker: ' + marker)
        values.append(json.JSONDecoder().raw_decode(full.split(marker, 1)[1].lstrip())[0])
    return values


def compare(before, after):
    if before.keys() != after.keys():
        raise ValueError('Requirement universe changed')
    if any(v is not None and type(v) is not bool for v in [*before.values(), *after.values()]):
        raise ValueError('Only measured bool or explicitly unknown allowed')
    known = [k for k in before if before[k] is not None and after[k] is not None]
    repairs = [k for k in known if before[k] is False and after[k] is True]
    harms = [k for k in known if before[k] is True and after[k] is False]
    return dict(known=known, unknown=[k for k in before if k not in known],
                repaired=repairs, damaged=harms,
                became_unknown=[k for k in before if before[k] is not None and after[k] is None],
                known_status_changed=bool(repairs or harms),
                scoped_improvement_without_new_harm=bool(repairs) and not harms
                and not any(before[k] is True and after[k] is None for k in before))


def dominates(a, b):
    if a.keys() != b.keys() or any(v is None for v in [*a.values(), *b.values()]):
        return None
    return all(int(a[k]) >= int(b[k]) for k in a) and any(a[k] != b[k] for k in a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); hashes = {}

    def raw(path):
        data = subprocess.check_output(['git', 'show', args.pin+':'+path])
        hashes[path] = digest(data)
        return data

    def read(path):
        return json.loads(raw(path))

    try:
        cells = list(csv.DictReader(io.StringIO(raw(RUN+'/LEARNING_RESULTS.csv').decode())))
        records = []; trajectories = []; source_fragments = []
        for cell in cells:
            base = RUN+'/evaluation/'+cell['id']
            result = read(base+'/ROLLOUT_RESULT.json')
            rounds = result['rounds']
            requests = [read(base+f'/recorded/round_{r["turn"]:03d}/request.json') for r in rounds]
            extracted = [extract(req) for req in requests]
            final_audit = read(base+'/recorded/final_audit.json')
            final_state = read(base+'/recorded/final_state.json')
            assert canonical(final_state) == final_audit['state_sha256']
            assert final_audit == result['final_audit']
            final_feedback = final_audit['feedback']
            assert final_feedback['checks'] == {x['requirement']:x['pass'] for x in final_audit['review']['checks']}
            rows = []
            for i, r in enumerate(rounds):
                state, before_feedback = extracted[i]
                after_feedback = extracted[i+1][1] if i+1 < len(rounds) else final_feedback
                ex = r['execution']
                assert ex['before_sha256'] == r['input_history']['current_native_state_sha256']
                expected_after = rounds[i+1]['input_history']['current_native_state_sha256'] if i+1 < len(rounds) else final_audit['state_sha256']
                assert ex['after_sha256'] == expected_after
                assert before_feedback['requested_mode'] == after_feedback['requested_mode'] == cell['condition']
                assert before_feedback['declared_requirements'] == after_feedback['declared_requirements']
                before, after = before_feedback['checks'], after_feedback['checks']
                delta = compare(before, after)
                outcome = ex['outcome']
                native = outcome['kind'] == 'native_reply'
                returned = outcome.get('receipt', {}).get('results', [])
                assert native is bool(returned)
                technical_success = native and all(x['output'].get('success') is True for x in returned)
                state_changed = ex['before_sha256'] != ex['after_sha256']
                if not state_changed:
                    assert before == after, 'A stationary state changed deterministic checks'
                calls = ex['calls']; assert len(calls) <= 1
                name = calls[0]['name'] if calls else None
                args_dict = calls[0]['arguments'] if calls else {}
                # This baseline sees only pre-action constraints and the action.
                # It does not read a measured post-state or outcome.
                permitted_ids = before_feedback['declared_requirements']['movable_ids']
                ordinary_task_guard = (name not in ('remove_furniture_tool', 'add_furniture_to_scene_tool')
                    and (name != 'move_furniture_tool' or args_dict.get('object_id') in permitted_ids))
                parent = {'objects':state['objects'], 'requested_mode':before_feedback['requested_mode'],
                          'declared_requirements':before_feedback['declared_requirements'], 'checks':before}
                row = dict(id=cell['id']+f'__{i:02d}', cell=cell['id'], arm=cell['arm'],
                    condition=cell['condition'], seed=int(cell['seed']), turn=i, call=calls[0] if calls else None,
                    parent_group=canonical(parent), before_state_sha256=ex['before_sha256'],
                    after_state_sha256=ex['after_sha256'], before=before, after=after, delta=delta,
                    native_called=native, tool_success=technical_success, outcome_kind=outcome['kind'],
                    state_changed=state_changed, ordinary_preservation_guard_allows=ordinary_task_guard,
                    preserve_prediction_matches=sum(before[k] == after[k] for k in delta['known']),
                    preserve_prediction_known=len(delta['known']),
                    training_admission=False, evaluation_only=True,
                    source_result=base+'/ROLLOUT_RESULT.json', source_request=base+f'/recorded/round_{i:03d}/request.json')
                records.append(row); rows.append(row)
                source_fragments.append(dict(id=row['id'], original_request_sha256=hashes[row['source_request']],
                    model_recorded_request_sha256=r['model'].get('request_sha256'),
                    parent=parent, before_measurements=before_feedback['measurements'],
                    after_measurements=after_feedback['measurements'], execution=ex,
                    raw_model_output=r['raw_output']))
            assert len(rows) == int(cell['model_calls'])
            assert sum(x['native_called'] for x in rows) == int(cell['native_calls'])
            trajectories.append(dict(cell=cell['id'], transitions=len(rows),
                initial=rows[0]['before'], final=rows[-1]['after'],
                final_matches_recorded_activity=final_audit['review']['requested_activity_status']==cell['activity_status']))
        groups = defaultdict(list)
        for r in records:
            groups[r['parent_group']].append(r)
        pairs=[]
        for parent, group in groups.items():
            # Deduplicate repeated identical actions. Never fabricate an unexecuted sibling.
            distinct = {}
            for r in group:
                key=canonical(r['call'])
                if key in distinct:
                    assert distinct[key]['after'] == r['after'], 'Nondeterministic measured effect; cannot collapse'
                else: distinct[key]=r
            for a,b in itertools.combinations(distinct.values(),2):
                ab,ba=dominates(a['after'],b['after']),dominates(b['after'],a['after'])
                pairs.append(dict(parent=parent,a=a['id'],b=b['id'],
                    a_dominates_b=ab,b_dominates_a=ba,
                    both_native_success=a['tool_success'] and b['tool_success'],
                    both_moves=bool(a['call'] and b['call'] and a['call']['name']==b['call']['name']=='move_furniture_tool'),
                    guard_a=a['ordinary_preservation_guard_allows'],guard_b=b['ordinary_preservation_guard_allows'],
                    scope='Observed one-step requirement-vector comparison, not human preference or full trajectory value'))
        summary=dict(trajectories=len(trajectories), independent_families=1, decisions=len(records),
            native_calls=sum(r['native_called'] for r in records), tool_success=sum(r['tool_success'] for r in records),
            state_changes=sum(r['state_changed'] for r in records),
            known_check_changes=sum(len(r['delta']['repaired'])+len(r['delta']['damaged']) for r in records),
            repaired_check_events=sum(len(r['delta']['repaired']) for r in records),
            damaged_check_events=sum(len(r['delta']['damaged']) for r in records),
            no_change_baseline_correct=sum(r['preserve_prediction_matches'] for r in records),
            no_change_baseline_known=sum(r['preserve_prediction_known'] for r in records),
            decisions_with_repairs=sum(bool(r['delta']['repaired']) for r in records),
            decisions_with_damage=sum(bool(r['delta']['damaged']) for r in records),
            decisions_improve_without_new_harm=sum(r['delta']['scoped_improvement_without_new_harm'] for r in records),
            successful_tools_with_damage=sum(r['tool_success'] and bool(r['delta']['damaged']) for r in records),
            ordinary_task_guard_rejections=sum(not r['ordinary_preservation_guard_allows'] for r in records),
            ordinary_guard_rejected_damage=sum(not r['ordinary_preservation_guard_allows'] and bool(r['delta']['damaged']) for r in records),
            ordinary_guard_allowed_damage=sum(r['ordinary_preservation_guard_allows'] and bool(r['delta']['damaged']) for r in records),
            unique_observed_parents=len(groups), observed_distinct_action_pairs=len(pairs),
            comparable_dominance_pairs=sum(bool(p['a_dominates_b'] or p['b_dominates_a']) for p in pairs),
            both_successful_move_dominance_pairs=sum(p['both_moves'] and p['both_native_success'] and bool(p['a_dominates_b'] or p['b_dominates_a']) for p in pairs),
            source_pin=args.pin, source_run=RUN, wall_seconds=time.perf_counter()-started,
            new_model_calls=0,new_training_updates=0,new_scene_executions=0,
            scope='Saved development trajectories only. One-step measured proxies, not whole-room, professional preference, or independent generalization. Native success is a receipt diagnostic, not a prospective predictor. No-change and task-guard controls are ordinary rules, not new algorithms.')
        for name,value in [('summary',summary),('pairs',pairs),('trajectories',trajectories),('sources',hashes)]:
            (args.output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        for name,values in [('transitions',records),('source_fragments',source_fragments)]:
            (args.output/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values))
        print(json.dumps(summary,ensure_ascii=False))
    except Exception as e:
        (args.output/'failure.json').write_text(json.dumps({'error':type(e).__name__,'message':str(e),
            'source_hashes_read':hashes,'elapsed':time.perf_counter()-started},indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
