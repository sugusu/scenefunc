#!/usr/bin/env python3
"""Read saved rollout evidence; no model, scene edits, or metric redefinition."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def fingerprint(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def text_of(messages):
    return '\n'.join(block['text'] for msg in messages
                     if isinstance(msg.get('content'), list)
                     for block in msg['content'] if block.get('type') == 'text')


def checks(review):
    return {row['requirement']: row['pass'] for row in review['checks']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    accepted_path = args.public/'review/FINAL_ACCEPTANCE.json'
    accepted = read(accepted_path)
    assert accepted['status'] == 'artifact_and_accounting_checks_passed'
    manifest = [fingerprint(accepted_path)]
    all_steps, cells = [], []
    for cell in accepted['cells']:
        public = args.public/'evaluation'/cell['id']/'recorded'
        result_path = public/'RESULT.json'
        result = read(result_path)
        runtime = Path(cell['result_path']).parent
        manifest.append(fingerprint(result_path))
        previous = result['initial_review']
        assert previous['requested_activity_status'] == 'fail', cell['id']
        initial = checks(previous)
        first_loss = None
        first_remove = None
        good_prefixes = []
        activity_prefixes = []
        seen_calls = set()
        duplicates = 0
        feedback_ok = 0
        loss_visible_count = 0
        removal_attempts = 0
        native_removals = 0
        for row in result['rounds']:
            turn = row['turn']
            rd = public/f'round_{turn:03d}'
            request_path = rd/'request.json'
            request = read(request_path)
            txt = text_of(request)
            marker = 'CURRENT MEASURED FEEDBACK (declared proxies; no reference solution):\n'
            assert txt.count(marker) == 1, (cell['id'], turn)
            supplied, _ = json.JSONDecoder().raw_decode(txt.split(marker)[1])
            before = checks(previous)
            assert supplied['checks'] == before, (cell['id'], turn, 'feedback mismatch')
            feedback_ok += 1
            assert '保留所有对象实例，仅可移动小桌和凳子' in txt
            visible_loss = [k for k,v in before.items() if initial.get(k) is True and v is False]
            loss_visible_count += bool(visible_loss)
            manifest.append(fingerprint(request_path))
            call = row.get('call')
            # Rejected valid JSON proposals also retain attempted tool identity.
            proposed = row.get('execution', {}).get('calls', [])
            removal = any(x.get('name') == 'remove_furniture_tool' for x in proposed)
            removal_attempts += removal
            if call:
                key = json.dumps(call, sort_keys=True)
                duplicates += key in seen_calls
                seen_calls.add(key)
            if call and call['name'] == 'remove_furniture_tool':
                native_removals += 1
                if first_remove is None:
                    first_remove = turn
            audit_path = runtime/f'round_{turn:03d}'/'audit.json'
            execution = row.get('execution', {})
            changed = execution.get('after_sha256') != execution.get('before_sha256')
            if changed:
                assert audit_path.exists(), (cell['id'], turn, 'missing changed-state audit')
                audit = read(audit_path)
                assert audit['state_sha256'] == execution['after_sha256']
                dest = args.out/'audits'/cell['id']/f'round_{turn:03d}.json'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(audit_path.read_bytes())
                manifest.append(fingerprint(dest))
                previous = audit['review']
            else:
                assert not audit_path.exists(), (cell['id'], turn, 'unexpected audit')
            if call and call['name'] == 'remove_furniture_tool':
                assert changed, (cell['id'], turn, 'removal did not change state')
            after = checks(previous)
            losses = [k for k,v in after.items() if initial.get(k) is True and v is False]
            newly_failed = [k for k,v in after.items() if before.get(k) is True and v is False]
            if losses and first_loss is None:
                first_loss = turn
            activity = previous['requested_activity_status'] == 'pass'
            if activity:
                activity_prefixes.append(turn)
            if activity and not losses:
                good_prefixes.append(turn)
            step = dict(cell_id=cell['id'],arm=cell['arm'],condition=cell['condition'],seed=cell['seed'],turn=turn,
                        input_feedback_exact=True,preservation_instruction_present=True,
                        visible_new_losses_before=visible_loss,call=call,
                        proposed_calls=proposed,rejection=row.get('rejection'),state_changed=changed,
                        requested_activity_pass=activity,new_losses_relative_initial=losses,
                        newly_failed_this_action=newly_failed,scoped_gain=activity and not losses,
                        request=fingerprint(request_path),raw_output=fingerprint(rd/'model/raw_output.txt'))
            all_steps.append(step)
        assert checks(previous)==checks(result['final_audit']['review'])
        assert previous['requested_activity_status']==cell['activity_status']
        cells.append(dict(id=cell['id'],arm=cell['arm'],condition=cell['condition'],seed=cell['seed'],
                          turns=len(result['rounds']),feedback_verified=feedback_ok,
                          first_new_loss_turn=first_loss,first_executed_removal_turn=first_remove,
                          native_removals=native_removals,parsed_removal_attempts=removal_attempts,
                          turns_with_prior_new_loss_visible=loss_visible_count,
                          repeated_executed_call_payloads=duplicates,
                          activity_pass_prefixes=activity_prefixes,
                          scoped_gain_prefixes=good_prefixes))
    assert len(all_steps)==72
    by_arm={}
    for arm in accepted['aggregate']:
        subset=[c for c in cells if c['arm']==arm]
        by_arm[arm]=dict(cells=len(subset),cells_with_executed_removal=sum(c['native_removals']>0 for c in subset),
                         native_removals=sum(c['native_removals'] for c in subset),
                         feedback_verified=sum(c['feedback_verified'] for c in subset),
                         turns_with_prior_new_loss_visible=sum(c['turns_with_prior_new_loss_visible'] for c in subset),
                         any_scoped_gain_prefix=sum(bool(c['scoped_gain_prefixes']) for c in subset),
                         any_activity_pass_prefix=sum(bool(c['activity_pass_prefixes']) for c in subset))
    # Audit actual materialized demonstrations without decoding tensors or running training.
    stop=args.public.parent/'P920-STOPDATA-v1_20261107T102628Z'
    encoding=read(stop/'ENCODING.json'); manifest.append(fingerprint(stop/'ENCODING.json'))
    base_encoding=Path(encoding['base_encoding']['path']); assert fingerprint(base_encoding)['sha256']==encoding['base_encoding']['sha256']
    base_dataset_record=read(base_encoding)['dataset']; dataset_path=Path(base_dataset_record['path']); assert fingerprint(dataset_path)['sha256']==base_dataset_record['sha256']
    dataset=read(dataset_path); terminal_path=Path(encoding['terminal_dataset']['path']); assert fingerprint(terminal_path)['sha256']==encoding['terminal_dataset']['sha256']
    terminal=read(terminal_path); manifest.extend(map(fingerprint,[base_encoding,dataset_path,terminal_path]))
    rows=dataset['rows']
    tool_counts=collections.Counter(cmd['name'] for row in rows for cmd in row['command'])
    coverage=[dict(id=r['id'],family_id=r['family_id'],commands=r['command'],
                   has_decision=bool(r.get('decision_target')),decision=r.get('decision_target')) for r in rows]
    for row, item in zip(rows, coverage):
        input_path = Path(row['execution_inputs']['path'])
        record = fingerprint(input_path)
        assert record['sha256'] == row['execution_inputs']['sha256']
        manifest.append(record)
        context = read(input_path)['context']
        item['pre_action_condition_evidence'] = {
            k: context[k] for k in ('design_request', 'request', 'current_activity_request')
            if k in context}
        item['execution_inputs'] = record
    output=dict(status='saved_trajectory_analysis_complete',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                model_calls=0,training_updates=0,scene_edits=0,source_run=args.public.name,
                aggregate=by_arm,cells=cells,steps=all_steps,
                training=dict(tool_demonstrations=len(rows),tool_counts=dict(tool_counts),
                              terminal_controls=len(terminal['rows']),decision_rows=sum(r['has_decision'] for r in coverage),
                              coverage=coverage,terminal_ids=[r['id'] for r in terminal['rows']]),
                limits=['Post-hoc analysis of one exposed development family, not a new trial.',
                        'Best observed prefix is descriptive only; no new policy or cherry-picked rerun.',
                        'Present feedback proves availability, not model attention or understanding.',
                        'Training content coverage does not establish training fit or causal explanation.'],sources=manifest)
    (args.out/'TRAJECTORY_ANALYSIS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['status','aggregate']},ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in output['training'].items() if k!='coverage'},ensure_ascii=False))


if __name__=='__main__':
    main()
