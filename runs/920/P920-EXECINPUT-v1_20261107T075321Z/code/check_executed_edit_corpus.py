"""Pre-action isolation, temporal conditions and rejected-call regression checks."""
import argparse
import copy
import json
from pathlib import Path

from edit_request import dumps, validate_action
from executed_edit_request import build_executed_request
from shared_action_contract import context_from_request


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    old = a.run/'corpus'; new = a.run/'corpus_phase'
    data = json.loads((new/'DATASET.json').read_text()); checks = []
    def check(name, ok):
        if not ok: raise AssertionError(name)
        checks.append(name)
    def rejects(name, inputs):
        try: build_executed_request(**inputs)
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    events = json.loads((a.run/'CURRENT_ACTIVITY_EVENTS.json').read_text())['events']
    for row in data['rows']:
        identifier = row['id']
        before = json.loads((old/'inputs'/f'{identifier}.json').read_text())
        after = json.loads((new/'inputs'/f'{identifier}.json').read_text())
        clean = copy.deepcopy(after); event = clean['context'].pop('current_activity_request', None)
        check(identifier+':only_explicit_activity_changed', before == clean)
        check(identifier+':condition_matches_declared_schedule', event == events.get(identifier))
        request, _ = build_executed_request(**after)
        ctx = context_from_request(request)
        check(identifier+':current_objects_preserved', ctx['objects'] == context_from_request(build_executed_request(**before)[0])['objects'])
        for call in row['command']: validate_action(call, request)
        if event:
            text = event['request']
            check(identifier+':no_object_asset_id_in_activity', not any(oid in text for oid in ctx['objects']) and
                  not any(asset['asset_id'] in text for asset in ctx.get('asset_catalog', [])))
        check(identifier+':shared_builder_reconstructs_saved_request', request == json.loads((new/'requests'/f'{identifier}.json').read_text()))
    args = json.loads((new/'inputs/storage_edit_03.json').read_text()); original = copy.deepcopy(args)
    request, meta = build_executed_request(**args)
    check('builder_does_not_mutate_inputs', original == args)
    check('latest_two_attempts_preserved', meta['included_attempts'] == [1, 2])
    current = copy.deepcopy(args); current['history'][-1]['attempt'] = current['next_attempt']
    rejects('current_or_future_receipt_rejected', current)
    unknown = copy.deepcopy(args); unknown['history'][-1]['gold_decision'] = 'TARGET_SENTINEL'
    rejects('unknown_history_or_target_fields_rejected', unknown)
    discontinuous = copy.deepcopy(args); discontinuous['history'][-1]['before_sha256'] = '0'*64
    rejects('discontinuous_trace_rejected', discontinuous)
    rejected = copy.deepcopy(args)
    r = rejected['history'][-1]; r['origin'] = 'model_execution'
    r['outcome'] = dict(kind='validation_rejection', message='Tool is not advertised.', native_call_made=False)
    r['after_sha256'] = r['before_sha256']
    req, _ = build_executed_request(**rejected)
    check('rejection_remains_not_executed', '"native_call_made":false' in req[1]['content'][-1]['text'])
    bad = copy.deepcopy(rejected); bad['history'][-1]['outcome']['native_call_made'] = True
    rejects('rejection_cannot_claim_native_execution', bad)
    oversized = copy.deepcopy(rejected); oversized['history'][-1]['outcome']['message'] = 'x'*13000
    rejects('oversized_latest_exchange_not_truncated', oversized)
    result = dict(status='passed', checks=checks, changed_activity_inputs=len(events),
        source_actions=22, documents=44, new_model_calls=0, native_calls=0, optimizer_updates=0,
        caveat='CPU contract fixtures only; no live model or new native execution.')
    with a.out.open('x') as f: f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='passed', checks=len(checks), changed_activity_inputs=len(events))), flush=True)


if __name__ == '__main__': main()
