"""CPU regression using actual stored proposals, rejections and native receipts.

No model generation, native execution, new scene observation or training occurs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from edit_request import from_recorded_request, validate_action, dumps
from live_edit_request import CONTRACT, build_live_request, exchange
from shared_action_contract import context_from_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recorded-rollout', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    print('Check prospective live boundary using recorded native success and rejection; zero model calls.', flush=True)
    sources = {}
    def read(path, raw=False):
        content = path.read_bytes()
        sources[str(path)] = hashlib.sha256(content).hexdigest()
        return content.decode() if raw else json.loads(content)
    result = read(a.recorded_rollout / 'RESULT.json')
    rows = copy.deepcopy(result['rounds'][:3])
    for row in rows:
        row['raw_output'] = read(a.recorded_rollout / f"round_{row['turn']:03d}/model/raw_output.txt", True)
    assert 'reply' in rows[0] and 'reply' in rows[1] and 'rejection' in rows[2]
    original = read(a.recorded_rollout / 'round_003/request.json')
    original_copy = copy.deepcopy(original)
    policy = dict(name=CONTRACT, max_recent_turns=2, max_history_chars=60000)
    request, meta = build_live_request(original, rows, policy)
    checks = []
    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    baseline = from_recorded_request(original)
    check('common builder preserves current state, task, schema, five views and current measured feedback',
          request[0] == baseline[0] and request[1]['content'][:-1] == baseline[1]['content'])
    check('source request not mutated', original == original_copy)
    check('whole recent native/rejected exchange selection explicit',
          meta['included_turns'] == [1, 2] and meta['omitted_older_attempts'] == 1
          and not meta['individual_exchanges_truncated'])
    history = json.loads(request[1]['content'][-1]['text'].split('\n')[-1])
    check('native outcome preserved verbatim, not replaced by claimed success',
          history['exchanges'][0]['outcome']['receipt'] == rows[1]['reply'])
    check('rejected raw proposal and real error preserved, never marked executed',
          history['exchanges'][1]['assistant_proposal'] == rows[2]['raw_output']
          and history['exchanges'][1]['outcome']['message'] == rows[2]['rejection']
          and history['exchanges'][1]['outcome']['native_call_made'] is False)
    poisoned = copy.deepcopy(rows)
    for row in poisoned:
        row['gold_action'] = 'DO_NOT_EXPOSE_REFERENCE'
        row['reference_design_reason'] = 'DO_NOT_EXPOSE_REFERENCE'
    check('unobserved target/reference fields cannot enter history',
          build_live_request(original, poisoned, policy) == (request, meta))
    new_limit = len(dumps([exchange(rows[2])]))
    _, bounded = build_live_request(original, rows, {**policy, 'max_history_chars': new_limit})
    check('character budget evicts complete older exchange only', bounded['included_turns'] == [2])
    rejects('oversized latest exchange fails rather than silently truncate',
            lambda: build_live_request(original, rows, {**policy, 'max_history_chars': new_limit - 1}))
    rejects('missing turn in history rejected', lambda: build_live_request(original, rows[1:], policy))
    contradictory = copy.deepcopy(rows)
    contradictory[2]['reply'] = {'results': []}
    rejects('ambiguous rejected-versus-executed outcome rejected',
            lambda: build_live_request(original, contradictory, policy))
    initial = read(a.recorded_rollout / 'round_000/request.json')
    _, empty = build_live_request(initial, [], policy)
    check('initial step explicitly records zero prior attempts',
          empty['total_prior_attempts'] == 0 and empty['included_turns'] == [])
    ctx = context_from_request(request)
    removed_id = rows[0]['call']['arguments']['object_id']
    check('current state retains actual deletion despite omission of old exchange', removed_id not in ctx['objects'])
    rejects('action on actually removed object rejected at advertised boundary',
            lambda: validate_action(rows[0]['call'], request))
    # Import/CLI smoke check of the actual new entry point. This does not exercise
    # session/model execution and must not be reported as a native rollout test.
    runner = Path(__file__).with_name('run_edit_policy_live.py')
    cli = subprocess.run([sys.executable, str(runner), '--help'], capture_output=True, text=True, timeout=30)
    check('actual prospective runner imports and exposes CLI', cli.returncode == 0)
    (a.out / 'cli.log').write_text(cli.stdout + cli.stderr)
    (a.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    report = dict(passed=True, checks=checks, recorded_source_sha256=sources,
                  included_history=meta, source_rows=3, new_model_calls=0,
                  native_calls=0, training_updates=0, processor_encoding_performed=False,
                  live_rollout_executed=False,
                  limitation='Recorded fixtures and live runner import only; no evidence of model, execution, design or learning benefit.')
    (a.out / 'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'PASS {len(checks)} checks; no model, native edits, processor encoding or training.', flush=True)


if __name__ == '__main__':
    main()
