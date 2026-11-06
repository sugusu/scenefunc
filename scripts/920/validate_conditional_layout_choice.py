"""Read-only verification of the completed choice batch; no inference or repairs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from probe_conditional_layout_choice import assess
from report_conditional_layout_choice import literal_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    read = lambda p: json.loads(Path(p).read_text())
    verified = {}

    def check(path, digest):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f'Hash mismatch: {path}'
        verified[str(path)] = actual

    protocol = read(run / 'PROTOCOL.json')
    summary = read(run / 'SUMMARY.json')
    review = read(run / 'REVIEW.json')
    status = read(run / 'STATUS.json')
    sources = read(root / 'research/920/paper_v1/CLAIM_EVIDENCE.json')['sources']
    for ref in sources.values():
        check(ref['path'], ref['sha256'])
    for name, digest in protocol['code_sha256'].items():
        check(root / 'scripts/920' / name, digest)
    check(run / 'PROTOCOL.json', status['configuration_and_inputs_sha256']['protocol'])
    check(run / 'PROTOCOL.json', review['protocol_sha256'])
    check(run / 'PRIVATE_SCORING.json', protocol['scoring_sha256'])
    check(run / 'SUMMARY.json', review['primary_summary_sha256'])
    check(root / 'scripts/920/report_conditional_layout_choice.py', review['report_script_sha256'])
    for ref in read(run / 'PRIVATE_SCORING.json')['images']:
        check(ref['path'], ref['sha256'])
    primary = {row['id']: row for row in summary['results']}
    secondary = {row['id']: row for row in review['rows']}
    assert len(primary) == len(secondary) == len(protocol['cells']) == 12
    calls = 0
    elapsed = 0.0
    counts = {}
    for cell in protocol['cells']:
        for key in ('plan', 'request'):
            check(cell[key]['path'], cell[key]['sha256'])
        directory = Path(cell['plan']['path']).parent
        model = read(directory / 'model/RESULT.json')
        receipt = read(directory / 'receipt/receipt.json')
        rawpath = directory / 'model/raw_output.txt'
        check(rawpath, model['raw_output_sha256'])
        check(rawpath, secondary[cell['id']]['raw_output_sha256'])
        for key in ('model_result', 'raw_output'):
            ref = primary[cell['id']][key]
            check(ref['path'], ref['sha256'])
        assert model['plan_sha256'] == cell['plan']['sha256']
        assert model['request_sha256'] == cell['request']['sha256']
        assert model['model_calls'] == 1 and model['training_updates'] == 0
        assert receipt['exit_code'] == 0 and receipt['status'] == 'completed'
        assert receipt['first_log'] and receipt['started_at'] < receipt['ended_at']
        assert len(model['encoding']['images']) == 5
        for ref in model['encoding']['images']:
            check(ref['origin'], ref['sha256'])
        plan = read(cell['plan']['path'])
        assert model['encoding']['input_tokens'] <= plan['budget']['max_input_tokens']
        assert model['output_tokens'] <= plan['budget']['max_new_tokens']
        check(plan['encoder'], plan['encoder_sha256'])
        for name, digest in plan['model_configuration_sha256'].items():
            check(Path(plan['model_directory']) / name, digest)
        if plan.get('adapter'):
            assert model['adapter_reload_exact'] is True
            for name, digest in plan['adapter']['sha256'].items():
                check(Path(plan['adapter']['directory']) / name, digest)
        raw = rawpath.read_text()
        result = assess(raw, model, cell['label_map'], cell['expected_endpoint'])
        for key, value in result.items():
            assert primary[cell['id']][key] == value
        literal = literal_selection(raw)
        literal['selected_endpoint'] = cell['label_map'].get(literal.get('candidate_id'))
        literal['matches_frozen_activity'] = literal['selected_endpoint'] == cell['expected_endpoint']
        assert literal == secondary[cell['id']]['posthoc_literal']
        arm = counts.setdefault(cell['arm'], dict(calls=0, primary_correct=0, primary_valid=0, posthoc_correct=0))
        arm['calls'] += 1
        arm['primary_correct'] += result['correct']
        arm['primary_valid'] += result['valid_response']
        arm['posthoc_correct'] += literal['matches_frozen_activity']
        calls += model['model_calls']
        elapsed += receipt['elapsed_seconds']
    assert calls == summary['actual_model_calls'] == review['actual_model_calls'] == 12
    assert abs(elapsed - review['model_subprocess_seconds']) < 1e-6
    for name, values in counts.items():
        reported = review['arms'][name]
        assert values['primary_correct'] == reported['primary']['correct']
        assert values['primary_valid'] == reported['primary']['valid']
        assert values['posthoc_correct'] == reported['posthoc_literal_correct']
    receipt = read(run / 'run_receipt/receipt.json')
    assert receipt['exit_code'] == 0 and status['exit_code'] == 0
    assert status['completed'] == status['planned'] == calls
    subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), status='passed',
                  claim_sources=len(sources), unique_hashed_files=len(verified),
                  hashes=verified, actual_model_calls=calls, recomputed_counts=counts,
                  model_subprocess_seconds=elapsed, git_diff_check='passed',
                  scope='Artifact integrity and exact score/accounting recomputation only. Not an independent professional review or model-quality acceptance.',
                  not_repeated=['base weight shard hashing (prior acceptance retained)', 'model inference', 'training', 'native editing'])
    with (run / 'VALIDATION.json').open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'hashes'}))


if __name__ == '__main__':
    main()
