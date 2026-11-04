"""Read-only publication checks on the completed feedback batch; no model/replay."""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from native_project import read_json, sha256, validate_bundle
from native_checkpoint import canonical_hash


def main():
    repo = Path(__file__).resolve().parents[2]
    public = repo / 'runs/920/P920-FEEDBACK-v2_20261104T200101Z'
    failed = repo / 'runs/920/P920-FEEDBACK-v1_20261104T195244Z'
    runtime = repo.parent / 'runtime' / public.name
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    checks = {}
    paths = sorted(set(subprocess.check_output(
        ['git', 'ls-files', '--modified', '--others', '--exclude-standard', '-z'],
        cwd=repo).decode().strip('\0').split('\0')))
    patterns = [rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
                rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}',
                rb'hf_[A-Za-z0-9]{30,}', rb'sk-[A-Za-z0-9_-]{35,}']
    counts = {'json': 0, 'python': 0, 'files': len(paths)}
    for relative in paths:
        f = repo / relative
        assert f.is_file(), relative
        data = f.read_bytes()
        assert len(data) <= 5 * 1024**2, 'Large Git artifact: ' + relative
        assert not any(re.search(pattern, data) for pattern in patterns), 'Credential signature: ' + relative
        if f.suffix == '.json':
            read_json(f)
            counts['json'] += 1
        if f.suffix == '.py':
            ast.parse(data, filename=relative)
            counts['python'] += 1
    checks['public_syntax_size_and_credential_signature_scan'] = counts
    historical = [x for x in paths if x.startswith('paris/') or
                  (x.startswith('runs/') and not x.startswith(
                      ('runs/920/' + public.name + '/', 'runs/920/' + failed.name + '/'))
                   and x != 'runs/920/latest.json')]
    assert not historical, historical
    stable = ['native_project.py', 'native_checkpoint.py', 'existing_project_session.py', 'catalog_project_session.py']
    for name in stable:
        relative = 'scripts/920/' + name
        assert subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=repo) == (repo / relative).read_bytes()
    checks['historical_runs_and_four_native_helpers_unchanged'] = True
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo).decode().strip() == 'runs/920'
    checks['current_branch'] = 'runs/920'
    indexes = [failed / 'SERVER_ARTIFACTS.json', public / 'SERVER_ARTIFACTS.json',
               repo / 'research/920/public_sources_v1/dignified_2025/SERVER_ARTIFACTS.json']
    for index in indexes:
        records = read_json(index)['files']
        for entry in records:
            f = Path(entry['path'])
            assert f.stat().st_size == entry['bytes'] and sha256(f) == entry['sha256'], str(f)
        total = sum(e['bytes'] for e in records)
        assert read_json(index).get('total_bytes', total) == total
        checks[str(index.relative_to(repo))] = {'files': len(records), 'bytes': total}
    code_count = 0
    for root in (failed, public):
        for planfile in root.rglob('*PLAN.json'):
            plan = read_json(planfile)
            archive = failed / 'code' if planfile.name == 'PACKING_PLAN.json' else root / 'code'
            for name, expected in plan.get('code_sha256', {}).items():
                assert sha256(archive / name) == expected, str(planfile) + ':' + name
                code_count += 1
    checks['frozen_code_hashes'] = code_count
    bundle_count = resource_count = image_count = 0
    protocol = read_json(public / 'RUN_PROTOCOL.json')
    ledger = read_json(public / 'COST_LEDGER.json')
    comparison = read_json(public / 'COMPARISON.json')
    actual_requests = {}
    for arm in ('G', 'U'):
        base = runtime / arm
        assert sha256(public / arm / 'POLICY_PLAN.json') == protocol['arms'][arm]['policy_plan_sha256']
        bundles = [base / 'prepare/checkpoint', *sorted((base / 'rollout').glob('round_*/checkpoint')),
                   base / 'rollout/final_checkpoint']
        assert len(bundles) == 11
        for bundle in bundles:
            manifest, *_ = validate_bundle(bundle)
            bundle_count += 1
            resource_count += len(manifest['files'])
        result = read_json(base / 'rollout/RESULT.json')
        assert result['model_calls'] == result['tool_calls'] == len(result['rounds']) == 9
        assert result['cumulative_lineage_tool_calls'] == 24 and result['training_updates'] == 0
        assert ledger['arms'][arm]['input_tokens'] == sum(row['model']['encoding']['input_tokens'] for row in result['rounds'])
        assert ledger['arms'][arm]['output_tokens'] == sum(row['model']['output_tokens'] for row in result['rounds'])
        for row in result['rounds']:
            rounddir = base / 'rollout' / f"round_{row['turn']:03d}"
            assert sha256(rounddir / 'request.json') == row['model']['request_sha256']
            observation = read_json(rounddir / 'observation.json')
            binding = read_json(Path(observation['state_binding']))
            manifest = read_json(rounddir / 'checkpoint/manifest.json')
            house = read_json(rounddir / 'checkpoint/house_state.json')
            assert binding['native_state'] == house['rooms'][manifest['selected_room_id']]
            assert canonical_hash(binding['native_state']) == binding['state_sha256'] == row['after_sha256']
            assert len(binding['images']) == 5
            for image in binding['images']:
                assert sha256(Path(image['path'])) == image['sha256']
                image_count += 1
        terminal = read_json(public / arm / 'TERMINAL_RESTORE_RESULT.json')
        assert len(terminal['checks']) == 10 and all(terminal['checks'].values())
        assert terminal['model_calls'] == terminal['tool_calls'] == terminal['scene_edits'] == 0
        assert terminal['input_manifest_sha256'] == sha256(base / 'rollout/final_checkpoint/manifest.json')
        assert read_json(base / 'rollout_execution/receipt.json')['exit_code'] == 0
        actual_requests[arm] = read_json(base / 'rollout/round_000/request.json')
    normalized = copy.deepcopy(actual_requests)
    feedback = {}
    for arm, request in normalized.items():
        before, raw = request[1]['content'][-1]['text'].split('\n\n## Independent current-state feedback\n', 1)
        request[1]['content'][-1]['text'] = before
        feedback[arm] = json.loads(raw)
    assert normalized['G'] == normalized['U']
    assert feedback['G']['geometry'] == feedback['U']['geometry']
    assert 'use' not in feedback['G'] and 'use' in feedback['U']
    assert comparison['new_model_calls'] == comparison['new_tool_calls'] == 18
    assert not comparison['whole_room_accepted'] and comparison['training_updates'] == 0
    checks['native_bundles_resources_and_views'] = {'bundles': bundle_count, 'resource_file_checks': resource_count,
                                                  'bound_image_checks': image_count}
    checks['actual_first_pair_and_recorded_costs'] = True
    checks['terminal_native_restore_checks'] = 20
    diff = subprocess.run(['git', 'diff', '--check'], cwd=repo, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout
    checks['tracked_diff_check'] = True
    report = {'started_at': started, 'ended_at': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': time.monotonic() - tick, 'status': 'passed', 'checks': checks,
              'scope': 'Read-only publication validation of recorded outputs. No inference, scene replay, training or additional professional assessment.',
              'script_sha256': sha256(Path(__file__)),
              'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()}
    target = Path(sys.argv[1])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
