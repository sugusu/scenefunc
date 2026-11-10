"""Verify stored input/source evidence, not a learned-quality evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO / 'scripts/autodl'))
from capture_critic_task_inputs import native_factory, sha, ADV
from critic_task_context import bind_factory_import, make_task_conditioned_reward


def main():
    capture = json.loads((ROOT / 'input_capture_v2/capture.json').read_text())
    rows = capture['rows']
    assert len(rows) == 6
    assert capture['model_calls'] == capture['training_steps'] == capture['new_generated_scenes'] == 0
    assert capture['model_attribute_accesses'] == capture['network_attempts'] == 0
    assert capture['missing_task_rejected']
    for item in capture['inputs_unchanged'] + capture['processor_files_unchanged']:
        assert sha(item['path']) == item['sha256'], item['path']
    old = capture['previous_context_probe_reused_only']
    assert sha(old['path']) == old['sha256'] and old['new_inference'] is False
    task_path = next(x['path'] for x in capture['inputs_unchanged']
                     if x['path'].endswith('paris_A_inference_20261104/request_B.json'))
    task = json.loads(Path(task_path).read_text())[1]['content'][-1]['text'].split('## Memory')[0].strip()
    for i, row in enumerate(rows):
        prompt = (ROOT / f'input_capture_v2/{row["name"]}.prompt.txt').read_text()
        formatted = (ROOT / f'input_capture_v2/{row["name"]}.formatted.txt').read_text()
        assert hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256']
        assert hashlib.sha256(formatted.encode()).hexdigest() == row['formatted_sha256']
        assert prompt in formatted
        expected_conditioned = i >= 3
        transported = task in prompt or json.dumps(task, ensure_ascii=False) in prompt
        assert transported == row['contains_original_task'] == expected_conditioned
        if expected_conditioned:
            payload = prompt.split('Original task shared by every candidate (JSON string):\n')[1].splitlines()[0]
            assert json.loads(payload) == task
        assert row['labels'] == list('ABC'[:row['image_count']])
        assert len(row['image_grid_thw']) == row['image_count']
        assert row['model_called'] is row['reward_computed'] is False
        for p, expected in zip(row['image_paths'], row['image_sha256']):
            assert sha(p) == expected
    for i in range(3):
        a, b = rows[i], rows[i+3]
        assert a['image_sha256'] == b['image_sha256']
        assert a['image_grid_thw'] == b['image_grid_thw']
        assert b['input_shapes']['input_ids'][1] > a['input_shapes']['input_ids'][1]
    assert rows[1]['image_count'] == 2 and rows[1]['legal_flags_are_fixture']
    assert rows[4]['image_count'] == 2 and rows[4]['legal_flags_are_fixture']
    factory, _, calls = native_factory()
    assert all(not c['explicit_user_text'] for c in calls) and len(calls) == 3
    spec = importlib.util.spec_from_file_location('verifier_native_advantage', ADV)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    original_ranker = native.advantage_from_critic
    original_gate = native.advantage_from_critic_with_legality
    cloned_factory = bind_factory_import(factory, native)
    assert cloned_factory.__code__ is factory.__code__
    control = make_task_conditioned_reward(factory, native, object(), object())
    assert control.context_namespace.advantage_from_critic_with_legality.__code__ is original_gate.__code__
    assert native.advantage_from_critic is original_ranker
    assert native.advantage_from_critic_with_legality is original_gate
    assert control.task_scope.get() is None
    assert not json.loads((ROOT / 'summary.json').read_text())['goal_achieved']
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    for item in manifest:
        p = REPO / item['path']
        assert p.is_file() and p.stat().st_size == item['bytes']
        assert sha(p) == item['sha256'], p
    with zipfile.ZipFile(ROOT / 'evidence.zip') as z:
        assert z.testzip() is None
        for item in manifest:
            assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256']
        archive_files = len(z.namelist())
    result = {'status': 'passed', 'scope': 'Stored source/input transport and private binding identity',
              'complete_protocol_captures': 6,
              'failed_v1_partial_prompts_retained': len(list((ROOT / 'input_capture_v1').glob('*.prompt.txt'))),
              'manifest_files_verified': len(manifest), 'archive_files_verified': archive_files,
              'source_body_code_identity_verified': True,
              'original_inputs_current_hash_match': True,
              'new_model_calls': 0, 'quality_or_training_benefit_verified': False}
    (ROOT / 'delivery_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
