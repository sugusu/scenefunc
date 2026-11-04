"""Freeze a bounded tool-contract intervention derived from unchanged v1 inputs."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

from audit_tool_contract import dispatched_tools

SOURCE_RUN = 'P920-DD-v1_20261104T083214Z'
LABELS = ('A_G', 'A_U', 'B_G', 'B_U')
UNAVAILABLE = ('snap_to_object_tool', 'rescale_furniture_tool', 'check_physics')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def corrected_system(original):
    system = original
    for name in UNAVAILABLE:
        pattern = r'(?m)^- ' + name + r'\([^\n]*(?:\n[ \t]+[^\n]*)*\n?'
        system, count = re.subn(pattern, '', system)
        if count != 1:
            raise ValueError(f'Expected exactly one definition for {name}, found {count}')
    replacements = {
        '3. Snap each piece to its wall / target furniture':
            '3. Align furniture using absolute move/rotate coordinates',
        '4. For corrections use move / remove / rescale':
            '4. For corrections use move / remove',
        '- Furniture with doors/drawers (dressers, cabinets, beds, nightstands, desks, refrigerators) must use orientation="away" when snapped to a wall':
            '- Furniture with doors/drawers (dressers, cabinets, beds, nightstands, desks, refrigerators) must use yaw so the functional front faces away from the wall',
        '(use these IDs when calling move / remove / rescale / snap)':
            '(use these IDs when calling move / remove)',
        '- `snap_to_object_tool.orientation="away"` to a wall is the correct idiom for storage furniture (wardrobes, dressers, cabinets, beds, nightstands, desks). Using "toward" makes the functional front face the wall and breaks usage.':
            '- For storage furniture (wardrobes, dressers, cabinets, beds, nightstands, desks), set yaw so the functional front faces away from the wall. Facing the wall breaks usage.',
        '# Validation\n\n': '',
    }
    for old, new in replacements.items():
        if system.count(old) != 1:
            raise ValueError(f'Unexpected source instruction count: {old[:70]}')
        system = system.replace(old, new)
    if any(name in system for name in UNAVAILABLE):
        raise ValueError('Unavailable tool name remains in the corrected system prompt')
    return system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'P920-TC-v1_\d{8}T\d{6}Z', args.run_id):
        raise ValueError('Use a distinct timestamped P920-TC-v1 run ID')
    root = args.root.resolve()
    source = root / 'runtime' / SOURCE_RUN
    run = root / 'runtime' / args.run_id
    pub = root / 'scenefunc/runs/920' / args.run_id
    run.mkdir(exist_ok=False)
    pub.mkdir(exist_ok=False)
    (run / 'work').symlink_to(source / 'work', target_is_directory=True)
    (run / 'drafts').symlink_to(source / 'drafts', target_is_directory=True)
    (run / 'execution').mkdir()
    out = run / 'comparison_inputs'
    out.mkdir()
    dump(run / 'previous_root_status.json', read(root / 'STATUS.json'))
    status = {
        'run_id': args.run_id, 'handoff_id': 'P920-TC-v1', 'stage': 'preparing_inputs',
        'status': 'preflight', 'updated_at': now(), 'run_directory': str(run),
        'question': 'Does removing unavailable tool declarations and their usage '
                    'instructions change tool validity or actual functional outcomes '
                    'under the same frozen G/U development inputs?',
        'completed_planned': {'formal_revision_rounds': {'completed': 0, 'planned': 4}},
        'configuration_hashes': {}, 'failure_reasons': [], 'process': None,
        'latest_verifiable_artifacts': [], 'next_step': 'CPU encode and freeze all four inputs.',
        'source_run_id': SOURCE_RUN, 'not_a_training_or_generalization_experiment': True,
    }
    dump(pub / 'STATUS.json', status)
    dump(root / 'STATUS.json', status)
    dump(pub.parent / 'latest.json', {k: status[k] for k in
         ['run_id', 'stage', 'status', 'updated_at', 'run_directory']})
    dispatch = source / 'work/paris_local_run_results_20261104/dispatch_corrected.py'
    supported = dispatched_tools(dispatch.read_text(encoding='utf-8'))
    schema_path = source / 'work/professional_data_ingest_20261104/native_src/stage_schemas.py'
    spec = importlib.util.spec_from_file_location('schema', schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    sys.path.insert(0, str(source / 'work/native_local_inference_20261104'))
    from local_vlm import LocalQwenEngine
    import torch
    model = read(root / 'MODEL_DOWNLOAD_PLAN.json')['destination']
    engine = LocalQwenEngine(model, out / 'encode', max_pixels=65536,
                            max_input_tokens=8192, max_new_tokens=2048)
    records = []
    for label in LABELS:
        old = source / 'comparison_inputs' / label
        directory = out / label
        directory.mkdir()
        original = read(old / 'request.json')
        messages = copy.deepcopy(original)
        messages[0]['content'] = corrected_system(original[0]['content'])
        declared = set(re.findall(r'^- ([A-Za-z_][A-Za-z_0-9]*)\(', messages[0]['content'], re.M))
        assert declared == supported == set(schema.FURNITURE_SCHEMAS) - set(UNAVAILABLE)
        for match in re.finditer(r'^- (\w+)\(([^\n]*)\) ->', messages[0]['content'], re.M):
            fields = re.findall(r'(\w+):', match.group(2))
            native = schema.FURNITURE_SCHEMAS[match.group(1)]
            assert set(fields) == set(native['required']) | set(native['optional'])
        assert messages[1:] == original[1:]
        dump(directory / 'request.json', messages)
        shutil.copyfile(old / 'feedback.json', directory / 'feedback.json')
        tensors, encoded = engine.encode(messages)
        torch.save(dict(tensors), directory / 'inputs.pt')
        old_record = read(old / 'receipt.json')
        assert encoded['images'] == old_record['encoding']['images']
        assert encoded['input_tokens'] <= old_record['input_tokens'] <= 8192
        for item in encoded['images']:
            assert sha(Path(item['origin'])) == item['sha256']
        checkpoint = source / 'drafts' / label[0] / 'checkpoint.json'
        assert sha(checkpoint) == old_record['checkpoint_sha256']
        record = {'label': label, 'arm': label[0], 'method': label[-1],
                  'input_tokens': encoded['input_tokens'], 'encoding': encoded,
                  'request_sha256': sha(directory / 'request.json'),
                  'input_tensor_sha256': sha(directory / 'inputs.pt'),
                  'feedback_sha256': sha(directory / 'feedback.json'),
                  'source_request_sha256': sha(old / 'request.json'),
                  'checkpoint_sha256': old_record['checkpoint_sha256'],
                  'scene_sha256': old_record['scene_sha256'],
                  'user_messages_unchanged': True, 'views_unchanged': True,
                  'executable_tool_names': sorted(supported)}
        dump(directory / 'receipt.json', record)
        records.append(record)
    assert not torch.cuda.is_initialized()
    for arm in ('A', 'B'):
        g, u = [read(out / (arm + '_' + method) / 'request.json') for method in ('G', 'U')]
        assert g[0] == u[0] and g[1]['content'] == u[1]['content'][:-1]
    scripts = Path(__file__).resolve().parent
    names = ['prepare_contract_diagnostic.py', 'comparison_model.py', 'run_formal_models.py',
             'apply_revision.py', 'process_revisions.py', 'score_comparison.py',
             'measure_scene.py', 'render_with_blender.py', 'audit_tool_contract.py']
    protocol = {
        'status': 'frozen_before_any_new_model_call', 'frozen_at': now(),
        'run_id': args.run_id, 'source_run_id': SOURCE_RUN, 'records': records,
        'intervention': 'Remove three unavailable tool declarations and their usage '
                        'instructions only; unchanged user messages, feedback, views, '
                        'drafts, assets, parser, dispatcher, scorer and model settings.',
        'removed_tools': list(UNAVAILABLE), 'executable_tools': sorted(supported),
        'dispatcher_sha256': sha(dispatch), 'native_schema_sha256': sha(schema_path),
        'script_sha256': {name: sha(scripts / name) for name in names},
        'model_revision': '63e235a0c0d6e5bc2f8f3485214df4457fd2e672',
        'probe_label': max(records, key=lambda x: x['input_tokens'])['label'],
        'max_input_tokens': 8192, 'max_new_tokens': 2048, 'max_pixels_per_image': 65536,
        'seed': 0, 'batch_size': 1, 'precision': 'float16', 'attention': 'native torch SDPA',
        'use_cache': True, 'max_tool_calls': 12, 'model_timeout_seconds': 1200,
        'formal_model_calls_planned': 4, 'engineering_probe_calls_planned': 1,
        'engineering_probe_output_tokens': 32, 'retries': 0, 'draft_model_calls_planned': 0,
        'failure_policy': 'Retain every attempt; no retry or extra continuation. '
                          'Over-budget/truncated/failed programs retain the starting draft.',
        'primary_endpoint': 'Per-arm unavailable-tool emission, plus total rejected calls.',
        'secondary_endpoints': ['actual executed actions', 'deleted required furniture',
                                'mesh surface witnesses', 'collision pairs', 'cost'],
        'comparison': 'Descriptive paired comparison with saved v1 outcomes; '
                      'one development room, four arms, greedy seed 0. No significance claim.',
        'remaining_limitations': ['Inherited image-annotation descriptions still conflict '
                                 'with the later unannotated-view caveat; unchanged in this intervention.',
                                 'One-turn bounded controller, not full interactive Scene-Designer.',
                                 'Whole-room usability remains unknown; no training or held-out evidence.'],
    }
    dump(out / 'PROTOCOL.json', protocol)
    dump(pub / 'COMPARISON_PROTOCOL.json', protocol)
    status.update(stage='inputs_frozen', updated_at=now(),
                  configuration_hashes={'comparison_protocol': sha(out / 'PROTOCOL.json')},
                  latest_verifiable_artifacts=['COMPARISON_PROTOCOL.json'],
                  next_step='Guarded longest-input 32-token engineering probe, then exactly four formal attempts.')
    dump(pub / 'STATUS.json', status)
    dump(root / 'STATUS.json', status)
    dump(pub.parent / 'latest.json', {k: status[k] for k in
         ['run_id', 'stage', 'status', 'updated_at', 'run_directory']})
    print(json.dumps({'run_id': args.run_id, 'tokens': {x['label']: x['input_tokens'] for x in records},
                      'probe_label': protocol['probe_label'], 'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
