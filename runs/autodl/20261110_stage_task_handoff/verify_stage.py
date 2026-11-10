"""Stored native-state/input/source verification; not generation validation."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d = json.loads((ROOT / 'actual_v2/result.json').read_text())
    assert d['model_calls'] == d['training_steps'] == d['new_generated_scenes'] == 0
    assert d['network_attempts'] == 0
    assert not d['whole_runtime_ready'] and not d['learned_advantage_verified']
    assert d['native_state_before_sha256'] == d['native_state_after_sha256']
    assert d['original_files_before'] == d['original_files_after']
    for p, expected in d['original_files_before'].items():
        assert sha(p) == expected, p
    assert len(d['selection_rows']) == len(d['workflow_rows']) == 3
    first = d['selection_rows'][0]
    assert first['condition'] == 'native'
    assert len(first['rendering_params']['include_objects']) == 6
    original = d['workflow_rows'][0]['public_context']
    for i in (1, 2):
        selected = d['selection_rows'][i]
        assert selected['same_scene_instance'] and selected['stopped_before_render_and_model']
        payload = json.loads(selected['prompt_kwargs']['scene_description'])
        assert payload['derived_room_request'] == first['prompt_kwargs']['scene_description']
        assert selected['prompt_kwargs']['furniture_list'] == first['prompt_kwargs']['furniture_list']
        assert selected['rendering_params'] == first['rendering_params']
        later = d['workflow_rows'][i]['public_context']
        assert later['task_provenance'] == payload
        assert 'original_room_request' not in later
        assert later['actual_state'] == original['actual_state']
        assert later['native_selection_proposal'] == original['native_selection_proposal']
        assert hashlib.sha256(payload['original_user_request'].encode()).hexdigest() == \
            selected['task_transport_audit'][0]['original_user_request_sha256']
    assert d['workflow_transform']['context_keyword_transformations'] == 1
    assert d['workflow_transform']['method_ast_identical_after_reverting_one_change']
    assert d['missing_original_task_rejected']
    sources = json.loads((ROOT / 'source_reading.json').read_text())
    for r in sources:
        assert sha(r['path']) == r['sha256']
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    for r in manifest:
        p = REPO / r['path']
        assert sha(p) == r['sha256'] and p.stat().st_size == r['bytes']
    with zipfile.ZipFile(ROOT / 'evidence.zip') as z:
        assert z.testzip() is None
        for r in manifest:
            assert hashlib.sha256(z.read(r['path'])).hexdigest() == r['sha256']
        archive_files = len(z.namelist())
    result = {'status': 'passed', 'manifest_files': len(manifest),
              'archive_files': archive_files, 'source_files_current_sha_verified': len(sources),
              'existing_scene_state_unchanged': True, 'new_model_calls': 0,
              'scope': 'Input provenance and stored-state consistency only',
              'new_complete_scene_or_method_benefit': False}
    (ROOT / 'delivery_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
