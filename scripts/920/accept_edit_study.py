"""Read-only acceptance of the completed bounded study; no model or scene edits."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Preserve prior acceptance; do not overwrite')
    p = a.study.resolve()
    batch = read(p/'EVALUATION_BATCH.json')
    summary = read(p/'reports/final/SUMMARY.json')
    training = read(p/'TRAINING_RESULT.json')
    assert batch['exit_code'] == 0 and len(batch['cells']) == 12
    assert training['status'] == 'completed' and training['exit_code'] == 0
    assert summary['terminal_cells'] == summary['cells_with_final_audit'] == 12
    matrix = read(p/'EVALUATION_COMMAND_MATRIX.json')['cells']
    assert {c['cell_id'] for c in batch['cells']} == {c['cell'] for c in matrix}
    assert len({(c['task'], c['arm'], c['seed']) for c in batch['cells']}) == 12
    for path, digest in read(p/'EVALUATION_EXECUTION_LOCK.json')['files'].items():
        assert sha(path) == digest, path
    train_seconds = 0.
    for arm, entry in training['arms'].items():
        r = entry['training']
        assert r['optimizer_updates'] == 8 and r['backward_completed'] == 176
        assert r['same_process_adapter_save_zero_reload_exact']
        assert entry['receipt']['exit_code'] == 0
        assert entry['receipt']['elapsed_seconds'] <= 7200
        train_seconds += entry['receipt']['elapsed_seconds']
    groups = defaultdict(lambda: dict(rooms=0, model_calls=0, native_calls=0,
        rejected_calls=0, explicit_think_outputs=0, generated_tokens=0,
        gpu_subprocess_seconds=0., gained_activity_rooms=0, preserved_activity_rooms=0,
        new_geometry_or_identity_failure_rooms=0, strict_proxy_pass_rooms=0))
    cells = []
    for cell in batch['cells']:
        folder = Path(cell['result_path']).parent
        public = p/'evaluation'/cell['cell_id']
        r = read(folder/'RESULT.json')
        plan = read(public/'PLAN.json')
        assert sha(public/'PLAN.json') == cell['plan_sha256']
        assert cell['exit_code'] == cell['receipt']['exit_code'] == 0
        assert r['status'] == 'policy_completed_pending_comparison'
        assert r['model_calls'] <= plan['budget']['model_calls']
        assert r['tool_attempts'] <= plan['budget']['tool_calls']
        assert r['gpu_subprocess_seconds'] <= plan['budget']['gpu_seconds_per_cell']
        assert r['generated_tokens'] <= plan['budget']['model_calls'] * plan['budget']['max_new_tokens']
        assert not r.get('finalization_error')
        bundle = Path(cell['final_checkpoint']['directory'])
        assert sha(bundle/'manifest.json') == cell['final_checkpoint']['manifest_sha256']
        manifest = read(bundle/'manifest.json')
        for name, info in manifest['files'].items():
            f = bundle/name
            assert f.stat().st_size == info['bytes'] and sha(f) == info['sha256'], str(f)
        glb = cell['export']['glb']
        assert sha(glb['path']) == glb['sha256']
        assert Path(glb['path']).stat().st_size == glb['bytes']
        observation = read(folder/'final_observation.json')
        assert len(observation['image_paths']) == 5
        for i, path in enumerate(observation['image_paths']):
            assert sha(path) == sha(public/'recorded/views'/f'{i}_{Path(path).name}')
        for name in ('RESULT.json', 'final_state.json', 'final_audit.json', 'final_observation.json'):
            assert sha(folder/name) == sha(public/'recorded'/name)
        models = sorted(folder.glob('round_*/model/RESULT.json'))
        assert len(models) == r['model_calls']
        think_count = 0
        for file in models:
            m = read(file)
            assert m['status'] == 'completed' and m['training_updates'] == 0
            assert len(m['encoding']['images']) == 5
            assert m['encoding']['input_tokens'] <= plan['budget']['max_input_tokens']
            assert m['output_tokens'] <= plan['budget']['max_new_tokens']
            if cell['arm'] != 'frozen_base_knowledge_feedback':
                assert m['adapter_reload_exact'] is True
            raw = file.with_name('raw_output.txt')
            assert sha(raw) == m['raw_output_sha256']
            think_count += int('<think>' in raw.read_text())
        row = next(x for x in summary['rows'] if x['cell'] == cell['cell_id'])
        s = next(x for x in row['comparison']['scenarios'] if x['width_m'] == .5)
        g = groups[cell['arm']]
        g['rooms'] += 1
        for field in ('model_calls', 'native_calls', 'generated_tokens', 'gpu_subprocess_seconds'):
            g[field] += r[field]
        g['rejected_calls'] += row['schema_or_output_rejections']
        g['explicit_think_outputs'] += think_count
        g['gained_activity_rooms'] += s['activity_transition'] == 'gained'
        g['preserved_activity_rooms'] += s['activity_transition'] == 'preserved'
        g['new_geometry_or_identity_failure_rooms'] += s['new_geometry_or_identity_failure']
        g['strict_proxy_pass_rooms'] += s['proxy_satisfied_without_new_declared_failure']
        cells.append(dict(cell_id=cell['cell_id'],result_sha256=sha(folder/'RESULT.json'),
                          checkpoint_manifest_sha256=sha(bundle/'manifest.json'),glb=glb))
    assert batch['gpu_subprocess_seconds'] <= 7200
    rays = []
    for command in read(p/'VISUAL_CONTINUITY_PROTOCOL.json')['execution_commands']:
        public = p/'evaluation'/command['cell']
        assert read(public/'VISUAL_CONTINUITY_RECEIPT.json')['exit_code'] == 0
        result = read(public/'VISUAL_CONTINUITY_RESULT.json')
        rays.append(dict(cell_id=command['cell'],seats=[{k:r[k] for k in
            ('table_id','chair_id','clear_rays','total_rays','chair_toward_sofa_cos')}
            for r in result['workspace_pairs']]))
    out = dict(status='bounded_execution_and_artifacts_accepted_no_design_gain',
        recorded_at=datetime.now(timezone.utc).isoformat(),run_id=p.name,
        actual_training_started_at=training['started_at'],actual_training_ended_at=training['ended_at'],
        actual_evaluation_ended_at=batch['ended_at'],actual_batch_exit_code=batch['exit_code'],
        completed_training_updates=16,completed_rooms=12,training_stage_seconds=train_seconds,
        evaluation_model_subprocess_seconds=batch['gpu_subprocess_seconds'],
        aggregates=dict(groups),cells=cells,milan_supplemental_rays=rays,
        final_summary_sha256=sha(p/'reports/final/SUMMARY.json'),
        limits=['Execution acceptance is not professional room acceptance.',
                'Two exposed projects and two seeds: no independent generalization claim.',
                'Static proxy preservation in Milan is not newly acquired utility.',
                'Training losses do not establish adequate final common-input fitting.',
                'No optimization, model forward or scene edit in this acceptance.'])
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],aggregates=out['aggregates']),indent=2))


if __name__ == '__main__':
    main()
