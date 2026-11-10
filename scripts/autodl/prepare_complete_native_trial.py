"""Freeze one bounded baseline continuation; never train, infer or restart.

This continues the actual model's earlier first floor step, not an independent
from-empty benchmark. Shared complete-room runtime is the research objective.
"""
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    repo=Path(__file__).resolve().parents[2];stage=repo/'runs/autodl/20261110_complete_native_runtime'
    protocol=stage/'trial_protocol.json'
    if protocol.exists():raise FileExistsError('Trial inputs/budget immutable')
    result=json.loads((stage/'services_actual_v2/result.json').read_text())
    encoded=json.loads((stage/'consumer_encoding.json').read_text())
    assert result['status']=='normal_native_services_furniture_tools_and_five_views_returned'
    assert encoded['input_tokens']==1565 and not encoded['model_weights_loaded']
    assert shutil.disk_usage(stage).free>5*1024**3+16*1024**2
    # Explicit new shared rendering resolution. Original camera/annotator stay;
    # all future conditions must use this same version, no old-pixel claim.
    path=stage/'runtime/src/render_observation.py';before=path.read_text()
    if not path.is_symlink():raise ValueError('Expected immutable observation reference before new adaptation')
    path.unlink()
    marker="str(blend), '--out', str(save_dir), '--overwrite']"
    assert before.count(marker)==1
    after=before.replace(marker,"str(blend), '--out', str(save_dir), '--overwrite', '--width', '512', '--height', '384', '--top-resolution', '512']")
    path.write_text(after)
    old=repo/'runs/autodl/20261110_shared_model_runtime/protocol.json';old_protocol=json.loads(old.read_text())
    for r in old_protocol['weight_sha256_verified']:
        stat=Path(r['path']).stat();assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(r['bytes'],r['mtime_ns'],r['inode'])
    source_files=[repo/'scripts/autodl'/n for n in ('complete_native_model_host.py','run_complete_native_trial.py',
        'owned_native_retrieval.py','remote_surface_workflow.py','remote_native_control.py',
        'native_manipuland_runner.py','task_grounded_completion.py','local_manipuland_workflow.py','native_clip_runtime.py')]
    source_files += list((stage/'runtime').rglob('*.py'))+list((stage/'runtime/prompts').glob('*'))
    for directory in ('Designer','tools','src_critic'):
        source_files += list((stage/'runtime'/directory).rglob('*.py'))
    native_root=repo/'runs/autodl/20261110_remote_native_controller/runtime/native_source'
    source_files += list(native_root.rglob('*.py'))+list((native_root/'configurations').rglob('*.yaml'))
    source_files += [repo/'runs/autodl/20261110_sdk_asset_qualification/native_data_v2/summary.json',
        Path('/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/vendor/hsm/data/preprocessed/clip_hssd_embeddings_index.yaml'),
        repo/'scripts/autodl/export_complete_trial_glb.py']
    first=repo/'runs/autodl/20261110_remote_native_controller/native_joint_actual_v2'
    actual=json.loads((first/'native_policy_proposal.json').read_text())
    tool=json.loads((first/'native_tool_results.json').read_text())
    assert len(actual['tool_calls'])==1 and actual['tool_calls'][0]['name']=='generate_room_specs'
    assert tool[0]['output']['success'] is True
    value=dict(status='frozen_before_trial',checkpoint=old_protocol['checkpoint'],
        max_model_forwards=32,max_total_output_tokens=16384,per_forward_output_cap=1024,
        actor_seconds=900,parent_seconds=1200,max_tool_calls=64,
        stage_policy_turns=dict(floor_plan=8,furniture=10,wall=3,ceiling=3),
        surface_steps_per_parent=2,surface_tool_cap=24,surface_seconds=300,
        initial_floor_state=str(first/'layout_after.json'),initial_floor_state_sha256=sha(first/'layout_after.json'),
        initial_floor_proposal=str(first/'native_policy_proposal.json'),initial_floor_proposal_sha256=sha(first/'native_policy_proposal.json'),
        initial_floor_tool_results=str(first/'native_tool_results.json'),initial_floor_tool_results_sha256=sha(first/'native_tool_results.json'),
        initial_generation_cost='Previously recorded 1 forward/141 output tokens; add to this continuation when reporting total generation',
        task='A bedroom with a bed, two nightstands, and a wardrobe in the corner of the room.',
        scope='One native joint baseline continuation across floor, furniture, wall, ceiling and common surface stage; no trained-method comparison',
        resolution_change=dict(before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=sha(path),
            width=512,height=384,top_resolution=512,samples=8,threads=2,original_pixels_equivalent=False),
        native_caller_feedback_attempts_share_budget=True,host_automatic_retries=False,
        previous_closed_phase_and_joint_budgets_reopened=False,new_training_steps=0,paid_API_calls=0,
        code_sha256={str(p.resolve()):sha(p) for p in source_files},
        weight_sha256_verified=old_protocol['weight_sha256_verified'])
    protocol.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(protocol)


if __name__=='__main__':main()
