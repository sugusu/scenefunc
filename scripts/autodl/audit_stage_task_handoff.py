"""Source-backed native parent-selection and workflow input capture, no model.

Uses a restored existing RoomScene. Selection rendering/model and workflow tool
dispatch are deliberately interrupted/substituted, and never scored as scenes.
"""
import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import socket
import sys
import time
from types import SimpleNamespace


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


class CapturedBeforeModel(BaseException):
    pass


async def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[2]
    source = repo / 'runs/autodl/20261110_live_manager_request/source'
    sys.path.insert(0, str(source))
    protected = [source / 'scenesmith/agent_utils/scene_analyzer.py',
        source / 'scenesmith/manipuland_agents/stateful_manipuland_agent.py',
        repo / 'scripts/autodl/local_manipuland_workflow.py',
        repo / 'scripts/autodl/complete_native_pipeline.py',
        repo / 'scripts/autodl/native_manipuland_runner.py', args.state, args.image,
        args.scene_root / 'room.sdf']
    before = {str(p): sha(p) for p in protected}
    attempts = []
    def reject(*a, **k):
        attempts.append('blocked')
        raise RuntimeError('No network during source input capture')
    socket.socket.connect = reject
    socket.create_connection = reject
    from omegaconf import OmegaConf
    from scenesmith.agent_utils.room import RoomScene
    from scenesmith.agent_utils.scene_analyzer import SceneAnalyzer
    from local_manipuland_workflow import LocalManipulandWorkflow
    from local_tool_vlm import LocalToolVLMBackend
    from task_grounded_completion import (TaskGroundedManipulandWorkflow,
        bind_original_task_to_selection, task_context, WORKFLOW_TRANSFORM)
    import torch
    scene = RoomScene(room_geometry=None, scene_dir=args.scene_root,
                      room_id='existing_state_input_capture', text_description='')
    scene.restore_from_state_dict(json.loads(args.state.read_text()))
    state_before = digest(scene.to_state_dict())
    derived = scene.text_description
    # Two explicit input-transport fixtures, not real preferences or labels.
    tasks = ['在当前房间的一张桌面放一个杯子，保留已有家具。',
             '在当前房间的两张桌面各放一个杯子，保留已有家具。']
    analyzer = SceneAnalyzer.__new__(SceneAnalyzer)
    analyzer.cfg = OmegaConf.create({'openai': {'furniture_analysis_max_retries': 1}})
    selector_rows = []
    def capture_analysis(*a, **kw):
        selector_rows.append({'prompt_kwargs': kw['prompt_kwargs'],
                              'settings_key': kw['settings_key'],
                              'rendering_params': kw['rendering_params'],
                              'same_scene_instance': kw['scene'] is scene})
        raise CapturedBeforeModel()
    analyzer.analyze_scene = capture_analysis
    for task in [None] + tasks:
        audit = []
        try:
            if task is None:
                analyzer.analyze_furniture_for_manipulands(scene, 'capture_only')
            else:
                with bind_original_task_to_selection(analyzer, task, audit=audit):
                    analyzer.analyze_furniture_for_manipulands(scene, 'capture_only')
        except CapturedBeforeModel:
            pass
        else:
            raise AssertionError('Expected native selection call to reach input capture')
        row = selector_rows[-1]
        row.update(condition='native' if task is None else 'original_task_passthrough',
                   task_transport_audit=audit, stopped_before_render_and_model=True)
        desc = row['prompt_kwargs']['scene_description']
        if task is None:
            assert desc == derived and all(t not in desc for t in tasks)
        else:
            payload = json.loads(desc)
            assert payload['original_user_request'] == task
            assert payload['derived_room_request'] == derived
        assert analyzer.analyze_scene is capture_analysis
        assert digest(scene.to_state_dict()) == state_before
    assert selector_rows[0]['prompt_kwargs']['furniture_list'] == selector_rows[1]['prompt_kwargs']['furniture_list']
    assert selector_rows[0]['rendering_params'] == selector_rows[2]['rendering_params']

    # Capture the actual local-workflow request after its original loop has
    # consumed a real image and real native state through read-only wrappers.
    url = 'data:image/png;base64,' + base64.b64encode(args.image.read_bytes()).decode()
    workflow_rows = []
    class CaptureBackend(LocalToolVLMBackend):
        def __init__(self):
            self.rows = []
        def create_completion(self, **kw):
            workflow_rows.append({'messages': kw['messages'],
                                  'scope': 'request capture; no completion supplied'})
            raise CapturedBeforeModel()
    def tool(name, callback):
        async def invoke(ctx, arguments):
            assert json.loads(arguments) == {}
            return callback()
        return SimpleNamespace(name=name, description='Read-only capture wrapper, not native SDK dispatch',
            params_json_schema={'type': 'object', 'properties': {}}, on_invoke_tool=invoke)
    tools = [tool('observe_scene', lambda: [{'type': 'image', 'image_url': url}]),
             tool('get_current_scene_state', scene.to_state_dict)]
    parent_id = next(str(o.object_id) for o in scene.objects.values()
                     if o.name == 'mobile_table')
    agent = SimpleNamespace(scene=scene, current_furniture_selection=SimpleNamespace(
        suggested_items=[], prompt_constraints='Capture fixture; no VLM selection performed', style_notes=''))
    for task in [None] + tasks:
        backend = CaptureBackend()
        kwargs = dict(out=args.out / ('workflow_' + str(len(workflow_rows))),
                      requested_model='capture_only', max_steps_per_parent=1,
                      context_factory=lambda **kw: SimpleNamespace(**kw))
        workflow = (LocalManipulandWorkflow(backend, **kwargs) if task is None else
                    TaskGroundedManipulandWorkflow(backend, original_user_request=task, **kwargs))
        try:
            await workflow(agent, parent_id, tools)
        except CapturedBeforeModel:
            pass
        else:
            raise AssertionError('Expected workflow to stop before inference')
        row = workflow_rows[-1]
        public = json.loads(row['messages'][-1]['content'][-1]['text'])
        # Do not duplicate inline image bytes in public artifacts.
        row.pop('messages')
        row.update(public_context=public, condition='native' if task is None else 'original_task_passthrough',
                   observation_image_sha256=sha(args.image),
                   stopped_before_model=True, synthetic_read_only_wrappers=True,
                   workflow_receipt=str(workflow.out / 'workflow.json'))
        if task is None:
            assert public['original_room_request'] == derived
        else:
            assert public['task_provenance'] == task_context(task, derived)
            assert 'original_room_request' not in public
        assert digest(scene.to_state_dict()) == state_before
    assert workflow_rows[0]['public_context']['actual_state'] == workflow_rows[1]['public_context']['actual_state']
    missing_rejected = False
    try:
        task_context('', derived)
    except ValueError:
        missing_rejected = True
    assert missing_rejected and not attempts
    assert all(sha(p) == h for p, h in before.items())
    dump(args.out / 'result.json', {
        'status': 'actual_native_selection_input_and_common_workflow_context_captured',
        'native_room_objects': len(scene.objects),
        'native_state_before_sha256': state_before,
        'native_state_after_sha256': digest(scene.to_state_dict()),
        'selection_rows': selector_rows, 'workflow_rows': workflow_rows,
        'workflow_transform': WORKFLOW_TRANSFORM,
        'original_files_before': before, 'original_files_after': {p: sha(p) for p in before},
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                        'torch_cuda_available': torch.cuda.is_available(),
                        'transformers_importable': importlib.util.find_spec('transformers') is not None},
        'model_calls': 0, 'new_generated_scenes': 0, 'training_steps': 0,
        'network_attempts': len(attempts), 'missing_original_task_rejected': missing_rejected,
        'whole_runtime_ready': False, 'learned_advantage_verified': False,
        'fixture_tasks_are_human_preference_data': False,
        'limits': ['Selection method uses real native filtering but stops before native rendering/model',
                   'Workflow uses explicit read-only wrappers, not native SDK dispatch',
                   'Fixture tasks test provenance only, not real-user preference or generation success',
                   'No end-to-end five-stage or changed model behavior measured'],
        'seconds': time.monotonic() - start,
        'parent_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    print(json.dumps({'status': 'captured', 'model_calls': 0,
                      'selection_inputs': len(selector_rows), 'workflow_inputs': len(workflow_rows),
                      'seconds': time.monotonic() - start}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--scene-root', type=Path, required=True,
                   help='Verified existing root for relative resource paths; never infer from JSON location')
    p.add_argument('--image', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
