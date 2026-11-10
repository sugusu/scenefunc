"""Ordinary task passthrough for the common small-object stage.

Original files stay unchanged. Preserve the original task before native parent
selection as well as at per-parent execution; a derived room description must
not be mislabeled as the original instruction. No model/asset/server is loaded
at import. This is a baseline repair, not a new generation algorithm.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

import local_manipuland_workflow as original_workflow


def task_context(original_user_request, derived_room_request):
    if not isinstance(original_user_request, str) or not original_user_request.strip():
        raise ValueError('An explicit original user request is required')
    if not isinstance(derived_room_request, str):
        raise TypeError('Derived room request must remain textual')
    return {
        'original_user_request': original_user_request,
        'derived_room_request': derived_room_request,
        'source_rule': 'The derived description and native parent suggestions are model '
                       'decompositions, not replacements for the original user request. '
                       'Apply original requirements relevant to this actual room; '
                       'unresolved relevance stays uncertain.'}


@contextmanager
def bind_original_task_to_selection(analyzer, original_user_request, *, audit):
    """Modify only furniture-analysis prompt kwargs; never change scene fields.

    Instance-scoped serialized use only. Native rendering, selection logic,
    retries and parsing stay unchanged. Existing own methods are restored.
    """
    task_context(original_user_request, '')  # Validate before binding.
    original = analyzer.analyze_scene
    had_own = 'analyze_scene' in analyzer.__dict__
    own = analyzer.__dict__.get('analyze_scene')

    def analyze(*args, **kwargs):
        if kwargs.get('settings_key') != 'furniture_analysis':
            return original(*args, **kwargs)
        patched = dict(kwargs)
        prompt_kwargs = dict(patched['prompt_kwargs'])
        payload = task_context(original_user_request, prompt_kwargs['scene_description'])
        prompt_kwargs['scene_description'] = json.dumps(payload, ensure_ascii=False)
        patched['prompt_kwargs'] = prompt_kwargs
        audit.append({'original_user_request_sha256': hashlib.sha256(original_user_request.encode()).hexdigest(),
                      'derived_room_request_sha256': hashlib.sha256(payload['derived_room_request'].encode()).hexdigest(),
                      'input_prompt_fields': sorted(prompt_kwargs),
                      'native_scene_mutated': False})
        return original(*args, **patched)

    analyzer.analyze_scene = analyze
    try:
        yield
    finally:
        if had_own:
            analyzer.analyze_scene = own
        else:
            del analyzer.analyze_scene


def _workflow_call_with_explicit_context():
    """Transform exactly one context keyword in the old, otherwise same loop.

    Separate module avoids changing frozen old evidence that hashes the original
    file. Fail if the expected original context-construction site changes.
    """
    source = Path(original_workflow.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != '137be5f8c5b19878d5a3f2a26ab49812964bb1f01abb19334d2ec7550d9995e2':
        raise ValueError('Old common workflow differs from the inspected version')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'LocalManipulandWorkflow')
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                            and n.name == '__call__'))
    original_ast = ast.dump(fn, include_attributes=False)
    changed = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
            for kw in node.keywords:
                if kw.arg == 'original_room_request':
                    assert ast.unparse(kw.value) == 'agent.scene.text_description'
                    kw.arg = 'task_provenance'
                    kw.value = ast.parse('task_context(self.original_user_request, '
                                         'agent.scene.text_description)', mode='eval').body
                    changed += 1
    if changed != 1:
        raise ValueError('Original workflow context site changed')
    reverted = copy.deepcopy(fn)
    for node in ast.walk(reverted):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
            for kw in node.keywords:
                if kw.arg == 'task_provenance':
                    kw.arg = 'original_room_request'
                    kw.value = ast.parse('agent.scene.text_description', mode='eval').body
    assert ast.dump(reverted, include_attributes=False) == original_ast
    tree = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = dict(vars(original_workflow), task_context=task_context)
    exec(compile(tree, str(source) + ':explicit_task_context', 'exec'), namespace)
    receipt = {'original_file': str(source),
               'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'original_method_ast_sha256': hashlib.sha256(original_ast.encode()).hexdigest(),
               'transformed_method_ast_sha256': hashlib.sha256(
                   ast.dump(fn, include_attributes=False).encode()).hexdigest(),
               'context_keyword_transformations': changed,
               'method_ast_identical_after_reverting_one_change': True,
               'change': 'original_room_request=derived -> task_provenance=task_context(original, derived)',
               'policy_loop_otherwise_unchanged': True}
    return namespace['__call__'], receipt


_call, WORKFLOW_TRANSFORM = _workflow_call_with_explicit_context()


class TaskGroundedManipulandWorkflow(original_workflow.LocalManipulandWorkflow):
    __call__ = _call

    def __init__(self, *args, original_user_request, **kwargs):
        task_context(original_user_request, '')
        self.original_user_request = original_user_request
        super().__init__(*args, **kwargs)


async def run_task_grounded_manipulands(agent, scene, *, original_user_request,
                                       workflow, receipt):
    from native_manipuland_runner import run_native_manipulands
    receipt.data['task_provenance'] = task_context(original_user_request, scene.text_description)
    receipt.data['selection_task_transport'] = []
    receipt.save()
    # The native parent-selection method calls this same analyzer. The binding
    # remains active only until the original stage returns; stage calls are serial.
    with bind_original_task_to_selection(agent.scene_analyzer, original_user_request,
                                        audit=receipt.data['selection_task_transport']):
        return await run_native_manipulands(agent, scene, workflow=workflow, receipt=receipt)


def task_grounded_manipuland_driver(tool_binding, *, max_steps_per_parent=4,
                                    max_tool_calls=48, max_seconds=600):
    """Explicit alternative callback for complete_native_pipeline.

    Same native constructors/tools and existing common local workflow. Existing
    service/model qualification is still required; this does not bypass guards.
    """
    async def run(native, session, policy, args):
        if tool_binding.backend is None:
            raise RuntimeError('Shared local model has not been bound')
        from scenesmith.manipuland_agents.stateful_manipuland_agent import StatefulManipulandAgent
        from native_manipuland_runner import NativeStageReceipt
        out = session.scene.scene_dir / 'rollout_manipuland'
        logger = native.ConsoleLogger(output_dir=session.scene.scene_dir)
        agent = native.BaseExperiment.build_manipuland_agent(cfg_dict=session.cfg_dict,
            compatible_agents={'stateful_manipuland_agent': StatefulManipulandAgent},
            logger=logger, render_gpu_id=args.render_gpu_id)
        try:
            receipt = NativeStageReceipt(out / 'stage_receipt.json')
            workflow = TaskGroundedManipulandWorkflow(tool_binding.backend,
                original_user_request=args.prompt, out=out / 'local_policy',
                requested_model=agent.cfg.openai.model,
                max_steps_per_parent=max_steps_per_parent,
                max_tool_calls=max_tool_calls, max_seconds=max_seconds)
            session.scene = await run_task_grounded_manipulands(agent, session.scene,
                original_user_request=args.prompt, workflow=workflow, receipt=receipt)
            logger.log_scene(scene=session.scene, name='scene_after_manipulands')
            export = session.scene.scene_dir / 'final_manipuland/manipuland.blend'
            native._save_roomscene_blend(agent, export)
            return {'status': receipt.data['status'], 'receipt': str(receipt.path),
                    'export': str(export), 'full_scene_success': None,
                    'scope': 'Task passthrough common baseline; not native SDK reproduction'}
        finally:
            agent.cleanup()
    return run
