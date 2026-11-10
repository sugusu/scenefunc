"""Execute pinned real SDK tools through the local adapter, without inference.

The exact native vision-tool closure is extracted with AST, not reimplemented.
Its rendering implementation deliberately replays one existing PNG; this tests
SDK wrapping, ToolContext and structured media, NOT rendering or scene quality.
"""
import argparse
import ast
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any


async def run(a):
    started = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.runtime.resolve()))
    os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = '1'
    network_attempts = []
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            network_attempts.append(event)
            raise RuntimeError('Network disabled in SDK qualification')
    sys.addaudithook(audit)
    import agents
    from agents import ToolOutputImage, ToolOutputText, function_tool
    from agents.tool_context import ToolContext
    from pydantic import ValidationError
    from local_manipuland_workflow import LocalManipulandWorkflow, observation_parts, output_value
    from local_tool_vlm import LocalToolVLMBackend, prepare_tool_messages
    assert Path(agents.__file__).resolve().is_relative_to(a.runtime.resolve())
    source = a.source.read_text()
    cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef)
               and x.name == 'ManipulandVisionTools')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == '_create_tool_closures')
    ns = dict(Any=Any, function_tool=function_tool,
              ToolOutputImage=ToolOutputImage, ToolOutputText=ToolOutputText)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(a.source), 'exec'), ns)
    raw = a.image.read_bytes()
    url = 'data:image/png;base64,' + base64.b64encode(raw).decode()
    class Replay:
        def _observe_scene_impl(self):
            return [ToolOutputImage(image_url=url, detail='high'),
                    ToolOutputText(text='Cached view replay for SDK integration only')]
        def _check_physics_impl(self):
            raise AssertionError('No physics should run in this SDK test')
    tools = ns['_create_tool_closures'](Replay())
    class UnusedBackend(LocalToolVLMBackend):
        def __init__(self):
            self.rows = []
        def create_completion(self, **kwargs):
            raise AssertionError('Model inference forbidden in SDK boundary test')
    workflow = LocalManipulandWorkflow(UnusedBackend(), out=a.out/'tool_log',
        requested_model='unused', max_tool_calls=5)
    returned = await workflow.invoke(tools, 'observe_scene', {})
    parts = observation_parts(returned)
    converted, images, records = prepare_tool_messages([dict(role='user', content=parts)])
    try:
        assert len(images) == 1
        assert records[0]['sha256'] == hashlib.sha256(raw).hexdigest()
        assert records[0]['requested_detail'] == 'high'
    finally:
        for im in images:
            im.close()
    # A second actual decorated tool verifies explicit context injection and
    # JSON -> Pydantic validation, independently of the no-argument native tool.
    contexts = []
    @function_tool(failure_error_function=None)
    async def context_echo(ctx: ToolContext[Any], count: int) -> str:
        contexts.append(dict(name=ctx.tool_name, call_id=ctx.tool_call_id,
                             arguments=ctx.tool_arguments))
        return json.dumps(dict(count=count))
    tools['context_echo'] = context_echo
    echo = await workflow.invoke(tools, 'context_echo', {'count': 7})
    assert json.loads(echo) == {'count': 7}
    assert contexts[0]['name'] == 'context_echo'
    rejected = []
    try:
        await workflow.invoke(tools, 'context_echo', {'count': {}})
    except Exception as exc:
        assert 'Invalid JSON input' in str(exc) or 'validation error' in str(exc)
        rejected.append(dict(case='invalid_typed_argument', exception=type(exc).__name__))
    else:
        raise AssertionError('Invalid typed input accepted')
    for label, value in [('file_id_only', [ToolOutputImage(file_id='offline-fixture')]),
                         ('text_only', [ToolOutputText(text='No image available')]),
                         ('opaque_string', [str(ToolOutputImage(image_url=url))])]:
        try:
            observation_parts(value)
        except (ValueError, TypeError) as exc:
            rejected.append(dict(case=label, exception=type(exc).__name__))
        else:
            raise AssertionError(label + ' silently accepted')
    try:
        ToolOutputImage()
    except ValidationError:
        rejected.append(dict(case='sdk_requires_image_source', exception='ValidationError'))
    else:
        raise AssertionError('SDK accepted missing source')
    assert not network_attempts
    assert len(contexts) == 1
    assert len(workflow.tool_calls) == 3
    summary = dict(status='passed', actual_sdk=True,
        versions={p: importlib.metadata.version(p) for p in
                  ('openai-agents', 'openai', 'pydantic', 'mcp', 'griffe')},
        native_source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        exact_native_method='_create_tool_closures',
        tool_schemas={k: v.params_json_schema for k, v in tools.items()},
        context_injection=contexts, rejections=rejected, actual_tool_invocations=3,
        recorded_statuses=[r['status'] for r in workflow.tool_calls],
        native_observation_tool_invocations=1, image=records[0],
        actual_render_calls=0, actual_geometry_calls=0, actual_model_calls=0,
        paid_api_calls=0, new_complete_scenes=0, network_attempts=network_attempts,
        full_runtime_ready=False, seconds=time.perf_counter()-started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Real SDK and original native closure, cached PNG replay only. '
                   'No model, renderer, asset server or geometry executed.')
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'source', 'image', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    asyncio.run(run(ap.parse_args()))
