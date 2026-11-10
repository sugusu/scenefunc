"""Build the author's actual physics prompt, then encode locally without weights.

No completion is produced. JSON response-format request is recorded, not claimed
to be a decoder constraint enforced by the local adapter.
"""
import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, d):
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    tick = time.monotonic()
    out = ROOT/'physics_input_v1'
    out.mkdir(exist_ok=False)
    for row in read(ROOT/'prompt_source_receipt.json'):
        assert sha(ROOT/'source'/row['path']) == row['sha256']
    package = types.ModuleType('native_physics_prompt')
    package.__path__ = [str(ROOT/'source/scenesmith/prompts')]
    sys.modules[package.__name__] = package
    manager = load(package.__name__+'.manager', Path(package.__path__[0])/'manager.py')
    registry = load(package.__name__+'.registry', Path(package.__path__[0])/'registry.py')
    encoder = load('native_physics_image_encoding', ROOT/'source/scenesmith/utils/openai.py')
    result = read(ROOT/'actual_v2/result.json')
    assert result['status'] == 'native_asset_converted_and_physics_views_rendered'
    image_paths = [Path(r['path']) for r in result['views']]
    assert all(sha(p) == row['sha256'] for p, row in zip(image_paths, result['views']))
    # Execute only native request-building statements; renderer already executed
    # and VLM boundary remains deliberately uncalled.
    analyzer = ROOT.parent/'20261110_native_asset_contract/tool_source/mesh_physics_analyzer.py'
    namespace = dict(prompt_type='hssd', MeshPhysicsPrompts=registry.MeshPhysicsPrompts,
        prompt_manager=manager.PromptManager(ROOT/'source/scenesmith/prompts/data'),
        image_paths=image_paths, mesh_path=Path(result['generated_gltf']),
        encode_image_to_base64=encoder.encode_image_to_base64)
    fn = next(n for n in ast.parse(analyzer.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == 'analyze_mesh_orientation_and_material')
    assignments = {'prompt_enum', 'system_prompt', 'encoded_images', 'user_content', 'messages'}
    selected = []
    for n in fn.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in n.targets):
            selected.append(n)
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'img_base64':
            selected.append(n)
    assert len(selected) == 6
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(analyzer), 'exec'), namespace)
    messages = namespace['messages']
    dump(out/'native_request_local_only.json', messages)
    from PIL import Image
    encodings = []
    for i, item in enumerate(messages[0]['content'][2:]):
        uri = item['image_url']['url']
        data = base64.b64decode(uri.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(data)) as im:
            encodings.append(dict(index=i, source_png_sha256=sha(image_paths[i]),
                encoded_bytes_sha256=hashlib.sha256(data).hexdigest(),
                declared_media_type=uri.split(';', 1)[0][5:],
                actual_payload_format=im.format, size=[im.width, im.height]))
    # Preserve author PNG URI / JPEG-byte behavior; do not quietly call these
    # byte-identical copies of the source PNG views.
    assert all(r['actual_payload_format'] == 'JPEG' for r in encodings)
    dump(out/'native_image_encoding.json', encodings)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    attempted = []

    def reject(*args, **kwargs):
        attempted.append(True)
        raise RuntimeError('Network disabled for local processor-only encoding')

    socket.socket.connect = reject
    socket.create_connection = reject
    sys.path.insert(0, str(WORK/'paris_local_run_results_20261104'))
    from local_vlm_sdpa import LocalQwenEngine
    import torch
    torch.set_num_threads(1)
    engine = LocalQwenEngine(WORK/'critic_public_runtime_20261104/checkpoint',
        out/'unused_model_calls', max_pixels=65536, max_input_tokens=8192, max_new_tokens=512)
    inputs, encoded = engine.encode(messages)
    assert engine._model is None and not torch.cuda.is_initialized() and not attempted
    receipt = dict(status='native_physics_request_built_and_cpu_encoded',
        input_tokens=encoded['input_tokens'], image_count=len(encoded['images']),
        processor_receipt=encoded, encoded_images=encodings,
        model_loaded=False, cuda_initialized=False, model_calls=0, paid_api_calls=0,
        training_updates=0, schema_constrained_generation_executed=False,
        native_requested_response_format={'type':'json_object'},
        analyzer_sha256=sha(analyzer), builder_source_lines=[[n.lineno,n.end_lineno] for n in selected],
        request_sha256=sha(out/'native_request_local_only.json'),
        prompt_sha256=hashlib.sha256(namespace['system_prompt'].encode()).hexdigest(),
        seconds=time.monotonic()-tick,
        limitation='Local processor has max_pixels65536, not unmodified official API vision preprocessing. Native text and encoded payload retained; no estimated material/mass/front or SDF fabricated.')
    dump(out/'result.json', receipt)
    print(json.dumps({k:receipt[k] for k in ('status','input_tokens','image_count','seconds','model_loaded')}) )


if __name__ == '__main__':
    main()
