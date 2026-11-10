"""Execute only the pinned native configuration class and inference kwargs.

No model module import, tokenizer, weights, tensor operation, CUDA or upstream
benchmark. This identifies a source entrypoint incompatibility, not policy
quality. The AST isolation is explicit and not a full import qualification.
"""
import __future__
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    model_path = a.source / 'src/nala/models/model7B.py'
    infer_path = a.source / 'scripts/infer.py'
    default_path = a.source / 'src/nala/eval_defaults.py'
    trees = {str(p): ast.parse(p.read_text()) for p in (model_path, infer_path, default_path)}
    config = next(n for n in trees[str(model_path)].body if isinstance(n, ast.ClassDef) and n.name == 'model_config')
    call = next(n for n in ast.walk(trees[str(infer_path)])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'model_config')
    defaults = {}
    for node in trees[str(default_path)].body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    kwargs = {}
    for keyword in call.keywords:
        if isinstance(keyword.value, ast.Name):
            kwargs[keyword.arg] = defaults[keyword.value.id]
        else:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    namespace = {}
    code = compile(ast.Module(body=[config], type_ignores=[]), str(model_path), 'exec',
                   flags=__future__.annotations.compiler_flag)
    exec(code, namespace)
    error = None
    try:
        namespace['model_config'](**kwargs)
    except TypeError as exc:
        error = str(exc)
    assert error and 'lora_r' in error
    result = {'status': 'pinned_config_and_inference_entrypoint_incompatible',
              'source_constructor': f'{model_path}:{config.lineno}',
              'source_call': f'{infer_path}:{call.lineno}',
              'native_signature': str(inspect.signature(namespace['model_config'])),
              'actual_inference_kwargs': kwargs, 'actual_error': error,
              'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (model_path, infer_path, default_path)},
              'AST_isolation': 'Only original model_config ClassDef, unmodified body; future annotation evaluation avoids imports. Caller kwargs/defaults are original literals.',
              'full_upstream_import_tested': False, 'model_loads': 0,
              'training_updates': 0, 'scene_generations': 0,
              'not_inferred': 'No claim about paper results, unreleased checkpoints, or whether another unpublished entrypoint resolves this mismatch',
              'source_patched': False}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'actual_error': error,
                      'model_loads': 0, 'source_patched': False}))


if __name__ == '__main__':
    main()
