"""Whole native controller modules with an explicit remote model adaptation.

Old source/results stay read-only. Copy the qualified source closure, remove
only in-process Transformers model loading, retain native classes/parsers/tool
loops, and fetch the small pinned default materials and original stage prompts.
No model or service is started here. This is ordinary backend preparation.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def adapt_controller_source(before):
    """Replace selected source spans; preserve Python3.11 f-string spelling.

    ast.unparse in the Python3.12 preparation host may emit PEP701 f-strings
    rejected by the native Python3.11 interpreter. Leave other bytes intact.
    """
    edits, changes = [], []
    for node in ast.parse(before).body:
        kind = None
        replacement = None
        if isinstance(node, ast.ImportFrom) and node.module == 'scenesmith.llm_provider':
            kind, replacement = 'external_provider_gateway_disabled', ''
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Name) and node.value.func.id == '_configure_llm_gateway':
            kind, replacement = 'external_provider_gateway_startup_disabled', ''
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('transformers'):
            kind, replacement = 'in_process_model_import', ''
        elif isinstance(node, ast.Try) and any(isinstance(n, ast.ImportFrom) and n.module
                and n.module.startswith('transformers') for n in ast.walk(node)):
            kind, replacement = 'in_process_model_import_fallback', ''
        elif isinstance(node, ast.FunctionDef) and node.name == 'model_generate':
            kind = 'in_process_generation_disabled'
            replacement = ('def model_generate(*args, **kwargs):\n'
                '    raise RuntimeError("In-process policy loading is disabled; use the explicit remote policy")\n')
        elif isinstance(node, ast.ClassDef) and node.name == 'ModelPolicy':
            kind = 'remote_native_policy_constructor'
            replacement = ('from remote_native_control import RemoteNativePolicy\n'
                'class ModelPolicy(RemoteNativePolicy):\n'
                '    def __init__(self, model_dir, max_new_tokens=512):\n'
                '        super().__init__(model_dir, max_new_tokens, parser=parse_assistant_text)\n')
        if kind:
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            edits.append((first, node.end_lineno, replacement))
            changes.append({'line': first, 'end_line': node.end_lineno, 'kind': kind})
    lines = before.splitlines(keepends=True)
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first-1:last] = [replacement]
    after = ''.join(lines)
    compile(after, '<remote-native-controller>', 'exec')
    return after, changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    work = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    runtime = args.out / 'runtime'
    native = runtime / 'native_source'
    source_stage = repo / 'runs/autodl/20261110_live_manager_request'
    manifest = json.loads((source_stage / 'source_manifest.json').read_text())
    assert shutil.disk_usage(args.out).free > 5 * 1024**3 + 16 * 1024**2
    copied = []
    for row in manifest['files']:
        src = source_stage / 'source' / row['path']
        assert sha(src) == row['actual_sha256']
        dst = native / row['path']
        write(dst, src.read_bytes())
        copied.append({'source': str(src), 'destination': str(dst), 'sha256': sha(src)})
    integration = work / 'native_execution_inventory_20261104/source'
    for src in integration.rglob('*.py'):
        dst = runtime / src.relative_to(integration)
        write(dst, src.read_bytes())
        copied.append({'source': str(src), 'destination': str(dst), 'sha256': sha(src)})
    for name in ('memory.py', 'scene_observation.py'):
        src = work / 'memory_replay_source_20261103/src' / name
        dst = runtime / 'src' / name
        write(dst, src.read_bytes())
        copied.append({'source': str(src), 'destination': str(dst), 'sha256': sha(src)})
    renderer = repo / 'runs/autodl/20261110_native_observation_export/checkpoint_renderer'
    for dirname in ('tools', 'src_critic'):
        for src in (renderer / dirname).rglob('*.py'):
            dst = runtime / src.relative_to(renderer)
            write(dst, src.read_bytes())
            copied.append({'source': str(src), 'destination': str(dst), 'sha256': sha(src)})
    patches = []
    for rel in ('Designer/agent_rollout.py', 'src/agent_rollout_decoupled.py', 'Designer/replay_chained.py'):
        path = runtime / rel
        before = path.read_text()
        after, removed = adapt_controller_source(before)
        path.write_text(after)
        compile(after, str(path), 'exec')
        patches.append({'path': str(path), 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': sha(path), 'semantic_changes': removed,
                        'formatting': 'Selected original line spans replaced; all other source bytes retained'})
    # Original stage templates, not a case-specific rewrite to fix generated outputs.
    designer_pin = 'a0565e50ed7d32c0e09132b9311c0b9b91165405'
    requests = []
    for stage in ('floor_plan', 'furniture', 'wall', 'ceiling'):
        for suffix in ('system_prompt', 'tool'):
            rel = f'prompts/{stage}_{suffix}'
            requests.append({'destination': runtime / rel,
                'url': f'https://raw.githubusercontent.com/ZJU-OmniAI/Scene-Designer/{designer_pin}/{rel}',
                'limit': 128 * 1024, 'git_blob': None})
    source_tree_path = work / 'native_room_bridge_20261104/source_tree.json'
    source_tree = json.loads(source_tree_path.read_text())
    assert source_tree['sha'] == manifest['base_pin']
    for row in source_tree['tree']:
        if row.get('type') == 'blob' and row['path'].startswith('materials/') and row['path'].endswith('.jpg'):
            requests.append({'destination': native / row['path'],
                'url': f'https://raw.githubusercontent.com/nepfaff/scenesmith/{manifest["base_pin"]}/{row["path"]}',
                'limit': row['size'], 'git_blob': row['sha']})
    assert sum(r['limit'] for r in requests) < 8 * 1024**2
    def fetch(row):
        record = {k: str(v) if isinstance(v, Path) else v for k, v in row.items()}
        try:
            with urllib.request.urlopen(row['url'], timeout=30) as response:
                data = response.read(row['limit'] + 1)
            if len(data) > row['limit']:
                raise ValueError('Pinned small-file byte cap exceeded')
            if row['git_blob']:
                assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_blob']
            write(row['destination'], data)
            record.update(status='fetched', bytes=len(data), sha256=sha(row['destination']))
        except Exception as exc:
            record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(fetch, requests))
    # New explicit common configuration, preserving the old overlay and source.
    import yaml
    original_overlay = repo / 'runs/autodl/20261110_native_service_contract/inputs/integration/scene_designer.yaml'
    cfg = yaml.safe_load(original_overlay.read_text())
    for stage in ('floor_plan', 'furniture', 'wall', 'ceiling', 'manipuland'):
        section = cfg[stage + '_agent']
        section.setdefault('session_memory', {})['enable_summarization'] = False
        section.setdefault('context_image_generation', {})['enabled'] = False
        if stage == 'floor_plan':
            section['materials']['use_retrieval_server'] = False
            continue
        manager = section.setdefault('asset_manager', {})
        manager['general_asset_source'] = 'hssd'
        manager.setdefault('router', {})['enabled'] = False
        strategies = manager['router'].setdefault('strategies', {})
        strategies.setdefault('articulated', {})['enabled'] = False
        strategies.setdefault('thin_covering', {}).setdefault('generator', {})['enabled'] = False
        manager['hssd'] = {
            'data_path': '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/official_assets/hssd-models',
            'preprocessed_path': '/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/vendor/hsm/data/preprocessed'}
    path = native / 'configurations/scene_designer_local.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = {'status': 'materialized' if all(r['status'] == 'fetched' for r in fetched) else 'partial_missing_small_files',
        'runtime_root': str(runtime.resolve()), 'native_root': str(native.resolve()),
        'native_pin': manifest['base_pin'], 'designer_pin': designer_pin,
        'copied_sources': copied, 'patches': patches, 'fetched': fetched,
        'new_configuration': str(path.resolve()), 'configuration_sha256': sha(path),
        'model_calls': 0, 'services_started': 0, 'paid_API_calls': 0,
        'scope': 'Whole controller module adaptation and ordinary native local materials; not runtime or task success'}
    (args.out / 'materialization.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert all(sha(r['source']) == r['sha256'] for r in copied)
    print(json.dumps({'status': result['status'], 'copied_files': len(copied),
                      'fetched_files': sum(r['status'] == 'fetched' for r in fetched),
                      'failed_files': sum(r['status'] != 'fetched' for r in fetched)}))


if __name__ == '__main__':
    main()
