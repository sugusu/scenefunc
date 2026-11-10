"""Extract source contracts behind future targets without running native tools."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.repository.resolve()
    source = root / 'runs/autodl/20261110_live_manager_request/source'
    manifest_path = source.parent / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bypath = {r['path']: r for r in manifest['files']}
    definitions = [
        ('scenesmith/agent_utils/room.py', 'RoomScene', 'move_object',
         'Assigns one object transform; no physical interaction is simulated by this method'),
        ('scenesmith/agent_utils/room.py', 'RoomScene', 'remove_object',
         'Deletes one registry object if found; not a task-completeness guarantee'),
        ('scenesmith/furniture_agents/tools/furniture_tools.py', 'FurnitureTools', '_move_furniture_impl',
         'Absolute pose, input validation, configured placement noise; endpoint does not by itself certify task relations'),
        ('scenesmith/furniture_agents/tools/furniture_tools.py', 'FurnitureTools', '_add_furniture_to_scene_impl',
         'Copies an existing registry template with a new identity/pose; does not itself acquire a new asset'),
        ('scenesmith/furniture_agents/tools/furniture_tools.py', 'FurnitureTools', '_generate_assets_impl',
         'Calls acquisition and reports partial/all-success; this is distinct from placement'),
        ('scenesmith/agent_utils/asset_manager.py', 'AssetManager', 'generate_assets',
         'Router branch and non-router HSSD/Objaverse/generated dispatch are different contracts'),
        ('scenesmith/agent_utils/asset_manager.py', 'AssetManager', '_retrieve_hssd_assets',
         'Actual asset result plus model-estimated orientation/material/mass change downstream geometry/metadata'),
    ]
    records = []
    hashes = {}
    for relative, cls, name, meaning in definitions:
        path = source / relative
        h = sha(path)
        if h != bypath[relative]['actual_sha256']:
            raise ValueError('Qualified source changed: ' + relative)
        hashes[str(path.relative_to(root))] = h
        tree = ast.parse(path.read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        if len(classes) != 1:
            raise ValueError('Class not unique: ' + cls)
        methods = [n for n in classes[0].body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        if len(methods) != 1:
            raise ValueError('Method not unique: ' + name)
        node = methods[0]
        calls = sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
        records.append({'source': relative, 'class': cls, 'method': name,
                        'start_line': node.lineno, 'end_line': node.end_lineno,
                        'source_sha256': h, 'calls': calls, 'reviewed_contract': meaning,
                        'analysis': 'AST structure plus separately read method body; not dynamic execution'})

    noise_path = source / 'scenesmith/agent_utils/placement_noise.py'
    noise_sha = sha(noise_path)
    if noise_sha != bypath['scenesmith/agent_utils/placement_noise.py']['actual_sha256']:
        raise ValueError('Placement source changed')
    hashes[str(noise_path.relative_to(root))] = noise_sha
    hashes[str(manifest_path.relative_to(root))] = sha(manifest_path)
    metadata_path = source.parent / 'asset_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    hashes[str(metadata_path.relative_to(root))] = sha(metadata_path)
    result = {
        'base_pin': manifest['base_pin'], 'source_methods': records,
        'noise_contract': 'Reviewed file uses a pose-derived seed. Runtime/version/config must still be fixed; not proof that all asset/model operations are deterministic.',
        'new_model_or_native_executions': 0,
        'reused_live_asset_observation': metadata,
        'important_limit': 'Requested wooden versus predicted plastic is an unresolved disagreement, not independently established material error or a valid preference label',
        'known_transition_scope': 'Exposed absolute-pose/delete operations with verified identity/config; not every native stage or latent observation',
        'missing_target_scope': ['independent semantic/material truth', 'successful complete-room future',
                                 'verified task-preferred goal geometry', 'held-out assets/rooms/operator effects'],
        'hashes_before': hashes,
        'hashes_after': {p: sha(root / p) for p in hashes},
    }
    result['old_inputs_unchanged'] = result['hashes_before'] == result['hashes_after']
    if not result['old_inputs_unchanged']:
        raise ValueError('Read-only source changed')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'methods': len(records), 'old_inputs_unchanged': True,
                      'source_pin': manifest['base_pin'], 'native_executions': 0}))


if __name__ == '__main__':
    main()
