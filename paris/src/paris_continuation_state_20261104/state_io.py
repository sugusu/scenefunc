"""Persist the bounded native floor-placement bridge, not a physics-ready room.

Native SceneObject serialization is used for both templates and placed instances.
The unchanged frozen shell is a referenced input; absolute asset paths plus hashes
are intentional because this is a same-workspace restart, not a portable dataset.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FORMAT = 'paris_floor_bridge_state_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def install_path_helper(G):
    # Load the existing pinned native helper; do not change its function body.
    path = WORK / 'native_state_and_supervision_20261104/source/scenesmith/utils/path_utils.py'
    spec = importlib.util.spec_from_file_location('native_checkpoint_paths', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    G['safe_relative_path'] = module.safe_relative_path


def save(path, scene, registry, asset_records, history, frozen_shell, G):
    install_path_helper(G)
    templates = [obj.to_dict() for obj in registry.list_all()]
    instances = [obj.to_dict() for obj in scene.objects.values()]
    # The first-turn adapter has not built full SDFs, surfaces, or physics state.
    if any(obj['support_surfaces'] or obj['sdf_path'] for obj in templates + instances):
        raise ValueError('This checkpoint contract only admits the bounded floor bridge')
    geometry_hashes = {}
    for obj in templates + instances:
        if obj['geometry_path']:
            geometry_hashes[obj['geometry_path']] = sha(obj['geometry_path'])
    payload = {
        'format': FORMAT,
        'room_id': scene.room_id,
        'room_type': scene.room_type,
        'text_description': scene.text_description,
        'surface_id_counter': scene._surface_id_counter,
        'frozen_shell': {'path': str(Path(frozen_shell).resolve()), 'sha256': sha(frozen_shell)},
        'asset_templates': templates,
        'placed_instances': instances,
        'asset_records': asset_records,
        'tool_history': history,
        'cumulative_tool_calls': len(history),
        'geometry_hashes': geometry_hashes,
        'physics_ready': False,
        'complete_scene_success': None,
    }
    document = {'payload': payload, 'payload_sha256': hashlib.sha256(canonical(payload).encode()).hexdigest()}
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return document


def restore(path, scene, registry, frozen_shell, G):
    """Validate every referenced file before committing any restored state."""
    document = json.loads(Path(path).read_text())
    state = document['payload']
    if hashlib.sha256(canonical(state).encode()).hexdigest() != document['payload_sha256']:
        raise ValueError('Checkpoint payload checksum mismatch')
    if state['format'] != FORMAT:
        raise ValueError('Unsupported checkpoint contract')
    if sha(frozen_shell) != state['frozen_shell']['sha256']:
        raise ValueError('Frozen shell changed')
    if state['room_id'] != scene.room_id:
        raise ValueError('Room identity changed')
    if state['cumulative_tool_calls'] != len(state['tool_history']):
        raise ValueError('Tool history count mismatch')
    for file, expected in state['geometry_hashes'].items():
        if sha(file) != expected:
            raise ValueError('Checkpoint asset content changed: ' + file)
    groups = [state['asset_templates'], state['placed_instances']]
    for group in groups:
        if len({o['object_id'] for o in group}) != len(group):
            raise ValueError('Duplicate identifiers in checkpoint')
        for o in group:
            if o['support_surfaces'] or o['sdf_path']:
                raise ValueError('Unsupported physics/surface state')
            if o['geometry_path'] not in state['geometry_hashes']:
                raise ValueError('Missing asset content hash')
    templates = [G['SceneObject'].from_dict(o) for o in groups[0]]
    instances = [G['SceneObject'].from_dict(o) for o in groups[1]]
    if scene.objects or registry.list_all():
        raise ValueError('Restore requires an empty target; no silent merge')
    for obj in templates:
        registry.register(obj)
    for obj in instances:
        scene.add_object(obj)
    scene._surface_id_counter = state['surface_id_counter']
    scene.room_type = state['room_type']
    scene.text_description = state['text_description']
    return state['asset_records'], state['tool_history']
