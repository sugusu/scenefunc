"""Bind pinned SceneSmith floor move/remove methods to the existing bridge.

No method body, coordinate convention or final-state test is replaced here.
This is the same zero-noise direct-call scope as native_kernels: no SDK server,
loop detector, SDF construction or collision validation is implied.
"""
import ast
import copy
from pathlib import Path


def make_tools(scene, registry, native):
    source = native.S / 'furniture_agents/tools/furniture_tools.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FurnitureTools')
    selected = ('_move_furniture_impl', '_remove_furniture_impl')
    functions = [copy.deepcopy(n) for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name in selected]
    assert {n.name for n in functions} == set(selected)
    for node in functions:
        assert len(node.decorator_list) == 1
        assert isinstance(node.decorator_list[0], ast.Name)
        assert node.decorator_list[0].id == 'log_scene_action'
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), native.G)
    owner_class = native.G['FurnitureTools']
    for node in functions:
        setattr(owner_class, node.name, native.G[node.name])
        native.READ.append({'path': str(source), 'method': node.name,
                            'start': node.lineno, 'end': node.end_lineno,
                            'body_modified': False, 'decorator_retained': 'log_scene_action'})
    owner = owner_class()
    owner.scene = scene
    owner.asset_manager = native.G['AssetManager']()
    owner.asset_manager.registry = registry
    owner.active_noise_profile = type('ZeroNoise', (), {
        'position_xy_std_meters': 0., 'rotation_yaw_std_degrees': 0.})()
    return {name: native.public_callable(owner, 'furniture_agents/tools/furniture_tools.py', name)
            for name in ('move_furniture_tool', 'remove_furniture_tool')}
