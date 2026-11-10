"""Execute selected ordinary-search moves through pinned native move kernels.

Restores original native records with original RoomScene/SceneObject classes.
Only unused SDK/service bootstrap is bypassed; move method and public closure
bodies are compiled unchanged. Zero placement noise is explicit. Missing mesh
resources are reported: this is NOT a complete native session, render or physics
check, and never substitutes the search's nominal coordinates as tool output.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

from audit_public_function_preview import dump
from public_function_preview import functional_checks
from search_public_function import primitive, activity_keys


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--search',type=Path,required=True);ap.add_argument('--native-source',type=Path,required=True)
    ap.add_argument('--state-source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    manifests=[json.loads((p/'manifest.json').read_text()) for p in (args.native_source,args.state_source)]
    files=[Path(__file__),args.search/'initial_public_inputs.json',args.native_source/'original_initial_state.json']
    for root,manifest in zip((args.native_source,args.state_source),manifests):
        if manifest['pin']!='e888297a8b0a2f3acbb8634edf9def59cceab836':raise ValueError('Wrong native source pin')
        for r in manifest['files']:
            if r.get('status','downloaded')!='downloaded':continue
            p=root/r['path']
            if sha(p)!=r['sha256']:raise ValueError('Native source hash mismatch')
            files.append(p)
    sys.path.insert(0,str(args.state_source.resolve()))
    import scenesmith.agent_utils
    scenesmith.agent_utils.__path__.append(str((args.native_source/'agent_utils').resolve()))
    from scenesmith.agent_utils.room import RoomScene, UniqueID
    from scenesmith.agent_utils.action_logger import log_scene_action
    from scenesmith.agent_utils.placement_noise import apply_placement_noise
    from pydrake.all import RigidTransform, RollPitchYaw
    response=load(args.native_source/'furniture_agents/tools/response_dataclasses.py','frozen_native_responses')
    source=args.native_source/'furniture_agents/tools/furniture_tools.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FurnitureTools')
    methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_check_floor_bounds','_move_furniture_impl')]
    if len(methods)!=2:raise ValueError('Expected two unmodified native methods')
    closure_owner=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_create_tool_closures')
    closure=copy.deepcopy(next(n for n in closure_owner.body if isinstance(n,ast.FunctionDef) and n.name=='move_furniture_tool'))
    if [ast.unparse(n) for n in closure.decorator_list]!=['function_tool']:raise ValueError('SDK wrapper changed')
    closure.decorator_list=[]
    factory=ast.FunctionDef(name='bind',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),
                           body=[closure,ast.Return(value=ast.Name(id='move_furniture_tool',ctx=ast.Load()))],decorator_list=[])
    selected=ast.ClassDef(name='FurnitureTools',bases=[],keywords=[],body=methods,decorator_list=[])
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    ns={'np':np,'math':math,'UniqueID':UniqueID,'RigidTransform':RigidTransform,'RollPitchYaw':RollPitchYaw,
        'log_scene_action':log_scene_action,'apply_placement_noise':apply_placement_noise,'console_logger':logging.getLogger('native-move')}
    ns.update({name:getattr(response,name) for name in ('FurnitureOperationResult','FurnitureErrorType','Position3D','Rotation3D')})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,selected,factory],type_ignores=[])),str(source),'exec'),ns)
    inputs=json.loads((args.search/'initial_public_inputs.json').read_text())
    original=json.loads((args.native_source/'original_initial_state.json').read_text())
    files.extend(args.search/(k[:12]+'.json') for k in inputs)
    before={str(p):sha(p) for p in files};dump(args.out/'input_hashes_before.json',before)
    rows=[];calls=0
    for key,item in inputs.items():
        found=json.loads((args.search/(key[:12]+'.json')).read_text())['solution']
        if found is None:continue
        parent=item['parent'];geometry=item['geometry'];mode=parent['requested_mode']
        folder=args.out/mode;folder.mkdir()
        scene=RoomScene(room_geometry=None,scene_dir=folder.resolve(),room_id=geometry['room_id'])
        scene.restore_from_state_dict(original);scene.action_log_path=folder/'native_actions.json'
        restored=scene.to_state_dict()
        def public(state):
            return {oid:{k:state['objects'][oid][k] for k in template}
                    for oid,template in parent['objects'].items()}
        if set(restored['objects'])!=set(parent['objects']):raise ValueError('Original initial object IDs differ')
        actual=public(restored)
        for oid in actual:
            for k,v in parent['objects'][oid].items():
                a=actual[oid][k]
                if k=='transform':
                    if not all(np.allclose(a[x],v[x],atol=1e-12,rtol=0) for x in a):raise ValueError('Initial pose differs')
                elif a!=v:raise ValueError('Original public object metadata differs')
        dump(folder/'before_state.json',restored)
        missing=[{'object_id':str(o.object_id),'field':field,'path':str(path)}
                 for o in scene.objects.values() for field in ('geometry_path','sdf_path')
                 if (path:=getattr(o,field)) is not None and not path.is_file()]
        owner=ns['FurnitureTools']();owner.scene=scene
        owner.active_noise_profile=SimpleNamespace(position_xy_std_meters=0.,rotation_yaw_std_degrees=0.)
        call_native=ns['bind'](owner);receipts=[]
        for call in found['actions']:
            tick=time.monotonic();result=json.loads(call_native(**call['arguments']));calls+=1
            receipts.append({'call':call,'result':result,'seconds':time.monotonic()-tick})
            if result.get('success') is not True:raise RuntimeError('Native selected move failed; no automatic retry')
        post=scene.to_state_dict();dump(folder/'after_state.json',post);dump(folder/'tool_receipts.json',receipts)
        observed=copy.deepcopy(parent);observed['objects']=public(post)
        functional=functional_checks(observed['objects'],geometry,parent['declared_requirements'],mode)
        moved={r['call']['arguments']['object_id'] for r in receipts}
        non_target_unchanged=all(post['objects'][oid]==restored['objects'][oid] for oid in restored['objects'] if oid not in moved)
        if not non_target_unchanged:raise ValueError('Native move changed a non-target record')
        initial_p=primitive(parent,geometry);final_p=primitive(observed,geometry)
        new={k:sorted(final_p[k]-initial_p[k]) for k in final_p}
        pose_matches=all(np.allclose(observed['objects'][oid]['transform'][k],found['state']['objects'][oid]['transform'][k],atol=1e-12,rtol=0)
                         for oid in observed['objects'] for k in ('translation','rotation_wxyz'))
        lost=[k for k,v in functional.items() if parent['checks'].get(k) is True and v is not True]
        rows.append({'mode':mode,'native_move_kernel_calls':len(receipts),'all_tool_success':True,
                     'all_object_ids_preserved':set(post['objects'])==set(restored['objects']),
                     'non_target_records_unchanged':non_target_unchanged,'observed_poses_match_selected_nominal':pose_matches,
                     'activity_pass':all(functional[k] for k in activity_keys(mode)),
                     'lost_initial_true_functional_checks':lost,'new_primitive_violations':new,
                     'remaining_failed_functional_checks':[k for k,v in functional.items() if v is False],
                     'missing_geometry_resource_references':missing,'full_scene_success':None})
    after={str(p):sha(p) for p in files};dump(args.out/'input_hashes_after.json',after)
    if before!=after:raise ValueError('Old inputs changed')
    dump(args.out/'summary.json',{'status':'native_kernel_execution_completed','results':rows,
        'native_move_kernel_calls':calls,'model_calls':0,'training_updates':0,'new_scene_generations':0,
        'full_scene_success':None,'full_native_session_executed':False,'mesh_or_physics_validated':False,
        'placement_noise':'Explicit zero-noise kernel test, not historical noise-profile restoration',
        'source_files_unchanged':before==after,'wall_seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Original state records, original native classes and unchanged move/closure bodies. No SDK wrappers, loop-detector bootstrap, renderer, resources restored or physics. Scalar checks remain the shared proxies.'})
    print(json.dumps({'status':'completed','native_move_kernel_calls':calls,'model_calls':0}),flush=True)


if __name__=='__main__':main()
