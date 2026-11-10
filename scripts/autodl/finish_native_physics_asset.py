"""Consume the real VLM reply through native canonical/SDF/registry routines.

Exact source dataclass/parser and AssetManager helper bodies are AST-isolated;
Blender/convex clients, material/inertia/SDF/registry modules and Drake are real.
No full AssetManager constructor, new VLM call or whole-room success is claimed.
"""
import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import types


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--resume-canonical', type=Path)
    a = ap.parse_args()
    stage = a.stage.resolve()
    repo = Path(__file__).resolve().parents[2]
    out = a.out.resolve() if a.out else stage/'asset_v1'
    out.mkdir(exist_ok=False)
    assert shutil.disk_usage(out).free > 5*1024**3 + 24*1024**2
    protocol = read(stage/'asset_protocol.json')
    assert sha(stage/'actual_v1/response.json') == protocol['response_sha256']
    assert read(stage/'actual_v1/process_receipt.json')['returncode'] == 0
    source = repo/'runs/autodl/20261110_native_observation_export/source'
    before = read(repo/'runs/autodl/20261110_native_asset_conversion/actual_v2/input_hashes.json')
    for p, h in before.items():
        assert sha(p) == h
    for r in read(stage/'source_receipt.json'):
        assert sha(stage/'source'/r['path']) == r['sha256']
    sys.path.insert(0, str(source))
    import numpy as np
    import trimesh
    import scenesmith.agent_utils
    import scenesmith.utils
    scenesmith.agent_utils.__path__.append(str(stage/'source/scenesmith/agent_utils'))
    scenesmith.utils.__path__.append(str(stage/'source/scenesmith/utils'))
    from scenesmith.agent_utils.room import SceneObject, ObjectType
    from scenesmith.agent_utils.asset_registry import AssetRegistry
    from scenesmith.agent_utils.blender import BlenderServer
    from scenesmith.agent_utils.mesh_canonicalization import canonicalize_mesh
    from pydrake.all import RigidTransform
    # The exact dataclass and response parser require no SDK constructor.
    analyzer = repo/'runs/autodl/20261110_native_asset_contract/tool_source/mesh_physics_analyzer.py'
    analyzer_ast = ast.parse(analyzer.read_text())
    native_physics = types.ModuleType('scenesmith.agent_utils.mesh_physics_analyzer')
    sys.modules[native_physics.__name__] = native_physics
    g = native_physics.__dict__
    g.update(dataclass=dataclass, np=np, time=time, console_logger=logging.getLogger('native_physics'),
             start_time=time.time())
    names = {'MeshPhysicsAnalysis','get_view_direction_from_image_number','get_front_axis_from_image_number'}
    nodes = [n for n in analyzer_ast.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(analyzer),'exec'),g)
    fn = next(n for n in analyzer_ast.body if isinstance(n,ast.FunctionDef) and n.name=='analyze_mesh_orientation_and_material')
    parse_block = fn.body[-1]
    assert isinstance(parse_block,ast.Try) and any(isinstance(n,ast.Return) for n in ast.walk(parse_block))
    parse_fn = ast.FunctionDef(name='parse_actual_response',args=ast.arguments(
        posonlyargs=[],args=[ast.arg(arg=k) for k in ('response_json','num_side_views','include_vertical_views','prompt_type')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=[parse_block],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[parse_fn],type_ignores=[])),str(analyzer),'exec'),g)
    response = read(stage/'actual_v1/response.json')
    physics = g['parse_actual_response'](response,4,False,'hssd')
    from scenesmith.agent_utils.materials import get_friction, load_materials
    assert physics.material in load_materials(), 'Do not substitute default friction for missing source data'
    low, high = physics.mass_range_kg
    assert all(math.isfinite(x) for x in (physics.mass_kg,low,high)) and 0 < low <= physics.mass_kg <= high
    assert physics.up_axis == '+Z' and physics.front_axis in ('+X','-X','+Y','-Y')
    dump(out/'parsed_physics.json',dict(asdict(physics), source='actual single model response; estimated, not physical ground truth',
        native_parser_lines=[parse_block.lineno,parse_block.end_lineno],friction=get_friction(physics.material)))
    # SDF module imports the actual isolated source dataclass above, not a fake answer.
    sdf_file = repo/'runs/autodl/20261110_native_collision_path/source/scenesmith/agent_utils/sdf_generator.py'
    spec=importlib.util.spec_from_file_location('native_sdf_writer',sdf_file)
    sdf=importlib.util.module_from_spec(spec);sys.modules[spec.name]=sdf;spec.loader.exec_module(sdf)
    input_mesh = Path(protocol['converted_mesh'])
    assert sha(input_mesh) == protocol['converted_mesh_sha256']
    target=out/'asset'/'native_nightstand.gltf'
    server = BlenderServer(port_range=(18220,18230),gpu_id=None,log_file=out/'blender.log')
    result=dict(status='started',new_model_calls=0,paid_api_calls=0,training_updates=0,new_complete_scenes=0,
                full_runtime_ready=False,physical_ground_truth=False)
    start=time.monotonic();collider=None;blog=None
    def deadline(signum,frame):
        raise TimeoutError('240-second asset downstream execution cap')
    old_alarm=signal.signal(signal.SIGALRM,deadline);signal.alarm(240)
    try:
        if a.resume_canonical:
            prior_stage=read(stage/'asset_v1/result.json')
            assert prior_stage['error']=='TypeError: required field "lineno" missing from stmt'
            recovery=read(stage/'downstream_recovery_protocol.json')
            assert sha(a.resume_canonical)==recovery['canonical_mesh_sha256']
            target.parent.mkdir()
            for row in recovery['canonical_resources']:
                src=Path(row['path']);assert sha(src)==row['sha256']
                shutil.copyfile(src,target.parent/src.name)
            assert target.name==a.resume_canonical.name
            result.update(canonicalization_seconds=0,canonicalization_reused=True,
                canonicalization_source=str(a.resume_canonical.resolve()),blender_terminal=True,
                recovery_scope='Continue after wrapper AST location error; no new VLM or canonicalization')
        else:
            server.start();server.wait_until_ready()
            bproc=server._server_process
            dump(out/'blender_process.json',{'pid':bproc.pid,'args':bproc.args})
            t=time.monotonic()
            actual=canonicalize_mesh(gltf_path=input_mesh,output_path=target,
                up_axis=physics.up_axis,front_axis=physics.front_axis,
                blender_server=server,object_type=ObjectType.FURNITURE)
            assert actual == target and target.is_file()
            result['canonicalization_seconds']=time.monotonic()-t
            server.stop()
            result['blender_terminal']=bproc.poll() is not None
        mesh=trimesh.load(target,force='mesh')
        assert isinstance(mesh,trimesh.Trimesh) and len(mesh.faces)>0
        bounds=mesh.bounds
        bbox_min_yup,bbox_max_yup=bounds
        bbox_min=np.array([bbox_min_yup[0],-bbox_min_yup[2],bbox_min_yup[1]])
        bbox_max=np.array([bbox_max_yup[0],-bbox_max_yup[2],bbox_max_yup[1]])
        bbox_min,bbox_max=np.minimum(bbox_min,bbox_max),np.maximum(bbox_min,bbox_max)
        result['native_zup_bounds']=[bbox_min.tolist(),bbox_max.tolist()]
        result['canonical_mesh_faces']=len(mesh.faces)
        collision_stage=repo/'runs/autodl/20261110_native_collision_path'
        from execute_native_collision_path import module
        from omegaconf import OmegaConf
        cfg=OmegaConf.create(read(collision_stage/'collision_configuration.json'))
        client_code=collision_stage/'source/scenesmith/agent_utils/convex_decomposition_server/client.py'
        client_module=module('native_asset_collision_client',client_code)
        port=None
        for candidate in range(18300,18310):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',candidate));port=candidate;break
                except OSError:continue
        assert port is not None
        collision_out=out/'collision';collision_out.mkdir()
        blog=(out/'collision_server.log').open('w')
        collider=subprocess.Popen([sys.executable,'-B',str(repo/'scripts/autodl/execute_native_collision_path.py'),
            '--stage',str(collision_stage),'--out',str(collision_out),'--serve',str(port)],
            stdout=blog,stderr=subprocess.STDOUT,env=dict(os.environ,NO_PROXY='127.0.0.1,localhost'),start_new_session=True)
        dump(out/'collision_process.json',{'pid':collider.pid,'port':port})
        client=client_module.ConvexDecompositionClient(host='127.0.0.1',port=port,timeout=120)
        ready=time.monotonic()
        while not client.health_check():
            assert collider.poll() is None
            if time.monotonic()-ready>20:raise TimeoutError('collision startup')
            time.sleep(.1)
        manager=repo/'runs/autodl/20261110_native_service_contract/inputs/asset_manager.py'
        tree=ast.parse(manager.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AssetManager')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_generate_collision_geometry','_create_scene_object')]
        scope=dict(Path=Path,trimesh=trimesh,SceneObject=SceneObject,RigidTransform=RigidTransform,time=time)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+methods,type_ignores=[])),str(manager),'exec'),scope)
        registry=AssetRegistry(auto_save_path=out/'registry.json')
        owner=types.SimpleNamespace(collision_client=client,collision_method=cfg.method,
            collision_coacd_cfg=cfg.coacd,collision_vhacd_cfg=cfg.vhacd,registry=registry)
        t=time.monotonic();pieces=scope['_generate_collision_geometry'](owner,target)
        result.update(collision_seconds=time.monotonic()-t,collision_pieces=len(pieces))
        assert pieces
        destination=out/'asset'/'native_nightstand.sdf'
        t=time.monotonic();written=sdf.generate_drake_sdf(target,pieces,physics,destination,asset_name='native_nightstand')
        assert written == destination
        result['sdf_seconds']=time.monotonic()-t
        config=types.SimpleNamespace(short_name='native_nightstand',description='wooden nightstand',image_path=None)
        created=scope['_create_scene_object'](owner,config,ObjectType.FURNITURE,destination,target,
            bbox_min,bbox_max,additional_metadata={'asset_source':'hssd','hssd_mesh_id':protocol['asset_id']})
        assert registry.size()==1 and registry.get(created.object_id) is created
        restored=AssetRegistry();restored.load_from_file(out/'registry.json')
        assert restored.size()==1 and restored.get(created.object_id).to_dict()==created.to_dict()
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.geometry import SceneGraph
        from pydrake.multibody.parsing import Parser
        plant=MultibodyPlant(time_step=.001);scene_graph=SceneGraph()
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        t=time.monotonic();models=Parser(plant).AddModels(str(destination));plant.Finalize()
        body=plant.GetBodyByName('base_link',models[0])
        result.update(drake_parse_seconds=time.monotonic()-t,drake_models=len(models),
            drake_collision_geometries=plant.num_collision_geometries(),
            drake_visual_geometries=len(plant.GetVisualGeometriesForBody(body)),
            drake_default_mass_kg=body.default_mass(),
            drake_inertia_physically_valid=body.default_spatial_inertia().IsPhysicallyValid())
        assert len(models)==1 and plant.num_collision_geometries()==len(pieces)
        assert abs(body.default_mass()-physics.mass_kg)<1e-9
        result.update(status='native_estimated_asset_exported_registered_and_parsed',
            registered_object_id=str(created.object_id),registry_roundtrip_exact=True,
            asset_sdf=str(destination),source_and_response_unchanged=sha(stage/'actual_v1/response.json')==protocol['response_sha256'],
            estimated_physics=asdict(physics),friction=get_friction(physics.material),
            scope='Sequential actual native helper/service path on one retrieved asset; AST-isolated parser/dataclass/manager methods. No full AssetManager constructor or full retrieval lifecycle in this call.')
    except BaseException as exc:
        result.update(status='downstream_failure_no_retry',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
        server.stop()
        if collider is not None:
            if collider.poll() is None:
                os.killpg(collider.pid,signal.SIGTERM)
                try:collider.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(collider.pid,signal.SIGKILL);collider.wait()
            result['collision_server_terminal']=collider.poll() is not None
        if blog is not None:blog.close()
        result.update(seconds=time.monotonic()-start,
            old_files_unchanged=all(sha(p)==h for p,h in before.items()),
            parent_peak_rss_mib_only=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            source_parser_sha256=sha(analyzer),native_manager_sha256=sha(manager) if 'manager' in locals() else None)
        dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
