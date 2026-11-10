"""Replay original deterministic placement utility on all four saved add poses.

This calls no scene manager, SDK dispatcher, asset generator or collision tool.
It tests a possible explanation of report/request differences, not scene quality.
The original function and original natural-profile constants are unmodified.
"""
import argparse
import ast
import hashlib
import json
import logging
import math
from pathlib import Path
import time


def main(repo,out):
    tick=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    root=repo/'runs/autodl/20261110_live_manager_request/source'
    src=root/'scenesmith/agent_utils/placement_noise.py'
    cfg=root/'configurations/furniture_agent/base_furniture_agent.yaml'
    import yaml
    import numpy as np
    from pydrake.all import RigidTransform,RollPitchYaw
    profile=yaml.safe_load(cfg.read_text())['placement_noise']['natural_profile']
    tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_placement_noise')
    namespace=dict(np=np,math=math,RigidTransform=RigidTransform,RollPitchYaw=RollPitchYaw,
                   console_logger=logging.getLogger('native_placement_source_replay'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
    packets=json.loads((repo/'runs/autodl/20261111_closed_loop_data/feedback_material_v2/offline_feedback_packets.json').read_text())
    results=[]
    for packet in packets:
        for index,(call,feedback) in enumerate(zip(packet['source_tool_calls'],packet['original_feedback'])):
            if call['name']!='add_furniture_to_scene_tool':continue
            args=call['arguments'];reported=feedback['output'];position=reported['position'];rotation=reported['rotation']
            # Native add path uses world meters and input/output degrees.
            # All selected saved add outputs have reported z=0. We do not infer
            # shape-bottom offsets for other assets or generalize this replay.
            assert position['z']==0.0
            rpy=RollPitchYaw(*[math.radians(args.get(k,0.)) for k in ['roll','pitch','yaw']])
            original=RigidTransform(rpy=rpy,p=[args['x'],args['y'],0.])
            transformed=namespace['apply_placement_noise'](original,**profile)
            after=transformed.translation();after_rpy=RollPitchYaw(transformed.rotation())
            predicted=dict(position=dict(zip(['x','y','z'],map(float,after))),
                           rotation=dict(zip(['roll','pitch','yaw'],[math.degrees(v) for v in after_rpy.vector()])))
            differences=[abs(predicted['position'][k]-position[k]) for k in ['x','y','z']]+[abs(predicted['rotation'][k]-rotation[k]) for k in ['roll','pitch','yaw']]
            results.append(dict(source_id=packet['id'],call_index=index,arguments=args,
                reported_position=position,reported_rotation=rotation,predicted=predicted,
                requested_reported_XY_delta_m={k:position[k]-args[k] for k in ['x','y']},
                maximum_absolute_component_error=max(differences),matched_at_1e_minus_12=max(differences)<1e-12,
                stage_or_full_scene_correctness=None))
    assert len(results)==4
    # Do not abort or drop a mismatch; an unsuccessful explanation remains data.
    record=dict(status='original_deterministic_noise_utility_replayed_on_all_saved_add_poses',
        cases=results,matched=sum(r['matched_at_1e_minus_12'] for r in results),total=len(results),
        original_natural_profile=profile,
        deterministic_transform_seed_observed_in_source=True,
        fresh_independent_randomness_claim=False,
        no_inference_of_unobservable_teacher_information_from_pose_delta=True,
        full_add_tool_or_geometry_executed=False,original_utility_executions=len(results),
        native_SDK_calls=0,new_model_generations=0,paid_API_calls=0,training_updates=0,
        source_function_lines=[node.lineno,node.end_lineno],
        source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),seconds=time.monotonic()-tick)
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='cases'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.out.resolve())
