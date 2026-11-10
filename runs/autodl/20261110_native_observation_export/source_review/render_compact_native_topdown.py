"""Render a new top view of an unchanged saved native scene, with unique names."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from native_project import import_project, sha256
from native_checkpoint import canonical_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=False)
    print(datetime.now(timezone.utc).isoformat(),'begin unchanged native topdown review render',flush=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    for pth,expected in plan['source_hashes'].items():
        if sha256(Path(pth))!=expected:raise ValueError('Input changed')
    sys.path.insert(0,plan['designer'])
    os.environ['SCENESMITH_ROOT']=plan['scenesmith']
    os.environ.update(SCENE_RENDER_THREADS='4',SCENE_RENDER_SAMPLES='8')
    from src import runtime
    from src_critic.blender_render import main as render
    from src.render_observation import annotate_live_topdown
    project=import_project(Path(plan['bundle']),a.out/'room_copy')
    project.room._scenefunc_structure_labels=True
    state_sha=canonical_hash(project.room.to_state_dict())
    if state_sha!=plan['native_room_state_sha256']:raise ValueError('Room state differs from captured blend binding')
    output=a.out/'step_000.topdown_compact.png'
    bounds=plan['room_box']
    sys.argv=['render_top','--blend',plan['blend'],'--out',str(output),'--resolution','1024','--room-box',json.dumps(bounds)]
    render()
    annotate_live_topdown(project.room,output)
    annotations=json.loads(Path(str(output)+'.annotations.json').read_text())
    if len(annotations['display_label_to_native_object_id'])!=21:raise ValueError('Not all furniture labels mapped')
    for pth,expected in plan['source_hashes'].items():
        if sha256(Path(pth))!=expected:raise ValueError('Input mutated')
    result={'status':'rendered_for_review','image':str(output),'image_sha256':sha256(output),
            'native_room_state_sha256':state_sha,'source_blend_sha256':sha256(Path(plan['blend'])),
            'label_count':21,'original_views_unchanged':True,'scene_edits':0,'model_calls':0,
            'label_map':annotations['display_label_to_native_object_id'],'visual_review':'pending'}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'labels':21,'source_state_unchanged':True}),flush=True)


if __name__=='__main__':main()
