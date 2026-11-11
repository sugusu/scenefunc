"""Execute/validate allocation programs against complete native room states.

No model, resizing, geometry invention or hidden pose search. This reuses the
two qualified ordinary skills and binds measured reservations to actual native
table poses. Region witnesses are independently checked after transformation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from shapely.geometry import Polygon,shape,mapping
from shapely.ops import unary_union
import trimesh

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scratch',type=Path,required=True);ap.add_argument('--native-source',type=Path,required=True);a=ap.parse_args()
    root=a.root.resolve();scratch=a.scratch.resolve();out=root/'executed_programs'
    out.mkdir(exist_ok=False);start=time.monotonic()
    sys.path.insert(0,str(Path('runs/autodl/20261110_public_function_search/original_evaluator').resolve()))
    sys.path.insert(0,str(a.native_source.resolve()))
    from native_project import validate_bundle,import_project
    from inspect_native_mesh_contacts import sdf_visuals,world_transform
    from activity_resource_model import evaluate
    protocol=read(root/'protocol.json');rows=read(root/'compiled_programs.json')['rows'];models=read(root/'model_probe.json')
    assert models['status'] in ['completed','failed']
    sources={n:scratch/b for n,b in [('shared','bundle'),('independent','completed_bundle')]}
    states={};metadata={};initial=None
    for name,bundle in sources.items():
        m,h,_,_=validate_bundle(bundle);state=h['rooms'][m['selected_room_id']]
        project=import_project(bundle,scratch/('activity_native_'+name))
        restored=project.room.to_state_dict()
        assert restored==state,'Native restore must preserve the full state, not just object count'
        states[name]=state;metadata[name]=dict(bundle_manifest_sha256=sha(bundle/'manifest.json'),
            objects=len(state['objects']),native_state_exact=True,
            actual_native_class=type(project.room).__module__+'.'+type(project.room).__name__)
    initial=states['shared']
    assert all(states['independent']['objects'][k]==v for k,v in initial['objects'].items())
    assert set(states['independent']['objects'])-set(initial['objects'])=={'ordinary_work_table_0','ordinary_work_chair_0'}
    measured=read(root/'measured_table_surface.json');old_top=shape(measured['polygon'])
    old_table=initial['objects']['meal_table_0'];old_T=world_transform(old_table)
    def surface(plan,oid):
        b=sources[plan];m,_,_,_=validate_bundle(b);obj=states[plan]['objects'][oid]
        mesh=trimesh.util.concatenate([x for _,x in sdf_visuals(b/('room_'+m['selected_room_id'])/obj['sdf_path'],b,{})])
        mesh.apply_transform(world_transform(obj));z=mesh.bounds[1,2];tri=mesh.triangles
        sel=(mesh.face_normals[:,2]>.99)&(tri[:,:,2].min(1)>=z-.03)
        return unary_union([Polygon(t[:,:2]) for t in tri[sel] if Polygon(t[:,:2]).area>1e-10])
    def translate_reservation(record,new_table):
        T=world_transform(new_table)@np.linalg.inv(old_T);p=shape(record['polygon']);coords=np.array([[x,y,measured['z'],1] for x,y in p.exterior.coords]);dest=(T@coords.T).T
        return Polygon(dest[:,:2])
    executions=[]
    for row in rows:
        choices=[('ordinary',0,'shared' if row['shared']['status']=='witness' else 'independent')]
        for m in models['attempts']:
            if m['task']==row['id'] and m['feedback']['valid']:choices.append(('local_model',m['turn'],m['feedback']['plan']))
        for policy,turn,plan in choices:
            selected=states[plan];name=f"{row['id']}_{policy}_{turn}"
            if plan=='shared':
                witnesses=row['shared']['witnesses'];regions={k:shape(v['polygon']) for k,v in witnesses.items()}
                supports={k:'meal_table_0' for k in regions}
            else:
                regions={k:shape(v['polygon']) for k,v in row['independent_meal_pair']['witnesses'].items() if k!='persistent_work'}
                regions['persistent_work']=translate_reservation(row['independent_work_witness'],selected['objects']['ordinary_work_table_0'])
                supports={k:'meal_table_0' for k in regions};supports['persistent_work']='ordinary_work_table_0'
            cover={k:bool(surface(plan,supports[k]).buffer(1e-8).covers(v)) for k,v in regions.items()}
            # This 1e-8 tolerance applies only to independent transform/readback
            # validation after the original 1e-6-inset search, not search acceptance.
            pairchecks=[dict(a=x,b=y,overlap_m2=float(regions[x].intersection(regions[y]).area),
                simultaneous=True,same_surface=supports[x]==supports[y]) for x,y in itertools.combinations(regions,2)]
            assert all(cover.values())
            assert all(not c['same_surface'] or c['overlap_m2']<=1e-12 for c in pairchecks)
            original_unchanged=all(selected['objects'][k]==v for k,v in initial['objects'].items())
            assert original_unchanged
            program=dict(task=row['id'],policy=policy,turn=turn,plan=plan,
                reservations={k:dict(surface_id=supports[k],polygon=mapping(v),active_modes=['work','meal'] if k=='persistent_work' else ['meal']) for k,v in regions.items()},
                modes={'work':{'resident':{'seat_id':'meal_chair_n_0' if plan=='shared' else 'ordinary_work_chair_0'}},
                       'meal':{'resident':{'seat_id':'meal_chair_n_0'},'guest':{'seat_id':'meal_chair_s_0'}}},
                correct_simultaneous_people_per_mode={'work':1,'meal':2},people_workspace_not_simultaneous=True,
                original_objects_exact=original_unchanged,covered_by_actual_tabletop=cover,region_pair_checks=pairchecks,
                final_native_objects=len(selected['objects']),new_furniture=0 if plan=='shared' else 2,
                native_state_path=name+'_room_state.json',native_restore=metadata[plan],
                geometry_facts_reference='../20261111_milan_completion/completion_result.json' if plan=='independent' else 'Unchanged original native scene; no new geometry',
                physical_material_objects_created=False,new_full_scene_generation=False,
                full_original_brief_satisfied=None,original_frozen_evaluator_modified=False,
                evidence_scope='Qualified resource reservation and existing complete-scene reconstruction, not ergonomic/user acceptance or new construction search')
            dump(out/(name+'_room_state.json'),selected);dump(out/(name+'_program.json'),program)
            executions.append(program)
    dump(root/'execution_result.json',dict(executions=executions,native_imports=metadata,model_calls=0,training_updates=0,
        reused_scene_placements_not_new_search=True,wall_seconds=time.monotonic()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    assert all(sha(Path(k))==v for k,v in protocol['source_inputs'].items())
    print(json.dumps(dict(programs=len(executions),native_imports=metadata,wall_seconds=time.monotonic()-start)))

if __name__=='__main__':main()
