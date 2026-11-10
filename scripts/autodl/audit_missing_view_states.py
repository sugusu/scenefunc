"""Resolve all missing-view states through archived image/state bindings.

Only source metadata is recovered. No image/geometry substitute, pose repair,
new supervision, model call or training is admitted by this audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


PIN='75fc783f86d4fea78714639e110fbe77476ee106'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(x):return digest(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def object_index(value):
    """Legacy AIR contexts serialize the same records as a list, not a map."""
    if isinstance(value,dict):return value
    if not isinstance(value,list):raise ValueError('Unknown object serialization')
    result={}
    for obj in value:
        ident=obj.get('object_id')
        if not isinstance(ident,str) or ident in result:
            raise ValueError('Missing or duplicate object identity')
        result[ident]=obj
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'states').mkdir()
    start=time.monotonic();reads={};cache={};trees={}
    def get(p):
        if p not in cache:
            raw=subprocess.check_output(['git','-c','gc.auto=0','show',PIN+':'+p]);reads[p]=digest(raw);cache[p]=json.loads(raw)
        return cache[p]
    def tree(root):
        if root not in trees:trees[root]=subprocess.check_output(['git','-c','gc.auto=0','ls-tree','-r','--name-only',PIN+':'+root],text=True).splitlines()
        return trees[root]
    pix=json.loads(Path('runs/autodl/20261110_exact_view_recovery/pixel_v1/pixel_equivalence.json').read_text())
    cov=json.loads(Path('runs/autodl/20261110_transfer_controls/actual_v3/coverage.json').read_text())
    byunit={u['unit_id']:u for u in cov['units']};rows=[]
    fields=['object_id','object_type','name','immutable','bbox_min','bbox_max','transform','scale_factor']
    for unit in pix['material']:
        if unit['complete_five_views']:continue
        uid=unit['unit_id'];u=byunit[uid];input=get(u['input_path']);assert reads[u['input_path']]==cov['source_sha256'][u['input_path']]
        pairs={(im['path'],im['sha256']) for im in unit['images']}
        run=unit['images'][0]['path'].split('/runtime/')[1].split('/')[0];root='runs/920/'+run
        step=int(re.search(r'step_(\d+)',unit['images'][0]['path']).group(1))
        candidates=[];files=tree(root)
        for name in files:
            if not name.endswith('state_binding.json'):continue
            short=Path(name).name
            if not (f'step_{step:03d}' in short or f'step_{step}_' in short or (step==0 and short=='initial_state_binding.json')):continue
            path=root+'/'+name;b=get(path)
            if {(im['path'],im['sha256']) for im in b.get('images',[])}!=pairs:continue
            state=b.get('native_state');assert state is not None and canonical(state)==b['state_sha256']
            candidates.append((path,state,'five_image_hashes_and_native_state_binding',b['state_sha256']))
        if not candidates:
            # AIR publishes a state-file receipt and the five image hashes;
            # other legacy records may instead link the state via tool history.
            names=[f'step_{step}_state.json',f'native/step_{step:02d}_state.json']
            if step==0:names+=['initial_state.json','native/initial_state.json']
            names+=['after_state.json']
            for name in names:
                if name not in files:continue
                p=root+'/'+name;state=get(p);h=canonical(state);proof=None
                if input['history'] and input['history'][-1].get('after_sha256')==h:
                    proof='exact_current_input_history_state_hash; images declared in same frozen input'
                if name=='after_state.json' and 'AFTER_VIEWS.json' in files:
                    v=get(root+'/AFTER_VIEWS.json')
                    if reads[p]==v['state']['sha256'] and {(im['path'],im['sha256']) for im in v['images']}==pairs:
                        proof='joint_state_file_and_five_image_receipt'
                if proof:candidates.append((p,state,proof,h))
        unique={h:(p,s,proof,h) for p,s,proof,h in candidates}
        record={'unit_id':uid,'family_id':unit['family_id'],'input_path':u['input_path'],
            'input_sha256':reads[u['input_path']],'run':run,'source_step':step,
            'missing_images':[im for im in unit['images'] if not im['available_with_explicit_equivalence']],
            'state_candidates':len(candidates),'distinct_states':len(unique)}
        if len(unique)==1:
            path,state,proof,h=next(iter(unique.values()));objects=object_index(state['objects']);visible=object_index(input['context']['objects'])
            mismatches=[]
            if set(objects)!=set(visible):mismatches.append({'field':'object_ids','state_only':sorted(set(objects)-set(visible)),'context_only':sorted(set(visible)-set(objects))})
            compared=0
            for oid,vo in visible.items():
                if oid not in objects:continue
                for field in fields:
                    if field not in vo:continue
                    compared+=1
                    if vo[field]!=objects[oid].get(field):mismatches.append({'object_id':oid,'field':field,'context':vo[field],'state':objects[oid].get(field)})
            dest=a.out/'states'/(uid+'.json');write(dest,state)
            record.update(status='resolved' if not mismatches else 'context_state_mismatch',state_source=path,
                state_canonical_sha256=h,state_proof=proof,object_count=len(objects),context_fields_compared=compared,
                context_mismatches=mismatches,local_state=str(dest),
                geometry_dependencies=sorted({o[k] for o in objects.values() for k in ['geometry_path','sdf_path'] if o.get(k)}),
                room_sdf_dependency=state['room_geometry'].get('sdf_path'),
                exact_native_bundle_available=False,exact_observation_available=False)
        else:record['status']='unresolved_or_ambiguous_state'
        rows.append(record);write(a.out/'units.json',rows)
        print(uid,record['status'],flush=True)
    write(a.out/'summary.json',{'pin':PIN,'missing_units':len(rows),
        'resolved_states':sum(x['status']=='resolved' for x in rows),
        'context_mismatches':sum(x['status']=='context_state_mismatch' for x in rows),
        'source_files':reads,'state_object_records':sum(x.get('object_count',0) for x in rows),
        'seconds':time.monotonic()-start,'new_model_calls':0,'new_views':0,'new_scenes':0,
        'training_ready':False,'scope':'Metadata closure only. State equality does not establish mesh/material/render equivalence.'})


if __name__=='__main__':main()
