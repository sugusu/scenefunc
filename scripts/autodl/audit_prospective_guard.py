"""Ordinary pre-action box guard on a frozen, complete development batch.

Only parent geometry, public preservation requirements and proposed actions
enter predict(). Future receipts are used solely for evaluation. This is a
candidate-filter diagnostic, not a rerun of any policy or a whole-room verifier.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def world_box(obj):
    """Rotate all local-box corners using normalized wxyz; no yaw-only shortcut."""
    w, x, y, z = obj['transform']['rotation_wxyz']
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Invalid quaternion')
    w, x, y, z = (v/norm for v in (w,x,y,z))
    rot = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    points = []
    for p in itertools.product(*zip(obj['bbox_min'], obj['bbox_max'])):
        points.append([sum(rot[i][j]*p[j] for j in range(3))+
                       obj['transform']['translation'][i] for i in range(3)])
    return ([min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)])


def furniture_pairs(objects):
    # The public table has no SDF path. Explicit furniture membership is used,
    # not inferred from hidden post-state metadata. Coverage checked separately.
    bounds = {oid:world_box(o) for oid,o in objects.items() if o['object_type']=='furniture'}
    pairs = []
    for (a,(al,ah)),(b,(bl,bh)) in itertools.combinations(sorted(bounds.items()),2):
        delta = [min(ah[i],bh[i])-max(al[i],bl[i]) for i in range(3)]
        if all(v>1e-6 for v in delta):
            pairs.append({'a':a,'b':b,'overlap_extents_m':delta})
    return pairs


def rectangular_inner_bounds(objects):
    """Fit an axis-aligned rectangle from outer wall boxes, or abstain.

    Interior partitions are excluded by extremal planes. Exactly two opposite
    full-span slabs per axis are required. Names and reference layouts unused.
    """
    walls = [world_box(o) for o in objects.values() if o['object_type']=='wall']
    if len(walls)<4:
        return None
    xslabs=[]; yslabs=[]
    for lo,hi in walls:
        dx,dy=hi[0]-lo[0],hi[1]-lo[1]
        if dx < dy/20: xslabs.append((lo,hi))
        if dy < dx/20: yslabs.append((lo,hi))
    if len(xslabs)<2 or len(yslabs)<2:
        return None
    left=min(xslabs,key=lambda b:b[0][0]);right=max(xslabs,key=lambda b:b[1][0])
    bottom=min(yslabs,key=lambda b:b[0][1]);top=max(yslabs,key=lambda b:b[1][1])
    centers=[(left[0][0]+left[1][0])/2,(bottom[0][1]+bottom[1][1])/2,
             (right[0][0]+right[1][0])/2,(top[0][1]+top[1][1])/2]
    tol=1e-8
    if not all(abs(b[0][1]-centers[1])<tol and abs(b[1][1]-centers[3])<tol for b in (left,right)):
        return None
    if not all(abs(b[0][0]-centers[0])<tol and abs(b[1][0]-centers[2])<tol for b in (bottom,top)):
        return None
    return [left[1][0],bottom[1][1],right[0][0],top[0][1]]


def outside_xy(objects, rect):
    if rect is None:return None
    bad=[]
    for oid,o in objects.items():
        if o['object_type']!='furniture':continue
        lo,hi=world_box(o)
        if lo[0]<rect[0]-1e-8 or lo[1]<rect[1]-1e-8 or hi[0]>rect[2]+1e-8 or hi[1]>rect[3]+1e-8:
            bad.append(oid)
    return sorted(bad)


def predict(parent, call):
    """No after-state, source labels, reference poses, or measured effects."""
    objects=parent['objects']; program=parent['declared_requirements']
    if call is None:return dict(status='unknown',reason='unparsed_action')
    name=call['name'];args=call['arguments']
    if name in ('remove_furniture_tool','add_furniture_to_scene_tool'):
        return dict(status='reject',reason='task_requires_original_instances')
    if name in ('get_current_scene_state','check_facing_tool'):
        return dict(status='allow',reason='read_only_tool')
    if name!='move_furniture_tool':return dict(status='unknown',reason='unsupported_tool')
    oid=args.get('object_id')
    if oid not in objects or oid not in program['movable_ids'] or objects[oid].get('immutable'):
        return dict(status='reject',reason='identity_or_fixed_object_rule')
    if set(args)!=set(('object_id','x','y','yaw')) or any(type(args[k]) not in (int,float) or not math.isfinite(args[k]) for k in ('x','y','yaw')):
        return dict(status='unknown',reason='unsupported_arguments')
    before_pairs=furniture_pairs(objects);rect=rectangular_inner_bounds(objects)
    candidate=copy.deepcopy(objects);a=math.radians(args['yaw'])
    candidate[oid]['transform']={'translation':[args['x'],args['y'],0.0],
                                'rotation_wxyz':[math.cos(a/2),0,0,math.sin(a/2)]}
    after_pairs=furniture_pairs(candidate)
    old={(p['a'],p['b']) for p in before_pairs}
    new=[p for p in after_pairs if (p['a'],p['b']) not in old]
    before_out=outside_xy(objects,rect);after_out=outside_xy(candidate,rect)
    new_out=None if rect is None else sorted(set(after_out)-set(before_out))
    return dict(status='reject' if new or new_out else 'allow' if rect is not None else 'unknown',
                reason='new_endpoint_box_violation' if new or new_out else 'no_new_supported_box_violation',
                new_overlap_pairs=new,new_outside_xy_ids=new_out,rectangular_inner_bounds=rect,
                pre_overlap_pairs=before_pairs,pre_outside_xy_ids=before_out,
                projected_overlap_pairs=after_pairs,projected_outside_xy_ids=after_out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--effects',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False,parents=True)
    start=time.perf_counter()
    paths=[args.effects/x for x in ('source_fragments.jsonl','transitions.jsonl')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fragments=[json.loads(l) for l in paths[0].read_text().splitlines()]
    outcomes={r['id']:r for r in map(json.loads,paths[1].read_text().splitlines())}
    after_ids={};final_hashes={}
    for index,f in enumerate(fragments):
        row=outcomes[f['id']]
        if index+1<len(fragments) and outcomes[fragments[index+1]['id']]['cell']==row['cell']:
            after_ids[f['id']]=set(fragments[index+1]['parent']['objects'])
        else:
            source=row['source_result'].rsplit('/',1)[0]+'/recorded/final_state.json'
            blob=subprocess.check_output(['git','show','75fc783f86d4fea78714639e110fbe77476ee106:'+source])
            state=json.loads(blob)
            canonical=hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            if canonical!=row['after_state_sha256']:raise ValueError('Final state hash differs')
            after_ids[f['id']]=set(state['objects'])
            final_hashes[source]=hashlib.sha256(blob).hexdigest()
    rows=[]
    for f in fragments:
        observed=outcomes[f['id']]
        pred=predict(f['parent'],observed['call'])
        pairs=furniture_pairs(f['parent']['objects'])
        pair_ids=lambda ps:{tuple(sorted((p['a'],p['b']))) for p in ps}
        original=f['before_measurements']['geometry']
        calibrated=pair_ids(pairs)==pair_ids(original['positive_3d_AABB_pairs'])
        outside=outside_xy(f['parent']['objects'],rectangular_inner_bounds(f['parent']['objects']))
        boundary_match=outside==sorted(original['outside_declared_wall_inner_faces'])
        # Outcome fields never enter the prediction above.
        post=f['after_measurements']['geometry']
        new_pairs=sorted(pair_ids(post['positive_3d_AABB_pairs'])-pair_ids(original['positive_3d_AABB_pairs']))
        new_outside=sorted(set(post['outside_declared_wall_inner_faces'])-set(original['outside_declared_wall_inner_faces']))
        new_missing=sorted(set(f['parent']['objects'])-after_ids[f['id']])
        extra_damage=bool(new_pairs or new_outside or new_missing)
        fine_damage=extra_damage or bool(observed['delta']['damaged'])
        projection_agrees=None
        if 'projected_overlap_pairs' in pred:
            projection_agrees=(pair_ids(pred['new_overlap_pairs'])==set(map(tuple,new_pairs))
                               and set(pred['new_outside_xy_ids'] or [])==set(new_outside))
        rows.append(dict(id=f['id'],prediction=pred,
            pre_pair_membership_agrees_with_archived_checker=calibrated,
            pre_xy_boundary_agrees_with_archived_checker=boundary_match,
            actual_native=observed['native_called'],actual_damage=bool(observed['delta']['damaged']),
            actual_improve_no_harm=observed['delta']['scoped_improvement_without_new_harm'],
            actual_new_overlap_pairs=new_pairs,actual_new_outside_ids=new_outside,
            actual_new_missing_ids=new_missing,actual_primitive_or_boolean_damage=fine_damage,
            actual_repair_without_primitive_or_boolean_damage=bool(observed['delta']['repaired']) and not fine_damage,
            projection_change_agrees_with_actual_record=projection_agrees,
            actual_repaired=observed['delta']['repaired'],actual_damaged=observed['delta']['damaged'],
            current_guard_allows=observed['ordinary_preservation_guard_allows'],
            actual_post_geometry=f['after_measurements']['geometry']))
    summary=dict(decisions=len(rows),independent_development_families=1,
        rejected=sum(r['prediction']['status']=='reject' for r in rows),
        allowed=sum(r['prediction']['status']=='allow' for r in rows),
        unknown=sum(r['prediction']['status']=='unknown' for r in rows),
        damage_steps=sum(r['actual_damage'] for r in rows),
        rejected_damage=sum(r['actual_damage'] and r['prediction']['status']=='reject' for r in rows),
        allowed_damage=sum(r['actual_damage'] and r['prediction']['status']=='allow' for r in rows),
        rejected_without_recorded_new_boolean_damage=sum(not r['actual_damage'] and r['prediction']['status']=='reject' for r in rows),
        improve_no_harm_steps=sum(r['actual_improve_no_harm'] for r in rows),
        rejected_improve_no_harm=sum(r['actual_improve_no_harm'] and r['prediction']['status']=='reject' for r in rows),
        pre_pair_matches=sum(r['pre_pair_membership_agrees_with_archived_checker'] for r in rows),
        pre_boundary_matches=sum(r['pre_xy_boundary_agrees_with_archived_checker'] for r in rows),
        primitive_or_boolean_damage_steps=sum(r['actual_primitive_or_boolean_damage'] for r in rows),
        hidden_primitive_damage_steps=sum(r['actual_primitive_or_boolean_damage'] and not r['actual_damage'] for r in rows),
        rejected_primitive_or_boolean_damage=sum(r['actual_primitive_or_boolean_damage'] and r['prediction']['status']=='reject' for r in rows),
        rejected_without_new_primitive_or_boolean_damage=sum(not r['actual_primitive_or_boolean_damage'] and r['prediction']['status']=='reject' for r in rows),
        rejected_already_invalid=sum(not r['actual_native'] and r['prediction']['status']=='reject' for r in rows),
        repaired_without_new_primitive_or_boolean_damage=sum(r['actual_repair_without_primitive_or_boolean_damage'] for r in rows),
        move_projection_matches=sum(r['projection_change_agrees_with_actual_record'] is True for r in rows),
        move_projection_compared=sum(r['projection_change_agrees_with_actual_record'] is not None for r in rows),
        final_state_source_hashes=final_hashes,
        source_sha256=hashes,wall_seconds=time.perf_counter()-start,
        model_calls=0,training_updates=0,native_tool_calls=0,
        scope='Post-hoc evaluation of a prospective ordinary filter on the entire saved batch. No policy rerun, saved-damage causal claim, complete activity prediction, or whole-room performance claim.')
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in paths):raise ValueError('Source changed')
    write(args.output/'summary.json',summary);write(args.output/'rows.json',rows)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
