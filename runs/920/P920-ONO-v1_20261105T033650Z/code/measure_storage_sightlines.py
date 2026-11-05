"""Conditional opaque-box sightline and approach diagnostics for saved states.

Shell openings come from actual cut SDF visuals, not uncut wall bounding boxes.
This is not a daylight, spaciousness, storage-capacity or professional score.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from inspect_native_mesh_contacts import sdf_visuals, world_transform
from measure_room_access import measure
from native_project import sha256


def segment_hit(p, q, lower, upper):
    p, q = np.asarray(p), np.asarray(q)
    lo, hi = 1e-7, 1-1e-7
    for axis in range(3):
        d = q[axis]-p[axis]
        if abs(d) < 1e-12:
            if p[axis] < lower[axis] or p[axis] > upper[axis]:
                return None
        else:
            a, b = sorted(((lower[axis]-p[axis])/d, (upper[axis]-p[axis])/d))
            lo, hi = max(lo, a), min(hi, b)
            if lo > hi:
                return None
    return float(lo)


def ray_checks():
    lo, hi = np.array([-1., -1., -1.]), np.array([1., 1., 1.])
    assert abs(segment_hit([-2,0,0],[2,0,0],lo,hi)-.25) < 1e-12
    assert segment_hit([-2,2,0],[2,2,0],lo,hi) is None
    assert segment_hit([-2,0,0],[-1.1,0,0],lo,hi) is None
    assert segment_hit([0,0,0],[2,0,0],lo,hi) == 1e-7
    # Parallel grazing is conservatively blocked; a positive gap is clear.
    assert segment_hit([-2,1,0],[2,1,0],lo,hi) is not None
    assert segment_hit([-2,1.00001,0],[2,1.00001,0],lo,hi) is None
    return 6


def occluders(state, room, root, screen_id, screen_height=None):
    result = []
    for oid, obj in state['objects'].items():
        if obj['object_type'] == 'wall' or obj.get('metadata', {}).get('structural_role') == 'opening':
            continue
        if obj['bbox_min'] is None or obj['bbox_max'] is None:
            continue
        lo, hi = np.array(obj['bbox_min'], dtype=float), np.array(obj['bbox_max'], dtype=float)
        if oid == screen_id and screen_height is not None:
            hi[2] = lo[2]+screen_height
        result.append((oid, np.linalg.inv(world_transform(obj)), lo, hi))
    # Exclude door/window leaves: the frozen metric assumes open doors and
    # unobstructed glazing; retain the actual wall pieces around each opening.
    for name, mesh in sdf_visuals(room/state['room_geometry']['sdf_path'], root, {}):
        if 'wall' in name:
            result.append(('shell::'+name, np.eye(4), mesh.bounds[0], mesh.bounds[1]))
    return result


def visibility(observer, targets, boxes, ignore=()):
    observer = np.asarray(observer, dtype=float); rows = []
    for target in targets:
        hits = []
        for oid, inverse, lo, hi in boxes:
            if oid in ignore:
                continue
            p = (inverse @ np.r_[observer, 1])[:3]
            q = (inverse @ np.r_[target, 1])[:3]
            t = segment_hit(p, q, lo, hi)
            if t is not None:
                hits.append((t, oid))
        hits.sort()
        rows.append({'target_xyz_m':list(map(float,target)), 'clear':not hits,
                     'first_occluder':hits[0][1] if hits else None})
    return {'clear_rays':sum(r['clear'] for r in rows), 'total_rays':len(rows),
            'first_occluder_counts':dict(Counter(r['first_occluder'] for r in rows if not r['clear'])),
            'rays':rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text()); definition = plan['definition']
    print(datetime.now(timezone.utc).isoformat(), 'saved-state opaque sightline checks; no scene edits',flush=True)
    for name,digest in plan['scripts'].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise ValueError('Frozen script changed')
    tests = ray_checks(); w = definition['window']; cx,cy,cz=w['center']
    window = [[x,cy,z] for x in np.linspace(cx-w['width_m']/2+.01,cx+w['width_m']/2-.01,w['samples_x'])
              for z in np.linspace(cz-w['height_m']/2+.01,cz+w['height_m']/2-.01,w['samples_z'])]
    rows = {}
    for label,spec in plan['states'].items():
        path=Path(spec['path'])
        if sha256(path)!=spec['sha256']:raise ValueError('State changed')
        state=json.loads(path.read_text());room,root=Path(spec['room']),Path(spec['root'])
        oid=definition['storage_object_id'];obj=state['objects'][oid];matrix=world_transform(obj)
        lo,hi=np.array(obj['bbox_min']),np.array(obj['bbox_max'])
        front=[(matrix@np.array([x,hi[1],z,1]))[:3] for x in np.linspace(lo[0]+.01,hi[0]-.01,definition['storage_front_samples_x'])
               for z in np.linspace(lo[2]+.01,hi[2]-.01,definition['storage_front_samples_z'])]
        boxes=occluders(state,room,root,definition['screen_object_id'])
        views={name:{'window':visibility(eye,window,boxes),
                     'storage_front':visibility(eye,front,boxes,ignore=(oid,)),
                     'observer_is_on_front_side':bool(np.dot(np.asarray(eye)-matrix[:3,3],matrix[:3,1])>hi[1])}
               for name,eye in definition['observers'].items()}
        access=[]
        for body in definition['body_widths_m']:
            target=(matrix@np.array([0,hi[1]+body/2+.04,0,1]))[:2]
            settings={**plan['access_settings'],'square_body_widths_m':[body],
                      'approach_points_xy_m':{**plan['access_settings']['approach_points_xy_m'],'storage_current_front':target.tolist()}}
            access.append(measure(state,room,root,settings)['scenarios'][0])
        sensitivity=[]
        for height in definition['screen_height_sensitivity_m']:
            boxes_cf=occluders(state,room,root,definition['screen_object_id'],height)
            for eye_height in definition['eye_height_sensitivity_m']:
                sensitivity.append({'screen_height_m':height,'eye_height_m':eye_height,
                    'views':{name:{'window':visibility([eye[0],eye[1],eye_height],window,boxes_cf),
                                   'storage_front':visibility([eye[0],eye[1],eye_height],front,boxes_cf,ignore=(oid,))}
                             for name,eye in definition['observers'].items()}})
        rows[label]={'views':views,'access':access,'sensitivity':sensitivity,
                     'storage_envelope_m3':float(np.prod(hi-lo)), 'storage_bbox_min':lo.tolist(),
                     'storage_bbox_max':hi.tolist(),'storage_object_id':oid,
                     'opaque_object_and_shell_boxes':len(boxes)}
    result={'status':'conditional_sightlines_and_access_measured','created_at':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':sha256(a.plan),'analytic_ray_checks_passed':tests,'states':rows,
            'native_edits':0,'model_calls':0,'training_updates':0,'professional_preference':'unknown',
            'limits':definition['limits']+['Same external envelope is not verified storage capacity.','Door leaves/glazing excluded under declared open-door/aperture assumptions.','Height variants are analytic counterfactuals on saved states, not additional executed trials.']}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({label:{name:{key:{'clear':value[key]['clear_rays'],'total':value[key]['total_rays']} for key in ['window','storage_front']} for name,value in data['views'].items()} for label,data in rows.items()}),flush=True)


if __name__=='__main__':main()
