"""Nine declared AABB counterfactuals, without pretending they were native replays."""
import argparse
import copy
import itertools
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from native_project import sha256
from inspect_native_mesh_contacts import world_transform


def envelopes(state):
    result={}
    for oid,o in state['objects'].items():
        if o['sdf_path'] is None:continue
        points=np.array(list(itertools.product(*zip(o['bbox_min'],o['bbox_max']))))
        points=np.c_[points,np.ones(len(points))]@world_transform(o).T
        result[oid]=(points[:,:3].min(0),points[:,:3].max(0))
    return result


def reposition(state,length,width,after):
    s=copy.deepcopy(state);dx=(length-5.2)/2;dy=(width-3)/2
    for oid,o in s['objects'].items():
        p=o['transform']['translation']
        if oid in ('recon_wardrobe_south_0','recon_folded_bed_cabinet_0','recon_wardrobe_north_0'):p[0]-=dx
        if oid in ('recon_tv_storage_0','recon_bookcase_0','recon_tv_0','recon_table_0','recon_stool_1_0','recon_stool_2_0'):p[0]+=dx
        if oid in ('recon_wardrobe_south_0','recon_tv_storage_0','recon_tv_0','recon_table_0'):p[1]-=dy
        if oid in ('recon_wardrobe_north_0','recon_bookcase_0'):p[1]+=dy
        if after and oid=='recon_sofa_0':p[1]+=dy
    return s


def evaluate(state,length,width):
    region=box(-length/2+.65,-.8,-length/2+2.6,.6);walls=box(-length/2+.01,-width/2+.01,length/2-.01,width/2-.01)
    env=envelopes(state);hits=[];out=[]
    for oid,(lo,hi) in env.items():
        shape=box(lo[0],lo[1],hi[0],hi[1])
        if hi[2]>.05 and lo[2]<1.8 and region.intersection(shape).area>1e-9:
            hits.append((oid,region.intersection(shape)))
        if not walls.buffer(1e-8).covers(shape):out.append(oid)
    pairs=[]
    for (a,(alo,ahi)),(b,(blo,bhi)) in itertools.combinations(env.items(),2):
        z=min(ahi[2],bhi[2])-max(alo[2],blo[2])
        if z<=1e-6:continue
        area=box(alo[0],alo[1],ahi[0],ahi[1]).intersection(box(blo[0],blo[1],bhi[0],bhi[1])).area
        if area>1e-9:pairs.append({'a':a,'b':b,'planar_overlap_m2':area,'vertical_overlap_m':float(z)})
    occupied=unary_union([h[1] for h in hits]).area
    return {'reserved_bounds_xyxy_m':list(region.bounds),'reserved_occupied_m2':occupied,
        'reserved_unoccupied_m2':region.area-occupied,'reserved_clear':occupied<1e-9,
        'reserved_hits':[{'id':oid,'area_m2':shape.area} for oid,shape in hits],
        'objects_outside_interior':out,'AABB_overlap_pairs':pairs}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());states={}
    print(datetime.now(timezone.utc).isoformat(),'begin fixed nine-case AABB sensitivity',flush=True)
    for label,item in plan['states'].items():
        f=Path(item['path'])
        if sha256(f)!=item['sha256']:raise ValueError('State changed')
        states[label]=json.loads(f.read_text())
    cases=[]
    for length,width in itertools.product(plan['lengths_m'],plan['widths_m']):
        entry={'length_m':length,'width_m':width,'native_execution':length==5.2 and width==3}
        for label,state in states.items():entry[label]=evaluate(reposition(state,length,width,label=='after'),length,width)
        entry['reserved_unoccupied_gain_m2']=entry['after']['reserved_unoccupied_m2']-entry['initial']['reserved_unoccupied_m2']
        cases.append(entry)
    result={'status':'nine_analytic_counterfactuals_measured','created_at':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha256(a.plan),'cases':cases,'native_replays':1,'analytic_only_cases':8,
        'model_calls':0,'tool_calls':0,'training_updates':0,
        'scope':'Keep actual asset bounds and actual native pose drift; translate fixed proxies by the frozen wall-anchor rules. Only central dimensions were natively executed. Grid routing, images and physical handling are not re-evaluated for counterfactuals.'}
    with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(cases),'after_reserved_clear':sum(x['after']['reserved_clear'] for x in cases),'positive_area_gain':sum(x['reserved_unoccupied_gain_m2']>0 for x in cases)}),flush=True)


if __name__=='__main__':main()
