"""Bounded local first-turn dispatcher. Native schema and floor placement;
metadata retrieval instead of external asset generation. Unsupported operations
are failures, never represented as completed actions or physical validation.
"""
import argparse, hashlib, importlib.util, json, math, re, shutil, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
HERE=Path(__file__).resolve().parent
W=HERE.parent;P=W/'paris_A_inference_20261104'
from typed_retrieval import rank as typed_rank
sys.path.insert(0,str(W/'paris_asset_contract_20261104'));from asset_frame import frame_for
sys.path.insert(0,str(W/'paris_continuation_state_20261104'));import state_io
sys.path.insert(0,str(W/'native_room_bridge_20261104'));import native_kernels as N
sys.path.insert(0,str(W/'paris_native_editing_20261104'));import native_editing
G=N.G
spec=importlib.util.spec_from_file_location('stage_schema',W/'professional_data_ingest_20261104/native_src/stage_schemas.py');schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
T=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def tokens(s):return set(re.findall(r'[a-z]+',s.lower()))-{'a','the','with','and','for','of','in','on','small','large','modern','wood','wooden'}
def shell_meshes(initial):
    result=trimesh.Scene()
    for obj in initial['room_structure']:
        w,h,d=obj['dimensions'];holes=obj.get('holes',[]);ang=-math.radians(obj.get('facing',0));tf=np.eye(4);tf[:3,:3]=[[math.cos(ang),-math.sin(ang),0],[math.sin(ang),math.cos(ang),0],[0,0,1]];tf[:3,3]=T@obj['location']
        xs=sorted(set([-w/2,w/2]+[q['location'][0]+sgn*q['dimensions'][0]/2 for q in holes for sgn in (-1,1)]));zs=sorted(set([-h/2,h/2]+[q['location'][1]+sgn*q['dimensions'][1]/2 for q in holes for sgn in (-1,1)]))
        for a,b in zip(xs,xs[1:]):
            for c,e in zip(zs,zs[1:]):
                x,z=(a+b)/2,(c+e)/2
                if not(-w/2<=x<=w/2 and -h/2<=z<=h/2):continue
                if any(abs(x-q['location'][0])<q['dimensions'][0]/2 and abs(z-q['location'][1])<q['dimensions'][1]/2 for q in holes):continue
                mesh=trimesh.creation.box(extents=[b-a,d,e-c]);mesh.apply_translation([x,0,z]);mesh.apply_transform(tf);mesh.visual.face_colors=list(bytes.fromhex(obj['color'][1:]))+[255]
                result.add_geometry(mesh,node_name=obj['name']+'_'+str(len(result.geometry)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--calls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    t=time.monotonic();initial=json.loads((P/'frozen_initial.json').read_text());floor=next(o for o in initial['room_structure'] if o['name']=='floor')
    geo=G['RoomGeometry'](sdf_tree=ET.ElementTree(ET.Element('sdf')),sdf_path=out/'not_physics_ready.sdf',length=floor['dimensions'][0],width=floor['dimensions'][2],wall_height=2.5,wall_thickness=.02)
    scene=G['RoomScene'](room_geometry=geo,scene_dir=out,room_id='paris_19_remote_studio',text_description='Frozen Paris development shell',action_log_path=out/'native_action_log.json');registry=G['AssetRegistry']();native=N.make_tools(scene,registry,{})
    native.update(native_editing.make_tools(scene,registry,N))
    catalog=json.loads((P/'local_asset_catalog.json').read_text());events=[];registered={};asset_records=[];history=[]
    if args.resume:asset_records,history=state_io.restore(args.resume,scene,registry,P/'frozen_initial.json',G)
    initial_asset_count=len(asset_records);initial_object_count=len(scene.objects)
    # Common immutable converted geometry is shared across arms. Exact paths are
    # retained, avoiding repeat material conversion and unnecessary mesh copies.
    pool={}
    for old in [P/'gpu_A_first/tools/asset_records.json',P/'gpu_B_first/tools/asset_records.json',HERE/'corrected_A_assets/asset_records.json',HERE/'corrected_B_assets/asset_records.json']:
        if old.exists():
            for record in json.loads(old.read_text()):pool.setdefault(record['asset_id_source'],record)
    measure_cache={}
    def measure(row):
        if row['id'] in measure_cache:return measure_cache[row['id']]
        frame=frame_for(row)
        if frame is None:raise ValueError('No explicit orientation metadata')
        m=trimesh.load(row['path'],force='mesh');u=np.array(frame['up'],dtype=float);u/=np.linalg.norm(u);f=np.array(frame['front'],dtype=float);f-=u*np.dot(f,u);f/=np.linalg.norm(f)
        tf=np.eye(4);tf[:3,:3]=np.column_stack([np.cross(f,u),f,u]).T;m.apply_transform(tf)
        measure_cache[row['id']]={'dimensions':m.extents.tolist()}
        return measure_cache[row['id']]
    def generate(a):
        desc,names,dims=a['object_descriptions'],a['short_names'],a['desired_dimensions']
        if not (len(desc)==len(names)==len(dims)<=24):return {'success':False,'error':'Equal-length lists and at most 24 requests required for this bounded first turn'}
        values=[];missing=[]
        for i,(description,name,wanted) in enumerate(zip(desc,names,dims)):
            if not isinstance(description,str) or not isinstance(name,str) or len(wanted)!=3 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in wanted):missing.append({'index':i,'error':'invalid request'});continue
            ranking=typed_rank(description,name,wanted,catalog,measure)
            ranked=[(-r['dimension_log_error'],r['row']['id'],r['row']) for r in ranking['candidates']]
            matches=ranked
            if not ranked:
                missing.append({'index':i,'error':'Matching meshes have no explicit orientation metadata; not safe for this placement adapter' if matches else 'No eligible cached mesh for the resolved category; no substitute fabricated','matching_cached_meshes':len(matches),'orientation_eligible':0,'candidates':[{'id':r[2]['id'],'name':r[2]['name'],'path':r[2]['path'],'score':r[0]} for r in matches[:3]]})
                continue
            score,_,row=ranked[0];path=Path(row['path'])
            if shutil.disk_usage(out).free<5*1024**3+max(8*1024**2,path.stat().st_size*3):missing.append({'index':i,'error':'5 GiB reserve; cannot persist this candidate'});continue
            try:
                mesh=trimesh.load(path,force='mesh')
                frame=frame_for(row);u=np.array(frame['up']);f=np.array(frame['front']);u/=np.linalg.norm(u);f=f-u*np.dot(f,u);f/=np.linalg.norm(f);right=np.cross(f,u)
                tf=np.eye(4);tf[:3,:3]=np.column_stack([right,f,u]).T
                assert np.allclose(tf[:3,:3]@u,[0,0,1]) and np.allclose(tf[:3,:3]@f,[0,1,0])
                mesh.apply_transform(tf);center=mesh.bounds.mean(axis=0);shift=np.array([-center[0],-center[1],-mesh.bounds[0,2]]);mesh.apply_translation(shift)
                aid=re.sub('[^a-zA-Z0-9_]','_',name)[:40]+'_'+str(len(asset_records))
                reused=pool.get(row['id'])
                if reused and reused['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and reused['coordinate_frame']==frame:
                    dest=Path(reused['geometry_path'])
                    saved=trimesh.load(dest,force='mesh')
                    assert np.allclose(saved.bounds,mesh.bounds,atol=1e-6,rtol=0)
                    mesh=saved
                else:
                    dest=out/(aid+'.glb');mesh.export(dest)
                pool[row['id']]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'coordinate_frame':frame,'geometry_path':str(dest)}
                actual=mesh.extents.tolist();asset=G['SceneObject'](object_id=G['UniqueID'](aid),object_type=G['ObjectType'].FURNITURE,name=name,description=row['name'],transform=G['RigidTransform'](),geometry_path=dest,bbox_min=mesh.bounds[0],bbox_max=mesh.bounds[1],metadata={'original_asset':row['id'],'metadata_based_alignment':True,'coordinate_frame':frame,'semantic_front_verified':False,'functional_quality_verified':False})
                registry.register(asset);registered[aid]=mesh
                r={'object_id':aid,'name':name,'description':row['name'],'width':actual[0],'depth':actual[1],'height':actual[2],'source':'existing HSSD mesh; ordinary category-first size ranking','asset_id_source':row['id'],'actual_name':row['name'],'categories':row['categories'],'requested_dimensions':wanted,'rescaled':False,'semantic_match':'unverified','coordinate_frame':frame,'dimension_log_error':-score,'resolved_type':ranking['resolved_type'],'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'geometry_path':str(dest)}
                asset_records.append(r);values.append(r)
            except Exception as e:missing.append({'index':i,'error':f'{type(e).__name__}: {e}'})
        return {'success':not missing,'assets':values,'missing':missing,'backend':'ordinary category-first retrieval; not original generative asset service'}
    for call in json.loads(args.calls.read_text()):
        name=call.get('name');a=call.get('arguments',{});errors=schema.validate_tool_call('furniture',name,a)
        try:
            if errors:response={'success':False,'error':'schema validation','details':errors}
            elif name=='generate_assets':response=generate(a)
            elif name in ('add_furniture_to_scene_tool','move_furniture_tool','remove_furniture_tool'):response=json.loads(native[name](**a))
            else:response={'success':False,'error':'This bounded first-turn bridge has not implemented '+str(name),'not_a_quality_failure':True}
        except Exception as e:response={'success':False,'error':f'{type(e).__name__}: {e}'}
        events.append({'call':call,'response':response})
    result=shell_meshes(initial)
    for obj in scene.objects.values():
        mesh=trimesh.load(obj.geometry_path,force='mesh');mesh.apply_transform(obj.transform.GetAsMatrix4());result.add_geometry(mesh,node_name=str(obj.object_id))
    result.export(out/'partial_scene.glb');read=trimesh.load(out/'partial_scene.glb',force='scene');assert set(result.graph.nodes_geometry)==set(read.graph.nodes_geometry)
    dump(out/'tool_responses.json',events);dump(out/'asset_records.json',asset_records)
    state_io.save(out/'checkpoint.json',scene,registry,asset_records,history+events,P/'frozen_initial.json',G)
    dump(out/'receipt.json',{'status':'bounded_dispatch_completed' if args.resume else 'first_turn_dispatch_completed','tool_calls':len(events),'cumulative_tool_calls':len(history)+len(events),'assets_loaded':len(asset_records),'new_assets_loaded':len(asset_records)-initial_asset_count,'restored_furniture':initial_object_count,'placed_furniture':len(scene.objects),'scene_nodes':len(read.graph.nodes_geometry),'shell_only':not scene.objects,'elapsed_seconds':time.monotonic()-t,'model_calls':0,'paid_api_calls':0,'physics_validated':False,'complete_room_success':None,'checkpoint_path':str(out/'checkpoint.json'),'resumed_from':str(args.resume.resolve()) if args.resume else None,'native_scope':'original argument validator and direct native floor-placement closure; generic local retrieval adapter; full original service not loaded'})
    print(json.dumps({'tool_calls':len(events),'assets_loaded':len(asset_records),'placed':len(scene.objects)}))

if __name__=='__main__':main()
