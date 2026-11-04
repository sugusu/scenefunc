"""Build an explicitly assumed Geneva development shell and portable catalog.

No published furniture coordinates or professional rationale are read. Existing
asset resources are reused without their room poses; this is not a design output.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

from build_reference_room import (add, box, numbers, prepare_mesh, save_sdf, sdf_root, write)
from native_project import SCHEMA, sha256, validate_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text())
    for item in plan['inputs'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Input changed')
    for filename, digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/filename) != digest:
            raise ValueError('Construction code changed')
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, plan['scenesmith'])
    from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix
    from scenesmith.agent_utils.house import HouseLayout, HouseScene, RoomGeometry, RoomSpec, PlacedRoom, ClearanceOpeningData
    from scenesmith.agent_utils.room import RoomScene, SceneObject, ObjectType, UniqueID, SupportSurface
    draft = json.loads(Path(plan['inputs']['builder']['path']).read_text())
    shell = draft['shell']; assumptions = plan['geometry_assumptions']
    length, width, height = shell['width_m'], shell['depth_m'], assumptions['height_m']
    rid = 'geneva_development'; bundle = a.out/'bundle'; roomdir = bundle/('room_'+rid)
    roomdir.mkdir(parents=True)
    root, link = sdf_root('geneva_development_shell', 'room_geometry_body_link')
    box(link, 'floor', [0,0,-0.025], [length,width,0.05], [0.72,0.64,0.54,1])
    box(link, 'ceiling', [0,0,height+0.025], [length,width,0.05], [1,1,1,1], visual=False)
    walls, normals, bindings, openings, opening_objects = [], {}, {}, [], []
    door = shell['door']; glass = shell['possible_glazing']
    cuts = {
        'east': [door['center_y_from_south_m']-width/2-door['width_m']/2,
                 door['center_y_from_south_m']-width/2+door['width_m']/2, 0, assumptions['door_height_m']],
        'west': [glass['start_y_from_south_m']-width/2, glass['end_y_from_south_m']-width/2,
                 assumptions['window_sill_m'], assumptions['window_sill_m']+assumptions['window_height_m']],
    }
    for name, axis, fixed, span, normal in [
        ('north',0,width/2,length,[0,-1]), ('south',0,-width/2,length,[0,1]),
        ('east',1,length/2,width,[-1,0]), ('west',1,-length/2,width,[1,0]),
    ]:
        cut = cuts.get(name); uc = sorted(set([-span/2,span/2]+(cut[:2] if cut else [])))
        zc = sorted(set([0,height]+(cut[2:] if cut else [])))
        for i,(lo,hi) in enumerate(zip(uc,uc[1:])):
            for j,(bottom,top) in enumerate(zip(zc,zc[1:])):
                u,z = (lo+hi)/2,(bottom+top)/2
                if cut and cut[0]<u<cut[1] and cut[2]<z<cut[3]:
                    continue
                center=[0,0,z];center[axis]=u;center[1-axis]=fixed
                size=[0.05,0.05,top-bottom];size[axis]=hi-lo
                box(link,f'{name}_{i}_{j}',center,size,[0.85,0.85,0.85,1])
        center=[0,0,height/2];center[1-axis]=fixed
        size=np.array([0.05,0.05,height]);size[axis]=span
        oid='shell_'+name
        wall=SceneObject(UniqueID(oid),ObjectType.WALL,name+' wall','Source-informed shell with explicit vertical assumptions',
            RigidTransform(center),bbox_min=-size/2,bbox_max=size/2,immutable=True)
        walls.append(wall);normals[oid]=np.array(normal,dtype=float)
        bindings[oid]={'asset_id':'procedural_shell:'+name,'identity_provenance':'researcher_development_assumption_not_survey'}
        if cut:
            kind='door' if name=='east' else 'window';u=(cut[0]+cut[1])/2;z=(cut[2]+cut[3])/2
            center=[fixed,u,z];size=np.array([0.015,cut[1]-cut[0],cut[3]-cut[2]])
            if kind=='window':box(link,'assumed_window',center,size,[0.55,0.75,0.9,0.22],collision=False)
            opening=SceneObject(UniqueID('shell_'+kind),ObjectType.WALL_MOUNTED,kind,
                'Assumed vertical size; no operability or daylight validation',RigidTransform(center),
                bbox_min=-size/2,bbox_max=size/2,immutable=True,metadata={'structural_role':'opening'})
            opening_objects.append(opening);bindings['shell_'+kind]={'asset_id':'procedural_opening:'+kind,'identity_provenance':'researcher_development_assumption_not_survey'}
            openings.append(ClearanceOpeningData('shell_'+kind,kind,name,center,cut[1]-cut[0],cut[2],cut[3]-cut[2],
                None,None,[fixed,cut[0]],[fixed,cut[1]],(cut[1]-cut[0])/2))
    sdf=roomdir/'room.sdf';save_sdf(sdf,root)
    floor=SceneObject(UniqueID('shell_floor'),ObjectType.FLOOR,'floor','Development floor',RigidTransform(),immutable=True,
        support_surfaces=[SupportSurface(UniqueID('S_0'),np.array([-length/2,-width/2,0]),np.array([length/2,width/2,0]),RigidTransform())])
    geometry=RoomGeometry(ET.parse(sdf),sdf,walls=walls,floor=floor,wall_normals=normals,length=length,width=width,
        wall_height=height,wall_thickness=0.05,openings=openings)
    room=RoomScene(geometry,roomdir,room_id=rid,room_type='studio',text_description='Geneva development shell; geometry assumptions are not surveyed facts.')
    for obj in walls+opening_objects:room.add_object(obj)
    room._surface_id_counter=1
    catalog=[]
    source=Path(plan['inputs']['resource_manifest']['path']).parent
    oldmanifest,oldstate,oldruntime,_=validate_bundle(source);oldrid=oldmanifest['selected_room_id'];oldroom=source/('room_'+oldrid)
    for oid in plan['reuse_object_resources']:
        template=copy.deepcopy(oldstate['rooms'][oldrid]['objects'][oid]);binding=oldruntime['rooms'][oldrid]['asset_bindings'][oid]
        if binding['representation']!='mesh':raise ValueError('Only real mesh resources may be reused')
        aid=binding['asset_id'];reg='catalog_'+aid
        for relative in [Path(template['sdf_path']).parent,Path(template['geometry_path']).parent]:
            shutil.copytree(oldroom/relative,roomdir/relative,dirs_exist_ok=True)
        template.update(object_id=reg,transform={'translation':[0.,0.,0.],'rotation_wxyz':[1.,0.,0.,0.]},
            support_surfaces=[],placement_info=None,immutable=False,image_path=None,
            metadata={'catalog_registry_id':reg,'asset_source':'hssd_catalog'},
            description='Fixed unscaled HSSD geometry; floor placement only; suitability not accepted.')
        catalog.append({'template':template,'binding':{'asset_id':aid,'identity_provenance':'catalog_template_not_historical_instance'},
            'limitations':['No articulation, support-surface edits, installation or functional validation.']})
    download=json.loads(Path(plan['inputs']['fixtures']['path']).read_text());groups={}
    for item in download['files']:groups.setdefault(item['asset_id'],{})[item['role']]=item
    for aid,items in groups.items():
        frame=json.loads(Path(items['coordinate_frame']['local_path']).read_text());up=np.array(frame['up'],dtype=float);up/=np.linalg.norm(up)
        front=np.array(frame['front'],dtype=float);front-=up*np.dot(up,front);front/=np.linalg.norm(front)
        rotation=np.column_stack([np.cross(front,up),front,up]).T
        # Explicitly recorded visual orientation correction; never a size change.
        correction=plan.get('fixture_front_correction_degrees',{}).get(aid,0.0)
        theta=math.radians(correction)
        rotation=np.array([[math.cos(theta),-math.sin(theta),0],
                           [math.sin(theta),math.cos(theta),0],[0,0,1]]) @ rotation
        spec={'asset_id':aid,'source_path':items['mesh']['local_path'],'sha256':items['mesh']['sha256'],'source_to_native_rotation':rotation.tolist()}
        visual,bounds,gltf,sdf_rotation,shift=prepare_mesh(spec,roomdir/'assets'/aid)
        objdir=roomdir/'objects'/aid;objdir.mkdir(parents=True);xml,body=sdf_root('catalog_'+aid)
        node=add(body,'visual',name='source_asset');rpy=RollPitchYaw(RotationMatrix(sdf_rotation)).vector()
        add(node,'pose',numbers([*shift,*rpy]));add(add(add(node,'geometry'),'mesh'),'uri',os.path.relpath(gltf,objdir))
        box(body,'collision_envelope',bounds.mean(axis=0),bounds[1]-bounds[0],[1,1,1,1],visual=False)
        asset_sdf=objdir/'object.sdf';save_sdf(asset_sdf,xml);reg='catalog_'+aid
        name=plan['fixture_names'][aid]
        obj=SceneObject(UniqueID(reg),ObjectType.FURNITURE,name,'Fixed actual fixture geometry; static candidate, functionality unknown.',
            RigidTransform(),geometry_path=visual,sdf_path=asset_sdf,bbox_min=bounds[0],bbox_max=bounds[1],
            metadata={'catalog_registry_id':reg,'asset_source':'hssd_catalog',
                'front_correction_degrees':correction,'source_to_native_rotation':rotation.tolist()})
        catalog.append({'template':obj.to_dict(scene_dir=roomdir),'binding':{'asset_id':aid,'identity_provenance':'catalog_template_not_historical_instance'},
            'limitations':['No lid/door articulation or plumbing simulation; enclosure use and privacy unverified.']})
    if len(catalog)!=plan['budget']['catalog_entries']:raise ValueError('Catalog count differs from plan')
    layout=HouseLayout(house_dir=bundle,house_prompt=draft['brief']);layout.room_specs.append(RoomSpec(rid,room_type='studio',width=width,length=length))
    layout.placed_rooms.append(PlacedRoom(rid,(0,0),length,width));layout.room_geometries[rid]=geometry
    house=HouseScene(layout);house.add_room(room)
    write(bundle/'house_state.json',house.to_state_dict());write(bundle/'runtime.json',{'rooms':{rid:{'room_type':'studio','next_surface_id':1,'asset_bindings':bindings}}})
    write(bundle/'catalog.json',{'schema':'scenefunc-native-catalog-v1','entries':catalog})
    write(bundle/'task.json',{'design_request':draft['brief'],'original_request':None,'request_provenance':'adapted',
        'unknowns':draft['unknowns']+[assumptions,'Catalog lacks a verified cooking appliance and privacy partitions; no whole-room completion yet.']})
    write(bundle/'provenance.json',{'plan_sha256':sha256(a.plan),'geometry_assumptions':assumptions,
        'source':plan['inputs']['builder'],'geneva_reference_furniture_coordinates_read':False,'training_eligible':False,
        'fixture_front_correction_degrees':plan.get('fixture_front_correction_degrees',{}),
        'resource_reuse':'Only mesh/SDF resources and template geometry; old room poses, demands and labels excluded.'})
    files={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(bundle.rglob('*')) if p.is_file()}
    write(bundle/'manifest.json',{'schema':SCHEMA,'coordinates':{'units':'metres','up':'+Z','frame':'native_room_local'},
        'project':{'project_id':'geneva_catalog_development_v1','family_id':'geneva_freaks_2014','provenance':'source_informed_development_shell_with_researcher_assumptions'},
        'selected_room_id':rid,'inputs':{'native_state':'house_state.json','runtime_metadata':'runtime.json','task':'task.json','asset_catalog':'catalog.json'},'files':files})
    validate_bundle(bundle)
    size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
    if size>plan['budget']['bundle_max_bytes']:raise ValueError('Bundle byte budget exceeded')
    write(a.out/'RESULT.json',{'status':'shell_and_catalog_prepared_not_furnished_room','native_objects':len(room.objects),'furniture_instances':0,
        'catalog_entries':len(catalog),'bundle_bytes':size,'manifest_sha256':sha256(bundle/'manifest.json'),'model_calls':0,'training_updates':0})
    print('Prepared development shell and catalog',len(catalog),size,flush=True)


if __name__=='__main__':main()
