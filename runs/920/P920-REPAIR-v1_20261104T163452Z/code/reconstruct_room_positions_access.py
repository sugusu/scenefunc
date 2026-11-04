"""Apply a declared researcher input reconstruction, never a policy trajectory.

Copies the frozen initial bundle, changes only planned positions and support
translations, preserves all identity/geometry bytes, and imports the new native
state. The source input and every prior session/checkpoint remain untouched.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from native_project import import_project, sha256, validate_bundle


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    source=Path(plan['source_bundle'])
    print(datetime.now(timezone.utc).isoformat(),'begin declared whole-room input position reconstruction',flush=True)
    if sha256(source/'manifest.json')!=plan['source_manifest_sha256']:
        raise ValueError('Frozen source changed')
    manifest,state,runtime,task=validate_bundle(source)
    if 'session' in manifest['inputs']:
        raise ValueError('Must reconstruct an input, not discard a session history')
    rid=manifest['selected_room_id']
    native=state['rooms'][rid]
    previous=copy.deepcopy(native)
    edits=[]
    for oid,spec in plan['positions'].items():
        obj=native['objects'][oid]
        if obj['sdf_path'] is None:
            raise ValueError('Structural changes are outside this plan')
        before=np.asarray(obj['transform']['translation'])
        if not np.allclose(before,spec['expected_translation'],atol=1e-12,rtol=0):
            raise ValueError('Planned source position differs: '+oid)
        after=np.asarray(spec['translation'],dtype=float)
        if after.shape!=(3,) or not np.isfinite(after).all():
            raise ValueError('Invalid position')
        delta=after-before
        obj['transform']['translation']=after.tolist()
        rotation_before=copy.deepcopy(obj['transform']['rotation_wxyz'])
        if 'rotation_wxyz' in spec:
            if obj['support_surfaces'] or any(v.get('metadata',{}).get('support_parent')==obj['name']
                                            for v in native['objects'].values()):
                raise ValueError('Rotation of a support parent is outside this reconstruction contract')
            rotation=np.asarray(spec['rotation_wxyz'],dtype=float)
            if rotation.shape!=(4,) or not np.isclose(np.linalg.norm(rotation),1,atol=1e-12,rtol=0):
                raise ValueError('Invalid explicit orientation')
            obj['transform']['rotation_wxyz']=rotation.tolist()
        for surface in obj['support_surfaces']:
            surface['transform']['translation']=(np.asarray(surface['transform']['translation'])+delta).tolist()
        edits.append({'object_id':oid,'asset_id':runtime['rooms'][rid]['asset_bindings'][oid]['asset_id'],
                      'before_translation_m':before.tolist(),'after_translation_m':after.tolist(),
                      'before_rotation_wxyz':rotation_before,'after_rotation_wxyz':obj['transform']['rotation_wxyz'],
                      'translation_distance_m':float(np.linalg.norm(delta)),'reason':spec['reason'],
                      'actor':'researcher_input_reconstruction','professional_target':False})
    for oid,obj in native['objects'].items():
        before=previous['objects'][oid]
        for key in set(before)-{'transform','support_surfaces'}:
            if obj[key]!=before[key]:
                raise ValueError('Unexpected non-position change')
        if obj['transform']['rotation_wxyz']!=before['transform']['rotation_wxyz'] and 'rotation_wxyz' not in plan['positions'].get(oid,{}):
            raise ValueError('Orientation changed')
        parent=obj.get('metadata',{}).get('support_parent')
        if obj.get('placement_info') and parent:
            parent_id='recon_'+parent+'_0'
            child_delta=np.asarray(obj['transform']['translation'])-np.asarray(before['transform']['translation'])
            parent_delta=np.asarray(native['objects'][parent_id]['transform']['translation'])-np.asarray(previous['objects'][parent_id]['transform']['translation'])
            if not np.allclose(child_delta,parent_delta,atol=1e-12,rtol=0):
                raise ValueError('Supported object did not follow its parent')
    args.out.mkdir(parents=True,exist_ok=False)
    bundle=args.out/'bundle'
    shutil.copytree(source,bundle)
    dump(bundle/manifest['inputs']['native_state'],state)
    task['unknowns'].append(plan['visible_input_disclosure'])
    dump(bundle/manifest['inputs']['task'],task)
    provenance=json.loads((bundle/'provenance.json').read_text())
    if 'input_reconstruction' in provenance:
        provenance.setdefault('input_reconstruction_history',[]).append(provenance['input_reconstruction'])
    provenance['input_reconstruction']={'plan_sha256':sha256(args.plan),
        'parent_manifest_sha256':plan['source_manifest_sha256'],'actor':'researcher_not_model_or_historical_designer',
        'purpose':'Prepare a complete development input; remove diagnosed import conflicts without deleting functions.',
        'edits':edits,'training_eligible':False,'activity_layout_optimized':False}
    for obj in provenance['objects']:
        obj['previous_actual_position_m']=obj['actual_position_m']
        obj['actual_position_m']=native['objects'][obj['object_id']]['transform']['translation']
        obj['actual_rotation_wxyz']=native['objects'][obj['object_id']]['transform']['rotation_wxyz']
    dump(bundle/'provenance.json',provenance)
    manifest['project']['project_id']=plan['project_id']
    manifest['project']['provenance']='researcher_reconstructed_and_position_corrected_generated_development_input_not_historical'
    for rel in manifest['files']:
        file=bundle/rel
        manifest['files'][rel]={'bytes':file.stat().st_size,'sha256':sha256(file)}
    dump(bundle/'manifest.json',manifest)
    validate_bundle(bundle)
    changed=[rel for rel in manifest['files'] if sha256(source/rel)!=sha256(bundle/rel)]
    allowed=[manifest['inputs']['native_state'],manifest['inputs']['task'],'provenance.json']
    if set(changed)!=set(allowed):
        raise ValueError('Unexpected bundle resource change')
    sys.path.insert(0,plan['scenesmith'])
    restored=import_project(bundle,args.out/'native_import')
    current=restored.room.to_state_dict()
    if set(current['objects'])!=set(previous['objects']):
        raise ValueError('Native object inventory changed')
    for oid,obj in current['objects'].items():
        if not np.allclose(obj['transform']['translation'],native['objects'][oid]['transform']['translation'],atol=1e-12,rtol=0):
            raise ValueError('Native import changed position')
    result={'status':'declared_input_reconstruction_built_and_imported','bundle':str(bundle),
        'manifest_sha256':sha256(bundle/'manifest.json'),'position_corrections':len(edits),'edits':edits,
        'furniture_objects':21,'native_objects':len(current['objects']),'geometry_resource_files_unchanged':True,
        'input_house_geometry_unchanged':native['room_geometry']==previous['room_geometry'],
        'source_manifest_unchanged':sha256(source/'manifest.json')==plan['source_manifest_sha256'],
        'geometry_acceptance':'pending_measurement','usability':'unknown','professional_sample_admitted':False,
        'model_calls':0,'policy_tool_calls':0,'training_updates':0}
    dump(args.out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','position_corrections','native_objects','manifest_sha256')}),flush=True)


if __name__=='__main__':
    main()
