"""Read historical JSON states and final SDF versions; never replay mismatched files.

This does not reconstruct or fix missing historical asset bytes. It records the
limits of what can be used as a counterfactual rollout starting state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET


def main(repo,stage,out_name):
    out=stage/out_name;out.mkdir(parents=True,exist_ok=False)
    prior=repo/'runs/autodl/20261110_complete_native_runtime'
    archive=prior/'evidence_solid_v2/actual_scene_evidence.tar.xz'
    receipt=json.loads((prior/'evidence_solid_v2/receipt.json').read_text())
    mapping=None;needed={};data={}
    with tarfile.open(archive,'r|xz') as stream:
        for member in stream:
            if member.name=='paths.json':
                mapping=json.load(stream.extractfile(member))
                for record in mapping['files']:
                    path=record['path']
                    if path=='last_actual_room_state.json' or path.endswith('.sdf') or re.search(r'/rollout_furniture/state_\d+\.json$',path):
                        needed.setdefault(record['sha256'],[]).append(path)
            elif member.name.startswith('blobs/') and member.name[6:] in needed:
                payload=stream.extractfile(member).read();digest=hashlib.sha256(payload).hexdigest()
                assert digest==member.name[6:]
                for path in needed[digest]:data[path]=payload
    assert mapping is not None and all(path in data for values in needed.values() for path in values)
    final_state=json.loads(data['last_actual_room_state.json'])
    final_objects=final_state['objects']
    rows=[];states=[]
    for path in sorted(data):
        if not re.search(r'/rollout_furniture/state_\d+\.json$',path):continue
        state=json.loads(data[path]);objects=state['objects'];count=0;mismatch=0
        for oid,obj in objects.items():
            if obj.get('object_type')!='furniture' or not obj.get('sdf_path'):continue
            raw=str(obj['sdf_path']);count+=1
            if raw.startswith(mapping['original_root']+'/'):
                target=raw[len(mapping['original_root'])+1:]
            else:target=str(Path('scene/room_bedroom')/raw)
            payload=data.get(target)
            if payload is None:
                rows.append(dict(state=path,object_id=oid,status='referenced_SDF_missing',path=raw));continue
            root=ET.fromstring(payload)
            scales={tuple(float(v) for v in e.text.split()) for e in root.findall('.//geometry/mesh/scale')}
            logical=float(obj.get('scale_factor',1.))
            final=final_objects.get(oid)
            final_logical=float(final.get('scale_factor',1.)) if final else None
            # Scene JSON stores the logical factor, while Drake loads the SDF
            # mesh scales. Comparing two historical logical versions is invariant
            # to any original baked-in asset scale; an absent explicit scale is
            # not itself an inconsistency.
            version_conflict=final_logical is not None and abs(logical-final_logical)>1e-8
            mismatch+=int(version_conflict)
            rows.append(dict(state=path,object_id=oid,logical_scale=logical,
                final_archived_logical_scale=final_logical,
                final_archived_SDF_path=target,final_archived_SDF_sha256=hashlib.sha256(payload).hexdigest(),
                SDF_explicit_mesh_scales=sorted(scales),
                explicit_scale_absence_is_not_a_conflict=not bool(scales),
                known_scale_version_conflict=version_conflict,
                historical_exact_asset_version_proved=False,
                earlier_file_version_reconstructed=False))
        states.append(dict(state=path,furniture_objects=count,scale_mismatches=mismatch))
    result=dict(status='historical_JSON_and_final_asset_versions_compared_without_replay',
        source_archive=str(archive),archive_hash_reference='Original persisted SHA256 in receipt; selected member bytes checked in this pass.',
        state_snapshots=len(states),states=states,objects_checked=len(rows),
        known_scale_version_conflicts=sum(x.get('known_scale_version_conflict') is True for x in rows),records=rows,
        consequence='A historical JSON plus the final mutable SDF archive is not automatically an exact historical scene. Keep old observations as observations; do not assign newly replayed rewards or expert counterfactual labels without matched asset versions.',
        no_historical_reward_or_positive_label_created=True,source_files_unchanged=True,
        model_forwards=0,training_updates=0,engine_queries=0,paid_API_calls=0)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','states')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out-name',default='historical_asset_versions_v2')
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve(),a.out_name)
