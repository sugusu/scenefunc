"""Replay both actual first model requests against the same ordinary correction.
No model outputs, task text, original assets, or original tool logs are modified.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import trimesh

P=Path(__file__).resolve().parent
W=P.parent
sys.path.insert(0,str(W/'paris_asset_contract_20261104'))
from asset_frame import frame_for
from typed_retrieval import rank, label


def main():
    start=time.monotonic()
    source=W/'paris_A_inference_20261104'
    catalog=json.loads((source/'local_asset_catalog.json').read_text())
    cache={}
    def measure(row):
        if row['id'] in cache:return cache[row['id']]
        frame=frame_for(row)
        if frame is None:raise ValueError('No explicit orientation metadata')
        mesh=trimesh.load(row['path'],force='mesh')
        up=np.array(frame['up'],dtype=float);up/=np.linalg.norm(up)
        front=np.array(frame['front'],dtype=float);front-=up*np.dot(front,up);front/=np.linalg.norm(front)
        right=np.cross(front,up)
        tf=np.eye(4);tf[:3,:3]=np.column_stack([right,front,up]).T
        mesh.apply_transform(tf)
        record={'dimensions':mesh.extents.tolist(),'frame':frame,
                'source_sha256':hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()}
        cache[row['id']]=record
        return record
    records=[];flat=[]
    for arm in ['A','B']:
        run=source/f'gpu_{arm}_first'
        calls=json.loads((run/'parsed_tool_calls.json').read_text())
        events=json.loads((run/'tools/tool_responses.json').read_text())
        for call,event in zip(calls,events):
            assert call==event['call'] and call['name']=='generate_assets'
            arguments=call['arguments']
            old={asset['name']:asset for asset in event['response']['assets']}
            for name,description,dimensions in zip(arguments['short_names'],arguments['object_descriptions'],arguments['desired_dimensions']):
                result=rank(description,name,dimensions,catalog,measure)
                selected=result['candidates'][0] if result['candidates'] else None
                # Type agreement is metadata-level; never call it human correctness.
                old_asset=old.get(name)
                original_agreement=bool(old_asset and set(result['resolved_type']['categories']).intersection(label(c) for c in old_asset['categories']))
                record={'arm':arm,'request_name':name,'description':description,
                        'requested_dimensions':dimensions,'original':old_asset,
                        'original_metadata_type_agrees':original_agreement,
                        'corrected':result}
                records.append(record)
                flat.append({'arm':arm,'requested_name':name,'original_asset':old_asset['actual_name'] if old_asset else '',
                    'original_metadata_type_agrees':original_agreement,
                    'resolved_type':','.join(result['resolved_type']['categories']),
                    'selected_asset':selected['row']['name'] if selected else '',
                    'selected_id':selected['row']['id'] if selected else '',
                    'actual_dimensions':json.dumps(selected['measurement']['dimensions']) if selected else '',
                    'status':'category_candidate_available' if selected else 'unavailable_abstained'})
    output={'scope':'Post-observation ordinary backend correction on fixed real requests; development diagnostic, not a held-out evaluation',
        'aliases_are_hand_defined_vocabulary':True,'model_calls':0,'paid_api_calls':0,
        'generated_complete_rooms':0,'new_mesh_exports':0,'inputs_changed':False,
        'elapsed_seconds':time.monotonic()-start,'unique_meshes_measured':len(cache),
        'rows':records}
    (P/'retrieval_replay.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    with (P/'retrieval_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:v for k,v in output.items() if k!='rows'},ensure_ascii=False))
    print(json.dumps({'requests':len(flat),'category_candidates':sum(r['status']=='category_candidate_available' for r in flat),'abstentions':sum(r['status']=='unavailable_abstained' for r in flat)}))

if __name__=='__main__':main()
