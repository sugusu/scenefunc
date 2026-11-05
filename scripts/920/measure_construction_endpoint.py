"""Report declared source-related endpoint measures; do not infer construction gain."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from native_project import validate_bundle
from measure_storage_consolidation import storage_measures


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args()
    root=(a.runtime/'native/checkpoint').resolve()
    manifest,state,_,_=validate_bundle(root);rid=manifest['selected_room_id']
    plan=json.loads((a.public/'SOURCE_MEASUREMENT.json').read_text())
    original_ids=plan['storage_ids'];plan['storage_ids']=[x.removeprefix('recon_') for x in original_ids]
    result=storage_measures(state['rooms'][rid],root/('room_'+rid),root,plan)
    result.update(recorded_at=datetime.now(timezone.utc).isoformat(),source_ids=original_ids,actual_ids=plan['storage_ids'],
        interpretation='Endpoint diagnostics for newly constructed instances. No improvement claim against the initially unfurnished space; not a historical trajectory or independent professional review.',
        source_plan_sha256=hashlib.sha256((a.public/'SOURCE_MEASUREMENT.json').read_bytes()).hexdigest(),
        checkpoint_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        model_calls=0,training_updates=0)
    with (a.runtime/'SOURCE_ENDPOINT.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'fixed_counts':[sum(v['reachable_from_entry'] for v in x['targets'].values()) for x in result['fixed_access']['scenarios']],
        'cabinet_front_counts':[sum(v['reachable_from_entry'] for v in x['targets'].values()) for x in result['storage_front_approaches']],
        'living_unoccupied_m2':result['living_region']['unoccupied_envelope_m2']}))

if __name__=='__main__':main()
