"""Check current raw ledgers, freeze hashes, companions and review delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);a=p.parse_args()
    stage=a.stage.resolve();repo=Path(__file__).resolve().parents[2];m=read(stage/'delivery_manifest.json')
    checks=[]
    def check(label,value):
        checks.append(dict(label=label,passed=bool(value)))
        assert value,label
    for row in m['files']+m['scripts']:
        path=repo/row['path'];check('content:'+row['path'],path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'])
    result=read(stage/'trial_actual_v2/result.json');ledger=result['model_ledger'];summary=read(stage/'summary.json')
    check('raw_forward_count',len(ledger['forwards'])==summary['corrected_continuation']['forwards']==23)
    check('raw_output_count',sum(x['output_tokens'] for x in ledger['forwards'])==4553)
    check('all_actual_forwards_finished',all(x['status']=='returned' for x in ledger['forwards']))
    check('one_frozen_model',len(ledger['model_loads'])==1 and all(x['all_parameters_frozen'] for x in ledger['forwards'])
          and len(set(x['model_object_id'] for x in ledger['forwards']))==1)
    check('deadline_status_preserved',read(stage/'review_traces/actor_deadline.json')['status']=='terminal_timeout')
    check('not_reclassified_success',result['status']=='failed_no_restart' and summary['whole_scene_success'] is None and not summary['goal_achieved'])
    check('closed_allowance',read(stage/'TRIAL_CLOSED_v2.json')['unused_allowance_carries'] is False)
    for row in result['owned_processes']:
        check('owned_trial_process_terminal:'+str(row['pid']),row['exit_code'] is not None and not Path('/proc',str(row['pid'])).exists())
    protocol=read(stage/'trial_protocol_v2.json')
    for name,h in protocol['code_sha256'].items():check('freeze:'+name,sha(name)==h)
    art=read(stage/'partial_artifact_v1/result.json');glb=stage/'partial_artifact_v1/partial_scene.glb'
    with glb.open('rb') as f:magic,version,length=struct.unpack('<4sII',f.read(12))
    check('GLB_header_and_cold_read_receipt',magic==b'glTF' and version==2 and length==glb.stat().st_size and art['cold_geometry_count']==17)
    check('exporter_nonzero_exit_visible',art['exporter_exit_code']==-11 and not art['exporter_process_passed'])
    certificate=read(stage/'floor_halfspace_certificate_v1/result.json')
    check('floor_discrepancy_retained',certificate['certified_with_native_distance_below_minus_1mm']==10 and not certificate['original_score_changed'])
    rt=read(stage/'mutable_asset_rollback_v2/result.json')
    check('same_JSON_with_different_SDF',rt['JSON_only_reset']['serialized_scene_equals_parent'] and
          rt['JSON_only_reset']['SDF_SHA256']!=rt['initial']['SDF_SHA256'])
    check('ordinary_byte_reset_executed',rt['same_action_after_byte_reset']['mesh_scale_values']!=rt['second_action_after_JSON_only']['mesh_scale_values']
          and rt['finally_restored']==rt['initial'])
    check('no_paid_API_or_training',summary['paid_API_calls']==summary['training_updates']==0)
    with zipfile.ZipFile(stage/'evidence.zip') as z:
        check('review_archive_integrity',z.testzip() is None)
        for row in m['files']+m['scripts']:
            if row['path'] in z.namelist():check('review_archive_member:'+row['path'],hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'])
    check('reserve_preserved',shutil.disk_usage(stage).free>=5*1024**3)
    proof=dict(status='delivery_and_raw_receipts_verified',checks=checks,total_checks=len(checks),
               scene_quality_or_new_method_not_certified=True,independent_FCL_reexecution=False,
               evidence_zip_sha256=sha(stage/'evidence.zip'),goal_achieved=False)
    (stage/'delivery_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='checks'}))


if __name__=='__main__':main()
