"""Wait for this exact live batch, then export/report each terminal arm once.

Never restarts inference, changes its inputs, edits scenes or performs training.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from native_project import sha256


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();plan=read(a.plan)
    for name,digest in plan['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Postprocessing code changed: '+name)
    root=Path(plan['runtime_directory']);public=Path(plan['public_directory']);out=root/'postprocessing'
    out.mkdir(exist_ok=False);now=lambda:datetime.now(timezone.utc).isoformat();start=time.monotonic()
    source=read(root/'batch_execution/receipt.json');assert source['pid']==plan['batch_pid'] and source['start_ticks']==plan['batch_start_ticks']
    state={'status':'waiting_for_live_batch','started_at':now(),'arms':[],'model_calls':0,'scene_edits':0,'training_updates':0,'plan_sha256':sha256(a.plan)}
    def save():state['updated_at']=now();write(out/'RESULT.json',state);write(public/'POSTPROCESS_STATUS.json',state)
    def run(stage,seconds,command):
        cmd=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(out/stage),'--timeout',str(seconds),'--']+command
        code=subprocess.call(cmd)
        if code:raise RuntimeError(stage+' failed with exit '+str(code)+'; no retry')
    save();print(now(),'begin finite postprocessing of existing batch',flush=True)
    try:
        for arm in plan['arms']:
            destination=root/arm;rollout_receipt=destination/'rollout_execution/receipt.json';audit_receipt=destination/'audit_execution/receipt.json'
            while True:
                if audit_receipt.exists() and read(audit_receipt).get('status')=='completed':break
                batch=read(root/'BATCH.json');actual_source=read(root/'batch_execution/receipt.json')
                if batch['status']=='failed':raise RuntimeError('Source inference batch failed; uncompleted arms will not be restarted')
                if actual_source['status']!='running':raise RuntimeError('Source batch terminal without required completed arm audit')
                # Confirm the original batch child is still the same host process.
                stat=Path('/proc')/str(plan['batch_pid'])/'stat'
                if not stat.exists() or stat.read_text().split(') ',1)[1].split()[19]!=plan['batch_start_ticks']:
                    raise RuntimeError('Original batch process missing/replaced; no restart')
                if time.monotonic()-start>plan['budget']['wait_seconds']:raise TimeoutError('Finite wait budget exhausted; source job untouched')
                time.sleep(10)
            assert read(audit_receipt)['exit_code']==0 and read(rollout_receipt)['exit_code']==0
            cp=destination/'rollout/final_checkpoint';export_plan={'bundle':str(cp),'manifest_sha256':sha256(cp/'manifest.json'),'code_sha256':{n:plan['code_sha256'][n] for n in ['export_catalog_geometry.py','inspect_native_mesh_contacts.py','native_project.py']},'budget':{'output_max_bytes':500_000_000},'include_object_types':['furniture']}
            write(out/(arm+'_EXPORT_PLAN.json'),export_plan);state['status']='processing_'+arm;save()
            run(arm+'_export',180,[plan['geometry_python'],str(Path(__file__).with_name('export_catalog_geometry.py')),'--plan',str(out/(arm+'_EXPORT_PLAN.json')),'--out',str(destination/'export')])
            run(arm+'_report',180,[plan['geometry_python'],str(Path(__file__).with_name('report_transfer_arm.py')),'--plan',str(public/arm/'PLAN.json'),'--out',str(destination/'report')])
            for name in ['REPORT.json','REPORT.md']:
                target=public/arm/name;shutil.copy2(destination/'report'/name,target)
            shutil.copy2(destination/'export/RESULT.json',public/arm/'EXPORT_RESULT.json')
            shutil.copy2(destination/'FINAL_AUDIT.json',public/arm/'FINAL_AUDIT.json')
            obs=read(destination/'rollout/final_observation.json')
            shutil.copy2(obs['image_paths'][0],public/arm/'final_topdown.png')
            state['arms'].append({'arm':arm,'completed_at':now(),'report':str(destination/'report/REPORT.json'),'glb':read(destination/'export/RESULT.json')['glb']});state['status']='waiting_for_live_batch';save()
        state['status']='all_three_terminal_exports_and_reports_completed'
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        state['ended_at']=now();save();print(now(),state['status'],flush=True)


if __name__=='__main__':main()
