"""Finish the approved study after its existing training process; never restart it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil

from native_project import sha256
from run_knowledge_policy import write


def read(p):
    return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,required=True)
    a=ap.parse_args();p=a.study.resolve();train=read(p/'TRAIN_PLAN.json')
    def validate_lock():
        lock=read(p/'EVALUATION_EXECUTION_LOCK.json')
        for path,digest in lock['files'].items():
            if sha256(Path(path))!=digest:raise ValueError('Frozen evaluation input changed: '+path)
    validate_lock()
    root=Path(train['runtime_directory']);now=lambda:datetime.now(timezone.utc).isoformat()
    if (p/'EVALUATION_BATCH.json').exists():raise FileExistsError('Batch already exists; inspect receipts, do not restart')
    result=dict(status='waiting_for_existing_training',started_at=now(),cells=[],gpu_subprocess_seconds=0.,training_started_by_this_process=False)
    def progress():
        write(p/'EVALUATION_BATCH.json',result)
        s=read(p/'STATUS.json');s.update(updated_at=now(),evaluation_batch_status=result['status'])
        counts=[]
        for arm in train['arm_order']:
            f=Path(train['training_output_root'])/arm/'RUN.json'
            counts.append(read(f).get('optimizer_updates',0) if f.exists() else 0)
        s['completed_planned']['training_updates']=[sum(counts),16]
        s['completed_planned']['evaluation_rooms']=[len(result['cells']),12]
        if result['status']!='waiting_for_existing_training':
            s.update(status=result['status'],latest_verifiable_artifact=str(p/'EVALUATION_BATCH.json'),failure_reason=result.get('error'))
        write(p/'STATUS.json',s)
        write(p.parent/'latest.json',dict(run_id=p.name,stage=s['status'],updated_at=s['updated_at'],run_directory=str(root)))
    progress();print(now(),'watch existing training; never launch optimizer',flush=True)
    try:
        deadline=time.monotonic()+14460
        while True:
            status=read(p/'TRAINING_RESULT.json')
            if status['status']=='completed':break
            if status['status']=='failed':raise RuntimeError('Existing training failed; evaluation not started')
            if time.monotonic()>deadline:raise TimeoutError('Existing training did not finish within approved two-arm budget')
            progress();time.sleep(20)
        for arm in train['arm_order']:
            r=read(Path(train['training_output_root'])/arm/'RUN.json')
            if r['status']!='completed' or r['optimizer_updates']!=8 or not r['same_process_adapter_save_zero_reload_exact']:
                raise ValueError('Final adapter has not passed fixed update/save-restore acceptance')
        validate_lock()
        matrix=read(p/'EVALUATION_COMMAND_MATRIX.json')['cells']
        order={arm:i for i,arm in enumerate(['frozen_base_knowledge_feedback','action_only','decision_action'])}
        matrix=sorted(matrix,key=lambda c:(c['task'],c['seed'],order[c['arm']]))
        native=str(p.parents[2].parent/'envs/scene_designer311_native/bin/python')
        # p is repo/runs/920/run, so repository is p.parents[2].
        repo=p.parents[2];scripts=repo/'scripts/920'
        result['status']='evaluation_running';progress()
        for cell in matrix:
            template=Path(cell['plan_template']);plan=read(template);public=Path(plan['public_directory']);runtime=Path(cell['runtime'])
            prepared=read(Path(plan['prepared_directory'])/'RESULT.json')
            if prepared['status']!='prepared_without_model':raise ValueError('Common initial request not ready')
            if cell['arm']!='frozen_base_knowledge_feedback':
                adapter=Path(plan.pop('adapter_pending_binding'))
                files=['adapter_config.json','adapter_model.safetensors']
                plan['adapter']=dict(directory=str(adapter),sha256={n:sha256(adapter/n) for n in files})
            planfile=public/'PLAN.json'
            if planfile.exists():raise FileExistsError('Cell plan already bound; inspect previous attempt')
            write(planfile,plan)
            cmd=[sys.executable,str(scripts/'run_stage.py'),'--receipt-dir',str(runtime/'rollout_receipt'),
                 '--timeout','1800','--',native,str(scripts/'run_edit_policy.py'),'--plan',str(planfile),
                 '--phase','rollout','--out',str(runtime/'rollout')]
            print(now(),'run',cell['cell'],flush=True)
            code=subprocess.call(cmd)
            report=read(runtime/'rollout/RESULT.json') if (runtime/'rollout/RESULT.json').exists() else {}
            result['gpu_subprocess_seconds']+=report.get('gpu_subprocess_seconds',0.)
            result['cells'].append(dict(cell_id=cell['cell'],arm=cell['arm'],task=cell['task'],seed=cell['seed'],
                exit_code=code,plan_sha256=sha256(planfile),receipt=read(runtime/'rollout_receipt/receipt.json'),
                result_path=str(runtime/'rollout/RESULT.json'),model_calls=report.get('model_calls'),
                native_calls=report.get('native_calls'),gpu_subprocess_seconds=report.get('gpu_subprocess_seconds'),
                final_checkpoint=report.get('checkpoint'),status=report.get('status')))
            # Archive reviewable small evidence even for a failed model attempt.
            saved=public/'recorded';saved.mkdir(exist_ok=False)
            for name in ('RESULT.json','final_state.json','final_audit.json','final_observation.json'):
                source=runtime/'rollout'/name
                if source.exists():shutil.copy2(source,saved/name)
            for rd in sorted((runtime/'rollout').glob('round_*')):
                target=saved/rd.name;target.mkdir()
                for sub in ('request.json','tool_reply.json','model/raw_output.txt','model/RESULT.json','model/output_ids.json'):
                    source=rd/sub
                    if source.exists():
                        dest=target/sub;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            observation=runtime/'rollout/final_observation.json'
            if observation.exists():
                views=saved/'views';views.mkdir()
                for i,path in enumerate(read(observation)['image_paths']):
                    source=Path(path);shutil.copy2(source,views/(str(i)+'_'+source.name))
            progress()
            if code or report.get('status')=='failed':raise RuntimeError('Cell infrastructure failure; preserve current batch, no retry')
            if result['gpu_subprocess_seconds']>7200:raise RuntimeError('Evaluation time ceiling reached')
            bundle=runtime/'rollout/final_checkpoint'
            export_plan=dict(bundle=str(bundle),manifest_sha256=sha256(bundle/'manifest.json'),
                code_sha256={n:sha256(scripts/n) for n in ('export_catalog_geometry.py','inspect_native_mesh_contacts.py','native_project.py')},
                budget=dict(output_max_bytes=150000000))
            export_file=public/'EXPORT_PLAN.json';write(export_file,export_plan)
            export_cmd=[sys.executable,str(scripts/'run_stage.py'),'--receipt-dir',str(runtime/'export_receipt'),
                '--timeout','120','--',native,str(scripts/'export_catalog_geometry.py'),'--plan',str(export_file),'--out',str(runtime/'export')]
            export_code=subprocess.call(export_cmd)
            if export_code:raise RuntimeError('Terminal scene export failed; preserve generated scene without rerunning')
            result['cells'][-1]['export']=read(runtime/'export/RESULT.json')
            write(public/'EXPORT_RESULT.json',result['cells'][-1]['export']);progress()
        result.update(status='twelve_cells_completed_pending_result_review',ended_at=now(),exit_code=0)
    except BaseException as exc:
        result.update(status='failed',ended_at=now(),exit_code=1,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        progress()


if __name__=='__main__':main()
