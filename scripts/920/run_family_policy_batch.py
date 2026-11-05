"""Run two frozen conditions sequentially; no training, retries or prompt revision."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from native_project import sha256


def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();plan=read(a.plan)
    repo=Path(__file__).resolve().parents[2];root=Path(plan['runtime']);public=Path(plan['public'])
    for name,h in plan['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h,name
    if (root/'BATCH.json').exists():raise ValueError('Existing batch cannot be restarted')
    now=lambda:datetime.now(timezone.utc).isoformat()
    state=dict(status='running_preparation',started_at=now(),arms=[],training_updates=0,plan_sha256=sha256(a.plan))
    def save():
        state['updated_at']=now();write(root/'BATCH.json',state)
        write(public/'STATUS.json',dict(run_id=plan['run_id'],question=plan['question'],status=state['status'],
            updated_at=state['updated_at'],started_at=state['started_at'],ended_at=state.get('ended_at'),
            configuration_sha256=sha256(a.plan),completed=sum(x.get('completed',False) for x in state['arms']),planned=2,
            first_log='begin frozen two-condition furnished-room policy',runtime_directory=str(root),
            latest_verifiable_artifact=str(root/'BATCH.json'),failure_reason=state.get('error'),
            next_step='Observe original handles; retain all outcomes. No training or inference restart.'))
        write(public.parent/'latest.json',dict(run_id=plan['run_id'],stage=state['status'],updated_at=now(),run_directory=str(root)))
    def stage(arm,label,seconds,python,script,*args):
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_stage.py')),
            '--receipt-dir',str(root/arm/(label+'_execution')),'--timeout',str(seconds),'--',python,
            str(Path(__file__).with_name(script)),*map(str,args)],cwd=repo,check=True)
    geo=str(repo.parent/'envs/geometry311/bin/python')
    print(now(),'begin frozen two-condition furnished-room policy',flush=True);save()
    try:
        configs={}
        for arm in plan['execution_order']:
            ref=plan['arms'][arm];assert sha256(Path(ref['path']))==ref['sha256'];q=read(ref['path'])
            assert q['adapter'] is None and q['budget']['model_calls']==q['budget']['tool_calls']==4 and q['budget']['training_updates']==0
            configs[arm]=q
            stage(arm,'prepare',q['budget']['prepare_seconds'],sys.executable,'run_knowledge_policy.py',
                '--plan',ref['path'],'--phase','prepare','--out',q['prepared_directory'])
        prepared={arm:read(Path(q['prepared_directory'])/'RESULT.json') for arm,q in configs.items()}
        assert len({x['state_sha256'] for x in prepared.values()})==1
        assert len({x['request_sha256'] for x in prepared.values()})==2
        assert all(len(x['encoding']['images'])==5 for x in prepared.values())
        view_lists=[read(Path(q['prepared_directory'])/'observation.json')['image_paths'] for q in configs.values()]
        assert view_lists[0]==view_lists[1]
        write(public/'PREPARATION_REVIEW.json',dict(recorded_at=now(),same_initial_state=True,same_five_images=True,
            distinct_condition_requests=True,arms={arm:{k:v[k] for k in ['state_sha256','request_sha256','encoding']} for arm,v in prepared.items()},
            model_calls=0,training_updates=0))
        for arm,q in configs.items():
            stage(arm,'initial_evaluation',60,geo,'evaluate_family_policy.py','--plan',public/'EVALUATION_PLAN.json',
                '--bundle',q['bundle'],'--condition',arm,'--out',root/arm/'INITIAL_EVALUATION.json')
        for arm,q in configs.items():
            row=dict(arm=arm,started_at=now(),completed=False);state['arms'].append(row);state['status']='running_'+arm;save()
            stage(arm,'rollout',q['budget']['rollout_seconds'],sys.executable,'run_knowledge_policy.py',
                '--plan',plan['arms'][arm]['path'],'--phase','rollout','--out',root/arm/'rollout')
            result=read(root/arm/'rollout/RESULT.json');row.update(model_calls=result['model_calls'],tool_calls=result['tool_calls'],stop_reason=result['stop_reason']);save()
            checkpoint=root/arm/'rollout/final_checkpoint'
            stage(arm,'evaluation',60,geo,'evaluate_family_policy.py','--plan',public/'EVALUATION_PLAN.json',
                '--bundle',checkpoint,'--condition',arm,'--out',root/arm/'EVALUATION.json')
            stage(arm,'audit',90,geo,'audit_family_choice_geometry.py','--plan',public/'AUDIT_PLAN.json',
                '--bundle',checkpoint,'--out',root/arm/'FINAL_AUDIT.json')
            export=dict(bundle=str(checkpoint),manifest_sha256=sha256(checkpoint/'manifest.json'),
                include_object_types=['furniture'],budget=dict(output_max_bytes=50000000),
                code_sha256={n:sha256(Path(__file__).parent/n) for n in ['export_catalog_geometry.py','native_project.py','inspect_native_mesh_contacts.py']})
            write(public/arm/'EXPORT_PLAN.json',export)
            stage(arm,'export',90,geo,'export_catalog_geometry.py','--plan',public/arm/'EXPORT_PLAN.json','--out',root/arm/'export')
            restore=dict(bundle=str(checkpoint),manifest_sha256=sha256(checkpoint/'manifest.json'),
                designer=q['designer'],scenesmith=q['scenesmith'],expected_tool_calls=result['tool_calls'],expected_catalog_entries=8,
                code_sha256={n:sha256(Path(__file__).parent/n) for n in ['restore_terminal_catalog.py','catalog_project_session.py','native_project.py','native_checkpoint.py','existing_project_session.py']})
            write(public/arm/'RESTORE_PLAN.json',restore)
            stage(arm,'restore',90,sys.executable,'restore_terminal_catalog.py','--plan',public/arm/'RESTORE_PLAN.json','--out',root/arm/'cold_restore')
            row.update(completed=True,ended_at=now(),evaluation=str(root/arm/'EVALUATION.json'));save()
        state['status']='two_condition_attempts_complete_pending_review'
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        state['ended_at']=now();save();print(now(),state['status'],flush=True)


if __name__=='__main__':main()
