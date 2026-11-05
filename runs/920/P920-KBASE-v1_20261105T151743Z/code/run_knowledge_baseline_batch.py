"""Two prospectively frozen development prompt arms, no adaptation or retry."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from native_project import sha256


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();plan=read(a.plan)
    for name,h in plan['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h,name
    root=Path(plan['runtime']);public=Path(plan['public']);root.mkdir(parents=True,exist_ok=True)
    if (root/'BATCH.json').exists():raise ValueError('Existing batch cannot be restarted')
    now=lambda:datetime.now(timezone.utc).isoformat();state={'status':'running','started_at':now(),'arms':[],'training_updates':0,'plan_sha256':sha256(a.plan)}
    def save():
        state['updated_at']=now();write(root/'BATCH.json',state)
        write(public/'STATUS.json',{'run_id':public.name,'question':plan['question'],'status':state['status'],'updated_at':now(),'started_at':state['started_at'],'ended_at':state.get('ended_at'),'run_directory':str(root),'configuration_hashes':{'plan':sha256(a.plan)},'completed_planned':{'arms':[sum(x.get('completed',False) for x in state['arms']),2]},'latest_verifiable_artifact':str(root/'BATCH.json'),'failure_reason':state.get('error'),'next_step':'Observe same batch; retain nine requirements and all failures, no training or inference restart.'})
        write(public.parent/'latest.json',{'run_id':public.name,'stage':'strong_prompt_same_knowledge_development','status':state['status'],'updated_at':now(),'run_directory':str(root)})
    def run(arm,stage,seconds,py,script,*args):
        dest=root/arm/(stage+'_execution');cmd=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(dest),'--timeout',str(seconds),'--',py,str(Path(__file__).with_name(script)),*map(str,args)]
        code=subprocess.call(cmd)
        if code:raise RuntimeError(arm+'/'+stage+' exit '+str(code)+'; stop without retry')
    save();print(now(),'begin frozen two-arm knowledge baseline',flush=True)
    try:
        plans={}
        for arm in plan['execution_order']:
            ref=plan['arms'][arm];assert sha256(Path(ref['path']))==ref['sha256'];q=read(ref['path']);plans[arm]=q
            assert q['adapter'] is None and q['budget']['training_updates']==0
        allowed={'comparison_arm','public_directory','runtime_directory','public','runtime','prepared_directory','prompt_supplement'}
        comparable=[{k:v for k,v in q.items() if k not in allowed} for q in plans.values()];assert comparable[0]==comparable[1]
        # Prepare both before any generation, without GPU calls. Shared native state and images, explicit different prompt treatment.
        for arm,q in plans.items():
            run(arm,'prepare',q['budget']['prepare_seconds'],sys.executable,'run_knowledge_policy.py','--plan',plan['arms'][arm]['path'],'--phase','prepare','--out',q['prepared_directory'])
        preps={arm:read(Path(q['prepared_directory'])/'RESULT.json') for arm,q in plans.items()}
        assert len({x['state_sha256'] for x in preps.values()})==1
        assert len({x['request_sha256'] for x in preps.values()})==2
        assert all(len(x['encoding']['images'])==5 for x in preps.values())
        observations=[read(Path(q['prepared_directory'])/'observation.json')['image_paths'] for q in plans.values()];assert observations[0]==observations[1]
        write(public/'PREPARATION_REVIEW.json',{'recorded_at':now(),'same_initial_native_state':True,'same_five_image_paths':True,'distinct_frozen_prompt_requests':True,'arms':{arm:{k:x[k] for k in ['state_sha256','request_sha256','encoding']} for arm,x in preps.items()},'model_calls':0,'training_updates':0})
        for arm,q in plans.items():
            row={'arm':arm,'started_at':now(),'completed':False};state['arms'].append(row);state['status']='running_'+arm;save()
            run(arm,'rollout',q['budget']['rollout_seconds'],sys.executable,'run_knowledge_policy.py','--plan',plan['arms'][arm]['path'],'--phase','rollout','--out',root/arm/'rollout')
            result=read(root/arm/'rollout/RESULT.json');row.update(model_calls=result['model_calls'],tool_calls=result['tool_calls'],stop_reason=result['stop_reason']);save()
            cp=root/arm/'rollout/final_checkpoint';gp=plan['geometry_python']
            run(arm,'audit',120,gp,'audit_transfer_room.py','--plan',public/'AUDIT_PLAN.json','--bundle',cp,'--out',root/arm/'FINAL_AUDIT.json')
            export={'bundle':str(cp),'manifest_sha256':sha256(cp/'manifest.json'),'code_sha256':{n:plan['code_sha256'][n] for n in ['export_catalog_geometry.py','inspect_native_mesh_contacts.py','native_project.py']},'budget':{'output_max_bytes':500_000_000},'include_object_types':['furniture']};write(public/arm/'EXPORT_PLAN.json',export)
            run(arm,'export',180,gp,'export_catalog_geometry.py','--plan',public/arm/'EXPORT_PLAN.json','--out',root/arm/'export')
            run(arm,'report',180,gp,'report_transfer_arm.py','--plan',public/arm/'PLAN.json','--out',root/arm/'report')
            run(arm,'schema_review',60,sys.executable,'review_transfer_actions.py','--plan',public/arm/'PLAN.json','--out',root/arm/'ACTION_SCHEMA_REVIEW.json')
            restore=dict(plan['restore_template'],bundle=str(cp),manifest_sha256=sha256(cp/'manifest.json'),expected_tool_calls=result['tool_calls']);write(public/arm/'RESTORE_PLAN.json',restore)
            run(arm,'restore',90,sys.executable,'restore_terminal_catalog.py','--plan',public/arm/'RESTORE_PLAN.json','--out',root/arm/'cold_restore')
            row.update(completed=True,ended_at=now(),report=str(root/arm/'report/REPORT.json'));save()
        state['status']='two_development_baselines_completed_pending_visual_review'
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        state['ended_at']=now();save();print(now(),state['status'],flush=True)

if __name__=='__main__':main()
