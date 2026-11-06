"""Prospective live shared-input runner. Historical run_edit_policy.py is unchanged.

Requires a new frozen plan and explicit input_contract; no old run continuation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from jsonschema.exceptions import ValidationError

from catalog_project_session import create_app, canonical_hash
from native_project import sha256
from native_checkpoint import rotation_roundtrip_differences
from run_knowledge_policy import request_messages, write
from shared_action_contract import ActionContract, schemas_from_observation, strict_loads
from edit_request import validate_action
from live_edit_request import build_live_request, validate_policy


def compact_rejection_message(exc):
    """Return a bounded receipt, keeping the full schema exception off prompt."""
    detail=exc.message if isinstance(exc,ValidationError) else str(exc)
    detail=' '.join(detail.split())
    prefix=type(exc).__name__+': '
    guidance=' Check advertised tool arguments and exact current object/asset IDs.'
    available=512-len(prefix)-len(guidance)
    if len(detail)>available:detail=detail[:available-3]+'...'
    return prefix+detail+guidance


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--phase',choices=['prepare','rollout'],required=True)
    a=ap.parse_args();p=json.loads(a.plan.read_text())
    validate_policy(p['input_contract'])
    audit_script=p['evaluation'].get('script','evaluate_edit_room.py')
    if audit_script not in ('evaluate_edit_room.py','evaluate_storage_room.py') or audit_script not in p['code_sha256']:
        raise ValueError('Unfrozen or unsupported task evaluator')
    for required in ('run_edit_policy_live.py', 'live_edit_request.py', 'edit_request.py'):
        if required not in p['code_sha256']:raise ValueError('Unfrozen live dependency: '+required)
    a.out.mkdir(parents=True,exist_ok=False)
    for name,digest in p['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Frozen code changed: '+name)
    if sha256(Path(p['evaluation']['plan']))!=p['evaluation']['sha256']:raise ValueError('Evaluation plan changed')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','LD_LIBRARY_PATH',
                'OPENAI_API_KEY','GOOGLE_API_KEY','SCENESMITH_LLM_BASE_URL','SCENESMITH_LLM_API_KEY'):
        os.environ.pop(key,None)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OPENAI_API_KEY='disabled-local-validation',OPENAI_BASE_URL='http://127.0.0.1:9/v1',
        OPENAI_AGENTS_DISABLE_TRACING='1',NO_PROXY='127.0.0.1,localhost',SCENE_RENDER_THREADS='4',
        SCENE_RENDER_SAMPLES='8',OPENSCENEAGENT_STUB_ASSETS='0',TOKENIZERS_PARALLELISM='false')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    now=lambda:datetime.now(timezone.utc).isoformat()
    result=dict(status='running',phase=a.phase,started_at=now(),rounds=[],model_calls=0,tool_attempts=0,
                native_calls=0,gpu_subprocess_seconds=0.,generated_tokens=0,training_updates=0,
                plan_sha256=sha256(a.plan),cell_id=p['cell_id'],input_contract=p['input_contract'])
    public=Path(p['public_directory']);locks=[];client=None;session=None;obs=None
    def progress():
        write(a.out/'RESULT.json',result);write(public/(a.phase.upper()+'_RESULT.json'),result)
    def stage(command,directory,seconds,env=None):
        cmd=[sys.executable,str(Path(__file__).with_name('run_stage.py')),'--receipt-dir',str(directory),
             '--timeout',str(max(1,int(seconds))),'--',*command]
        code=subprocess.call(cmd,env=env)
        return code,json.loads((directory/'receipt.json').read_text())
    def audit(bundle,target):
        cmd=[sys.executable,str(Path(__file__).with_name(audit_script)),'--plan',p['evaluation']['plan'],
             '--bundle',str(bundle),'--out',str(target)]
        code,_=stage(cmd,target.parent/(target.stem+'_receipt'),120)
        if code:raise RuntimeError('Current-state audit failed; no stale feedback')
        return json.loads(target.read_text())
    def request(observation,feedback,rejection=None):
        # Reuse the established native state/tool/image representation. The
        # supplement is shared across all arms, with only source-pool knowledge.
        msg=request_messages(session,observation,p['budget']['tool_calls']-result['tool_attempts'],p,None)
        compact=dict(intersections=[x['objects'] for x in feedback['positive_aabb_pairs'] if x['triangle_surface_intersection_detected']],
            outside=[x['object_id'] for x in feedback['bounds'] if x['outside_nominal_room_xy']],
            identity=feedback['activity']['identity_asset_dimensions_preserved'],
            activity=[{k:v for k,v in s.items() if k!='witness'} for s in feedback['activity']['scenarios']],
            unknown=p.get('task_unknowns',['real ergonomics','guest sleeping','comfort','capacity','facility installation']),previous_rejection=rejection)
        suffix='\nCURRENT MEASURED FEEDBACK (declared proxies; no reference solution):\n'+json.dumps(compact,separators=(',',':'))
        user=next(m for m in reversed(msg) if m['role']=='user')
        if isinstance(user['content'],str):user['content']+=suffix
        else:user['content'].append(dict(type='text',text=suffix))
        current,metadata=build_live_request(msg,result['rounds'],p['input_contract'])
        write(a.out/'latest_input_history.json',metadata)
        result['latest_input_history']=metadata
        return current
    def model_step(req,out,mode,turn):
        remaining=p['budget']['gpu_seconds_per_cell']-result['gpu_subprocess_seconds']
        if mode=='generate' and remaining<15:return None
        child=dict(p);child['inference']={**p['inference'],'seed':p['inference']['seed']+turn}
        childplan=out.parent/'model_plan.json';write(childplan,child)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpu_uuid'] if mode=='generate' else '')
        cmd=[p['model_python'],str(Path(__file__).with_name('edit_study_model_step.py')),'--plan',str(childplan),
             '--request',str(req),'--out',str(out),'--mode',mode]
        timeout=min(p['budget']['seconds_per_model_call'],remaining) if mode=='generate' else 180
        code,receipt=stage(cmd,out.parent/'model_receipt',timeout,env)
        if mode=='generate':
            result['gpu_subprocess_seconds']+=receipt['elapsed_seconds'];result['model_calls']+=1
        record=json.loads((out/'RESULT.json').read_text()) if (out/'RESULT.json').exists() else {}
        result['generated_tokens']+=record.get('output_tokens',0)
        progress()
        if code:
            if code==124 and timeout==remaining:
                result['stop_reason']='cell_gpu_budget_exhausted';return None
            raise RuntimeError(f'Model stage exited {code}; no retry')
        return record
    progress();print(now(),'begin study cell',p['cell_id'],a.phase,flush=True)
    try:
        if a.phase=='rollout':
            for path in p['gpu_lock_paths']:
                lock=Path(path).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
            values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip().split(',')
            if values[0].strip()!=p['gpu_uuid'] or int(values[1])>512 or int(values[2])<23000:raise RuntimeError('GPU0 unavailable')
        from fastapi.testclient import TestClient
        app,server=create_app(Path(p['designer']),Path(p['scenesmith']));client=TestClient(app)
        def post(route,payload=None):
            reply=client.post(route,json=payload or {})
            if reply.status_code!=200:raise RuntimeError('Native route '+route+': '+reply.text[:500])
            return reply.json()
        prepared=Path(p['prepared_directory'])
        bundle=Path(p['bundle']) if a.phase=='prepare' else prepared/'checkpoint'
        if a.phase=='prepare' and sha256(bundle/'manifest.json')!=p['manifest_sha256']:raise ValueError('Initial bundle changed')
        post('/session/catalog',dict(bundle_directory=str(bundle),output_directory=str(a.out/'house')))
        session=server._state.session
        post('/session/stage',dict(stage='furniture',stub_assets=False,skip_render=False))
        if a.phase=='prepare':
            obs=post('/session/observation');write(a.out/'observation.json',obs)
            result['checkpoint']=post('/session/checkpoint',dict(output_directory=str(a.out/'checkpoint')))
            feedback=audit(a.out/'checkpoint',a.out/'initial_audit.json')
            req=a.out/'request.json';write(req,request(obs,feedback))
            result['encoding']=model_step(req,a.out/'model','encode',0)['encoding']
            result['state_sha256']=canonical_hash(session.scene.to_state_dict())
            result['request_sha256']=sha256(req);result['status']='prepared_without_model'
        else:
            prepared_result=json.loads((prepared/'RESULT.json').read_text())
            if prepared_result['status']!='prepared_without_model':raise ValueError('Initial preparation not accepted')
            old=json.loads((prepared/'checkpoint/house_state.json').read_text())['rooms'][session.scene.room_id]
            result['restoration_differences']=rotation_roundtrip_differences(old,session.scene.to_state_dict())
            obs=json.loads((prepared/'observation.json').read_text());feedback=json.loads((prepared/'initial_audit.json').read_text())
            result['stop_reason']='call_budget_exhausted';rejection=None
            for turn in range(p['budget']['model_calls']):
                if result['tool_attempts']>=p['budget']['tool_calls']:break
                rd=a.out/f'round_{turn:03d}';rd.mkdir();req=rd/'request.json';write(req,request(obs,feedback,rejection));rejection=None
                record=model_step(req,rd/'model','generate',turn)
                if record is None:result['stop_reason']='cell_gpu_budget_exhausted';break
                raw=(rd/'model/raw_output.txt').read_text();row=dict(turn=turn,model=record,raw_output=raw,
                    input_history=result['latest_input_history']);result['rounds'].append(row)
                blocks=re.findall(r'<tool_call>(.*?)</tool_call>',raw,re.DOTALL)
                if not blocks and '<done>' in raw and not record.get('output_rejected_reason'):
                    result['stop_reason']='model_done';progress();break
                result['tool_attempts']+=1
                context=session.current_context();contract=ActionContract(schemas_from_observation(obs),context)
                try:
                    if record.get('output_rejected_reason') or len(blocks)!=1 or raw.count('<tool_call>')!=1:
                        raise ValueError('Expected one complete tool block')
                    call=contract.validate(blocks[0],context)['call']
                    validate_action(call,json.loads(req.read_text()))
                except (ValueError,ValidationError) as exc:
                    # Schema errors are policy outcomes, consume the same budget,
                    # and return the actual rejection to all arms identically.
                    rejection=compact_rejection_message(exc)
                    row['rejection']=rejection
                    row['full_validation_error']=f'{type(exc).__name__}: {exc}'
                    write(rd/'validation_error.json',dict(feedback=rejection,full_error=row['full_validation_error']))
                    progress();continue
                before=canonical_hash(session.scene.to_state_dict())
                reply=post('/session/step',dict(tool_calls=[call]));result['native_calls']+=1
                write(rd/'tool_reply.json',reply);row.update(call=call,reply=reply,before_sha256=before)
                row['checkpoint']=post('/session/checkpoint',dict(output_directory=str(rd/'checkpoint')))
                after=canonical_hash(session.scene.to_state_dict());row['after_sha256']=after
                if after!=before:
                    obs=post('/session/observation');feedback=audit(rd/'checkpoint',rd/'audit.json')
                write(rd/'observation.json',obs);progress()
            result['status']='policy_completed_pending_comparison'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',stop_reason='infrastructure_error')
        raise
    finally:
        if session is not None and a.phase=='rollout':
            try:
                result['checkpoint']=post('/session/checkpoint',dict(output_directory=str(a.out/'final_checkpoint')))
                write(a.out/'final_state.json',session.scene.to_state_dict())
                if obs is not None:write(a.out/'final_observation.json',obs)
                result['final_audit']=audit(a.out/'final_checkpoint',a.out/'final_audit.json')
            except BaseException as exc:
                result['finalization_error']=f'{type(exc).__name__}: {exc}';result['status']='failed'
        result['ended_at']=now();progress()
        if client is not None:
            try:client.post('/session/close',json={})
            finally:client.close()
        for lock in locks:lock.close()
        print(now(),result['status'],flush=True)
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
