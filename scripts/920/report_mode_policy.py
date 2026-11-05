"""Archive fixed activity-policy attempts without promoting model failures to success."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from native_project import sha256
from native_checkpoint import canonical_hash, rotation_roundtrip_differences
from run_mode_policy import write


def read(path): return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    plan=read(a.plan);pub=Path(plan['public_directory']);rt=Path(plan['runtime_directory'])
    batch=read(rt/'batch/RESULT.json');assert batch['status'] in ['completed','completed_with_failures']
    assert batch['completed']==batch['planned']==4
    initial=read(rt/'prepare/initial_state.json');cases=[];view_bindings=0;checkpoints=0
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    def check_obs(obs_path,state):
        nonlocal view_bindings
        obs=read(obs_path);binding=read(obs['state_binding'])
        assert canonical_hash(binding['native_state'])==binding['state_sha256']
        differences=rotation_roundtrip_differences(binding['native_state'],state)
        assert obs['image_paths']==[x['path'] for x in binding['images']]
        assert len(binding['images'])==5
        for row in binding['images']:assert sha256(Path(row['path']))==row['sha256']
        view_bindings+=1
        return obs,differences
    def check_checkpoint(path):
        nonlocal checkpoints
        manifest=read(path/'manifest.json')
        for name,row in manifest['files'].items():
            f=path/name;assert f.stat().st_size==row['bytes'] and sha256(f)==row['sha256']
        checkpoints+=1
    for name in ['RESULT.json','initial_state.json','observation.json']:
        copy(rt/'prepare'/name,pub/'prepare'/name)
    check_obs(rt/'prepare/observation.json',initial);check_checkpoint(rt/'prepare/checkpoint')
    initial_obs=read(rt/'prepare/observation.json')
    copy(Path(initial_obs['image_paths'][0]),pub/'views/initial_topdown.png')
    for cp in plan['case_plans']:
        cfg=read(cp);cid=cfg['case_id'];out=rt/'rollouts'/cid;dest=pub/'cases'/cid
        result=read(out/'RESULT.json');receipt=read(rt/'execution'/cid/'receipt.json')
        before=read(out/'initial_state.json');rotation_roundtrip_differences(initial,before)
        for f in out.glob('*.json'):copy(f,dest/f.name)
        for f in (rt/'prepare'/cid).glob('*.json'):copy(f,dest/'prepared'/f.name)
        for f in (rt/'prepare'/cid/'encode').glob('*.json'):copy(f,dest/'prepared/encode'/f.name)
        model_rows=[];rounds=[]
        for rd in sorted(out.glob('round_*')):
            for f in rd.glob('*.json'):copy(f,dest/rd.name/f.name)
            model_path=rd/'model/RESULT.json'
            if model_path.exists():
                model=read(model_path);assert model['request_sha256']==sha256(rd/'request.json')
                model_rows.append(model)
                for f in (rd/'model').glob('*'):
                    if f.is_file() and f.suffix in ('.json','.txt'):copy(f,dest/rd.name/'model'/f.name)
            if (rd/'model_execution/receipt.json').exists():
                for f in (rd/'model_execution').glob('*'):
                    if f.is_file():copy(f,dest/rd.name/'model_execution'/f.name)
            obs,_=check_obs(rd/'before_observation.json',read(rd/'before_state.json'))
            row={'round':rd.name,'before_condition_pass':read(rd/'before_evaluation.json')['condition_pass']}
            if (rd/'after_state.json').exists():
                check_obs(rd/'after_observation.json',read(rd/'after_state.json'));check_checkpoint(rd/'checkpoint')
                ev=read(rd/'after_evaluation.json');row.update(after_condition_pass=ev['condition_pass'],
                    geometry_ok=ev.get('geometry_ok'),fixed_access_ok=ev.get('fixed_access_ok'),identity_ok=ev['identity_ok'])
            rounds.append(row)
        calls=sum(x['model_calls'] for x in model_rows);assert calls==result['model_calls']<=6
        assert result['tool_calls']<=6
        final_path=out/'final_evaluation.json'
        final=read(final_path) if final_path.exists() else None
        terminal_checkpoint=out/'final_checkpoint'
        if terminal_checkpoint.exists():check_checkpoint(terminal_checkpoint)
        if (out/'final_observation.json').exists():
            obs,_=check_obs(out/'final_observation.json',read(out/'final_state.json'))
            copy(Path(obs['image_paths'][0]),pub/'views'/(cid+'_topdown.png'))
            copy(Path(obs['image_paths'][1]),pub/'views'/(cid+'_south.png'))
        for f in (rt/'execution'/cid).glob('*'):
            if f.is_file():copy(f,pub/'execution'/cid/f.name)
        attempted=[c for row in result['rounds'] for c in row['calls']]
        replies=[x for row in result['rounds'] for x in row.get('reply',{}).get('results',[])]
        cases.append({'case':cid,'activity':cfg['condition'],'knowledge':cfg['knowledge_arm'],
            'stage_exit':receipt['exit_code'],'actual_seconds':receipt['elapsed_seconds'],
            'model_calls':calls,'tool_calls':result['tool_calls'],
            'input_tokens':sum(x.get('encoding',{}).get('input_tokens',0) for x in model_rows),
            'output_tokens':sum(x.get('output_tokens',0) for x in model_rows),
            'generation_seconds':sum(x.get('generation_seconds',0) for x in model_rows),
            'initial_condition_pass':read(out/'initial_evaluation.json')['condition_pass'],
            'final_condition_pass':final['condition_pass'] if final else None,
            'completed_attempt':result['status']=='completed' and receipt['exit_code']==0,
            'stop_reason':result.get('stop_reason'),'error':result.get('error'),
            'native_accepted_actions':sum(isinstance(x.get('output'),dict) and x['output'].get('success') is True for x in replies),
            'targeted_object_ids':sorted({x['arguments']['object_id'] for x in attempted if 'object_id' in x.get('arguments',{})}),
            'repeated_exact_tool_requests':len(attempted)-len({canonical_hash(x) for x in attempted}),
            'rounds':rounds,'final_state_sha256':sha256(out/'final_state.json') if (out/'final_state.json').exists() else None,
            'editable_checkpoint':str(terminal_checkpoint) if terminal_checkpoint.exists() else None,
            'sweep_warnings':[{'turn':x['turn'],'diagnostic':x['actual_sweep']} for x in result['rounds']
                              if 'actual_sweep' in x and not x['actual_sweep']['clear_under_declared_sweep_model']]})
    for name in ['prepare_execution','batch_execution']:
        for f in (rt/name).glob('*'):
            if f.is_file():copy(f,pub/'execution'/name/f.name)
    copy(rt/'batch/RESULT.json',pub/'BATCH_RESULT.json')
    rows=[]
    for f in sorted(rt.rglob('*')):
        if f.is_file():rows.append({'path':str(f.relative_to(rt)),'bytes':f.stat().st_size,'sha256':sha256(f)})
    write(pub/'SERVER_FILES.json',{'runtime':str(rt),'files':rows,'recovery':'Use fixed model/config/source hashes and independent prepared bundle/checkpoints. No re-execution needed to publish. Original assets and all five-view images remain on server.'})
    result={'run_id':plan['run_id'],'evaluated_at':datetime.now(timezone.utc).isoformat(),
        'status':batch['status'],'cases':cases,'model_calls':sum(x['model_calls'] for x in cases),
        'tool_calls':sum(x['tool_calls'] for x in cases),'training_updates':0,'independent_professional_reviews':0,
        'verified_view_bindings':view_bindings,'verified_checkpoints':checkpoints,
        'interpretation':'Within one exposed researcher reconstruction. Compare final states and costs, distinguish already-passing work tasks from hosting changes. No paired-training or generalization claim.',
        'unknowns':['Human comfort and handling, real wheel joints/brakes/load, full residential facility use and expert quality.',
                    'Four greedy attempts in one scene are not independent projects or a statistical effect estimate.',
                    'Source knowledge adds85 initial tokens; controls share original activity brief, geometry/access feedback, model and ceilings, not identical compute.',
                    'Mode checker uses a declared near-parallel same-axis join and conservative AABBs; valid configurations outside this criterion remain outside the claimed scope.']}
    write(pub/'RESULT.json',result)
    write(pub/'STATUS.json',{'run_id':plan['run_id'],'question':plan['question'],'status':batch['status'],
        'phase':'final_evaluation','updated_at':result['evaluated_at'],'started_at':read(rt/'batch_execution/receipt.json')['started_at'],
        'ended_at':read(rt/'batch_execution/receipt.json')['ended_at'],'actual_batch_exit':read(rt/'batch_execution/receipt.json')['exit_code'],
        'completed':4,'planned':4,'model_calls':result['model_calls'],'tool_calls':result['tool_calls'],
        'configuration_hashes':{'plan':sha256(a.plan),'inputs':sha256(pub/'INPUT_FREEZE.json')},
        'latest_verifiable_artifact':str(pub/'RESULT.json'),'failure_reason':[x['error'] for x in cases if x['error']],
        'next_step':'Review final actual scenes and errors; no retries or input retuning on these cases.'})
    write(pub.parent/'latest.json',{'run_id':plan['run_id'],'stage':'final_evaluation','status':batch['status'],
        'updated_at':result['evaluated_at'],'run_directory':str(rt),'status_path':str(pub/'STATUS.json')})
    print(json.dumps({'cases':len(cases),'model_calls':result['model_calls'],'tool_calls':result['tool_calls'],
        'outcomes':{x['case']:x['final_condition_pass'] for x in cases}},ensure_ascii=False))


if __name__=='__main__':main()
