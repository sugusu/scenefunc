"""Summarize complete, matched current-activity trajectories without target poses."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from native_checkpoint import canonical_hash
from native_project import sha256
from score_activity_transition import score_transition
from run_knowledge_policy import write


def image_hash(path):
    with Image.open(path) as image:
        image=image.convert('RGB')
        return [list(image.size),hashlib.sha256(image.tobytes()).hexdigest()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();plan=json.loads((a.run/'BATCH_PLAN.json').read_text())
    batch=json.loads((a.run/'BATCH_RESULT.json').read_text())
    if batch['status']=='running':raise ValueError('Do not freeze a partial running result')
    refs={};rows=[];matched={};paths={};matched_arms={};frozen={};archived={}
    def read(path):
        path=Path(path);refs[str(path)]=sha256(path)
        return json.loads(path.read_text())
    for cell in plan['cells']:
        for path,digest in cell['frozen_inputs'].items():
            if sha256(Path(path))!=digest:raise ValueError('Frozen batch input changed: '+path)
            frozen[path]=digest
        folder=Path(cell['output'])
        if not (folder/'RESULT.json').exists():continue
        r=read(folder/'RESULT.json');p=read(cell['plan'])
        for path,digest in p['code_sha256'].items():
            source=Path(path) if Path(path).is_absolute() else Path(__file__).parent/path
            if sha256(source)!=digest:raise ValueError('Frozen source changed: '+str(source))
            frozen[str(source)]=digest
        public=Path(p['public_directory'])/'recorded'
        for name in ('RESULT.json','final_state.json','final_audit.json','final_observation.json',
                     'restored_request.json','restored_audit.json'):
            if (folder/name).exists():
                if sha256(folder/name)!=sha256(public/name):raise ValueError('Archived evidence differs: '+name)
                archived[str(public/name)]=sha256(public/name)
        if (folder/'final_observation.json').exists():
            for i,path in enumerate(read(folder/'final_observation.json')['image_paths']):
                dest=public/'views'/f'{i}_{Path(path).name}'
                if sha256(Path(path))!=sha256(dest):raise ValueError('Archived view differs')
                archived[str(dest)]=sha256(dest)
        row=dict(cell=cell['cell_id'],condition=cell['condition'],arm=cell['arm'],
            status=r['status'],model_calls=r['model_calls'],tool_attempts=r['tool_attempts'],
            native_calls=r['native_calls'],valid_completion=r['valid_completion'],
            gpu_subprocess_seconds=r['gpu_subprocess_seconds'],stop_reason=r.get('stop_reason'),
            trajectory_gate_pass=False,whole_room_success='unknown')
        if 'final_audit' not in r or 'initial_review' not in r:
            row['evaluation']='incomplete';rows.append(row);continue
        executions=[x['execution'] for x in r['rounds'] if 'execution' in x]
        changes=sum(e['before_sha256']!=e['after_sha256'] for e in executions)
        score=score_transition(r['initial_review'],r['final_audit']['review'],attempts=r['tool_attempts'],
            native_calls=r['native_calls'],state_changes=changes,generated_tokens=r['generated_tokens'],
            gpu_seconds=r['gpu_subprocess_seconds'])
        if score!=r['transition_score']:raise ValueError('Stored transition score mismatch')
        state=read(folder/'final_state.json');rooms=read(folder/'restored_checkpoint/house_state.json')['rooms']
        if len(rooms)!=1:raise ValueError('Expected one restored room')
        initial=next(iter(rooms.values()))
        final_hash=canonical_hash(state);initial_hash=canonical_hash(initial)
        if final_hash!=r['final_audit']['state_sha256']:raise ValueError('Final measurement/state mismatch')
        preserved_ids=set(state['objects'])==set(initial['objects'])
        satisfied=score['initial_status']=='pass'
        full_pass=(r['status']=='policy_completed_pending_comparison' and r['valid_completion']
            and score['measured_success'] and not score['newly_failed'] and preserved_ids
            and (not satisfied or (changes==0 and initial_hash==final_hash)))
        refusals=[];rejections=[]
        for attempt in r['rounds']:
            if 'rejection' in attempt:rejections.append(attempt['rejection'])
            for native in attempt.get('reply',{}).get('results',[]):
                output=native.get('output',{})
                if isinstance(output,dict) and output.get('success') is False:
                    refusals.append(dict(tool=native.get('name'),output=output))
        row.update(trajectory_gate_pass=full_pass,transition=score,initial_state_sha256=initial_hash,
            final_state_sha256=final_hash,preserved_object_ids=preserved_ids,native_refusals=refusals,
            validation_rejections=rejections,terminal_scene=str(folder/'final_checkpoint'),
            final_state=str(folder/'final_state.json'))
        rows.append(row)
        req=read(folder/'restored_request.json');pixels=[]
        for message in req:
            if isinstance(message['content'],list):
                for part in message['content']:
                    if part['type']=='image':
                        pixels.append(image_hash(part['image']));part['image']='{ACTUAL_MATCHED_VIEW}'
        key=cell['condition'];signature=dict(nonimage_request=req,pixels=pixels)
        if key in matched and matched[key]!=signature:
            raise ValueError('Methods did not receive identical initial inputs: '+key)
        matched[key]=signature;paths[cell['cell_id']]=refs[str(folder/'restored_request.json')]
        matched_arms.setdefault(key,[]).append(cell['arm'])
    summary=dict(status='reviewed' if batch['status']=='completed_pending_review' else 'reviewed_incomplete_batch',
        independent_project_families=1,exposure='training-exposed diagnostic',seed=plan['seed'],
        planned_trajectories=len(plan['cells']),recorded_trajectories=len(rows),rows=rows,
        exact_initial_inputs_across_methods=all(len(v)==3 for v in matched_arms.values()) and len(matched_arms)==4,
        input_comparison_arms=matched_arms,initial_request_hashes=paths,input_hashes=refs,
        frozen_inputs_reverified=frozen,archived_artifacts_reverified=archived,
        model_calls=sum(r['model_calls'] for r in rows),gpu_subprocess_seconds=sum(r['gpu_subprocess_seconds'] for r in rows),
        training_updates=0,professional_gain_established=False,generalization_gain_established=False)
    if summary['model_calls']!=batch['model_calls'] or abs(summary['gpu_subprocess_seconds']-batch['gpu_subprocess_seconds'])>1e-6:
        raise ValueError('Batch/trajectory resource accounting differs')
    summary['arms']={arm:dict(trajectories=sum(r['arm']==arm for r in rows),
        valid_completion=sum(r['arm']==arm and r['valid_completion'] for r in rows),
        measured_pass=sum(r['arm']==arm and r.get('transition',{}).get('measured_success',False) for r in rows),
        satisfied_start_count=sum(r['arm']==arm and r.get('transition',{}).get('initial_status')=='pass' for r in rows),
        satisfied_start_gate_pass=sum(r['arm']==arm and r.get('transition',{}).get('initial_status')=='pass' and r['trajectory_gate_pass'] for r in rows),
        deficit_start_count=sum(r['arm']==arm and r.get('transition',{}).get('initial_status')=='fail' for r in rows),
        deficit_resolved=sum(r['arm']==arm and r.get('transition',{}).get('measured_incremental_gain',False) for r in rows),
        introduced_failure=sum(r['arm']==arm and bool(r.get('transition',{}).get('newly_failed')) for r in rows),
        trajectory_gate_pass=sum(r['arm']==arm and r['trajectory_gate_pass'] for r in rows))
        for arm in ('frozen_base','action_only','decision_action')}
    write(a.run/'SUMMARY.json',summary)
    lines=['# Matched complete-trajectory diagnostic','',
        '| Condition | Weights | Initial → final measured status | Valid completion | Gate | Calls |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        sc=r.get('transition',{});lines.append(f"| {r['condition']} | {r['arm']} | {sc.get('initial_status','?')} → {sc.get('final_status','?')} | {r['valid_completion']} | {r['trajectory_gate_pass']} | {r['model_calls']} |")
    lines += ['', '| Weights | Already-satisfied starts: gate passed | Deficit starts: measured repair | New hard failures |',
        '| --- | --- | --- | --- |']
    for arm,counts in summary['arms'].items():
        lines.append(f"| {arm} | {counts['satisfied_start_gate_pass']}/{counts['satisfied_start_count']} | {counts['deficit_resolved']}/{counts['deficit_start_count']} | {counts['introduced_failure']} |")
    lines += ['',f"Actual calls: {summary['model_calls']}; GPU subprocess seconds: {summary['gpu_subprocess_seconds']:.3f}; new training updates: 0.",'',
        'Gate requires declared activity pass, no new declared hard failure, original object identities and valid native completion. Satisfied starts additionally require no state changes. A refusal, exhausted budget or explanation alone is not completion.',
        '', 'One training-exposed project, four starts, one seed; these are not twelve independent projects. All methods use the same current inputs and feedback. Old adapters predate the matched/native-stop input changes. No new-training, professional quality, generalization or pairing-mechanism gain is established. Full-room comfort, physical moveability and other unknowns remain unknown.',
        '', 'The frozen completion gate is stricter than the official rollout parser: malformed no-tool emissions can terminate upstream but are rejected here. See PARSER_SENSITIVITY.json for a post hoc, zero-generation audit. It does not replace primary scores or assert an alternative trajectory was executed.']
    (a.run/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['arms']),flush=True)


if __name__=='__main__':main()
