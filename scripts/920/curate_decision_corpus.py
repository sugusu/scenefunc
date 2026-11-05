"""Derive a source-only candidate corpus from locked executions; no model or replay."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def locked(root,ref):
    p=root/ref['path']; assert sha(p)==ref['sha256'],str(p); return read(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    plan=read(a.plan);repo=Path(__file__).resolve().parents[2];base=repo.parent
    pub=repo/'runs/920'/plan['run_id'];out=base/'runtime'/plan['run_id']/'dataset'
    assert sha(Path(__file__))==plan['code_sha256'];assert not out.exists()
    out.mkdir(parents=True);(pub/'dataset').mkdir();rows=[];now=datetime.now(timezone.utc).isoformat()
    source_index=locked(repo,plan['construction_index']);old=locked(repo,plan['old_edit_manifest'])
    provenance=locked(repo,plan['source_records']);source_by_id={x['decision_id']:x for x in provenance['records']}
    originals=[]
    for case in source_index['cases']:
        data=locked(repo,dict(path=case['candidate_manifest'],sha256=case['candidate_manifest_sha256']))
        root=Path(case['server_input_root'])
        for row in data['rows']:
            ref=row['documents'].get('paired',row['documents'].get('shared'))
            originals.append((case['case']+'_build',row['step'],locked(root,ref),root,ref,row))
    for row in old['decisions']:
        group,step=row['id'].split('_step_')
        originals.append((group+'_edit',int(step),locked(base,row['paired']),base,row['paired'],row))
    for group,step,original,root,ref,source_row in originals:
        uid=f'{group}_{step:02d}';spec=plan['groups'][group];doc=deepcopy(original)
        context=json.loads(doc['task_context']['scene_description']);oldcontext=deepcopy(context)
        if spec.get('replacement_brief'):
            context['design_request']=spec['replacement_brief']
        if group=='waterloo_edit':
            context['current_requested_mode']='four_person_shared_activity' if step<=2 else 'two_simultaneous_individual_workplaces'
        doc['task_context']['scene_description']=json.dumps(context,ensure_ascii=False,sort_keys=True)
        reason=spec['reason'] if step==spec.get('paired_step') else ''
        source=source_by_id[spec['decision_id']]
        doc['provenance'].update(kind='source_only_curated_reconstruction_candidate',
            family_id=source['family_id'],allocation=source['allocation'],
            reason_origin='attributed_source_paraphrase' if reason else 'none_action_only',
            training_eligible=False,training_authorized=False,
            task_reconditioned=context!=oldcontext,original_episode=dict(path=str(root/ref['path']),sha256=ref['sha256']))
        doc['evaluation_only']['curation']=dict(original_reason=original['turns'][0]['content'],
            original_task_context=oldcontext,supervision_class='source_linked_sequence_decision' if reason else 'ordinary_executed_action',
            reason_scope=spec['scope'],task_revision_scope='Research-authored demonstration relabeling; no new execution or model success claimed.')
        images=[]
        for path in original['turns'][0]['images']:
            q=(root/path).resolve();images.append(dict(path=str(q),sha256=sha(q)))
        assert len(images)==5
        assert [x['sha256'] for x in images]==[x['sha256'] for x in source_row['images']]
        doc['turns'][0]['images']=[x['path'] for x in images]
        doc['turns'][0]['content']=reason
        documents={}
        for kind in (['reason','paired','separate'] if reason else ['shared']):
            sample=deepcopy(doc)
            if kind=='reason':sample['turns'][0]['tool_calls']=[]
            if kind in ['separate','shared']:sample['turns'][0]['content']=''
            path=out/'episodes'/uid/kind/'episode.json';write(path,sample)
            documents[kind]=dict(path=str(path.relative_to(base)),sha256=sha(path),supervised_role='reason' if kind=='reason' else 'action')
            shutil.copy2(path,pub/'dataset'/f'{uid}_{kind}.json')
        assert doc['turns'][0]['tool_calls']==original['turns'][0]['tool_calls']
        rows.append(dict(id=uid,group=group,step=step,family_id=source['family_id'],decision_id=source['decision_id'],
            source_episode=dict(path=str(root/ref['path']),sha256=ref['sha256']),images=images,documents=documents,
            command=doc['turns'][0]['tool_calls'],reason=reason,reason_scope=spec['scope'],
            input_change=context!=oldcontext,original_reason_removed=original['turns'][0]['content']!=reason,
            consequences=spec['consequences'],choice_limits=spec['choice_limits']))
    arms={arm:[] for arm in ['separate','paired']}
    for row in rows:
        for arm in arms:
            if row['reason']:arms[arm].extend([row['documents']['reason'],row['documents'][arm]])
            else:arms[arm].append(row['documents']['shared'])
    assert len(rows)==36 and sum(bool(x['reason']) for x in rows)==5
    counts=dict(actions_per_arm=36,source_reason_occurrences_per_arm=5,unique_source_reasons=4,
        source_families=4,execution_groups=6,ordinary_action_only=31,samples_per_arm=41,
        newly_executed_actions=0,new_model_calls=0,training_updates=0)
    data=dict(schema='curated-source-action-corpus-v1',created_at=now,audit_only=True,training_authorized=False,
        root=str(base),counts=counts,rows=rows,arms=arms,
        caveat='Same existing executions, newly separated supervision. Not new projects, expert labels, or newly executed goal-conditioned trajectories.')
    write(pub/'DATASET.json',data)
    write(pub/'SOURCE_KNOWLEDGE.json',{key:source_by_id[value['decision_id']] for key,value in plan['groups'].items()})
    write(pub/'CURATION_CHECKS.json',dict(recorded_at=now,plan_sha256=sha(a.plan),counts=counts,
        all_actions_identical_to_source=True,all_five_view_hashes_identical=True,
        reason_occurrences=[x['id'] for x in rows if x['reason']],
        changed_tasks=[x['id'] for x in rows if x['input_change']],
        source_inputs_not_modified=True,original_mixed_reasons_preserved_in_evaluation_only=True))
    print(json.dumps(counts))


if __name__=='__main__':main()
