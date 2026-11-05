"""Inspect existing data's causal contrast and evidence coverage; never encode or train."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); plan=read(a.plan); root=Path(plan['root']); repo=Path(plan['repo'])
    assert sha(Path(__file__))==plan['code_sha256']
    for ref in plan['inputs'].values():
        assert sha(repo/ref['path'])==ref['sha256'],ref['path']
    print(datetime.now(timezone.utc).isoformat(),'inspect locked source/action records; no encoding or optimization',flush=True)
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]; inventory=Counter(); anchors=[]
    for key in ['curated','lifecycle']:
        data=read(repo/plan['inputs'][key]['path'])
        for row in data['rows']:
            docs={}
            for kind,ref in row['documents'].items():
                f=root/ref['path']; assert sha(f)==ref['sha256']; docs[kind]=read(f)
            action_doc=docs.get('separate',docs.get('shared'))
            action=action_doc['turns'][0]['tool_calls']; assert len(action)==1
            inventory[action[0]['name']]+=1
            family=row.get('family_id',action_doc['provenance'].get('family_id'))
            assert family
            identifier=row.get('id',f'lifecycle_{row["step"]:02d}')
            item=dict(id=identifier,family_id=family,action=action[0],has_source_reason='reason' in docs,
                evidence_kind='researcher_reconstructed_actions',source_files=row['documents'])
            if 'reason' in docs:
                r,p,s=(docs[k] for k in ['reason','paired','separate'])
                assert r['turns'][0]['content']==p['turns'][0]['content'] and not s['turns'][0]['content']
                assert not r['turns'][0]['tool_calls'] and p['turns'][0]['tool_calls']==s['turns'][0]['tool_calls']
                contexts=[]
                for doc in [r,p,s]:
                    contexts.append(dict(system=doc['system'],task_context=doc['task_context'],images=doc['turns'][0]['images']))
                assert contexts[0]==contexts[1]==contexts[2]
                assert len(contexts[0]['images'])==5
                anchor=dict(id=identifier,family_id=family,source_reason=r['turns'][0]['content'],
                    reason_and_action_share_full_task_context=True,reason_and_action_share_five_images=True,
                    reason_context_is_not_disconnected_case_knowledge=True,
                    manipulated_edge='teacher-forced reason prefix -> action prediction',
                    choice_limit=row.get('choice_limits','Task explicitly permits removal of no-longer-required bed; no executed alternative for continued sleeping demand.'),
                    inference='Same-scene reason teaching occurs in both arms; no claim that separate lacks project association.')
                anchors.append(anchor)
            rows.append(item)
    assert len(rows)==39 and len(anchors)==6
    assert len({x['family_id'] for x in rows})==5
    review=read(repo/plan['inputs']['conditional_review']['path'])
    assert len(review['records'])==2
    # A conditional pair already exists, but its opposite requirement is constructed,
    # and neither task is an independent held-out whole dwelling.
    comparison=read(repo/plan['inputs']['conditional_summary']['path'])
    assert comparison['choices']['perimeter']['clear_rays']==15
    assert comparison['choices']['divider']['clear_rays']==0
    registry=read(repo/plan['inputs']['registry']['path'])
    episodes=read(repo/plan['inputs']['native_episodes']['path'])
    discrepancies=[]
    for family,prefixes in [('osaka_f_house_coilkma',['P920-FCHOICE-']),('milan_paper_paper_associates',['P920-PAPER-'])]:
        record=next(x for x in registry['projects'] if x['family_id']==family)
        matches=[x for x in episodes['episodes'] if any(x['run_id'].startswith(prefix) for prefix in prefixes)]
        for ref in matches:
            assert sha(repo/ref['artifact'])==ref['sha256']
        actual=len(matches)
        if record['derived_native_episodes']!=actual:
            discrepancies.append(dict(family_id=family,registry_before=record['derived_native_episodes'],
                ledger_executed_reconstructions=actual,source_linked=sum(x['scope']!='researcher_contrast_to_source_principle' for x in matches),
                evidence=matches,meaning='Executed reconstruction count, including retained failed versions and contrasts, not independent source decisions.'))
    report=dict(status='existing_material_and_intervention_audit_completed',created_at=datetime.now(timezone.utc).isoformat(),
        counts=dict(existing_actions=39,source_reason_positions=6,unique_source_reasons=len(set(x['source_reason'] for x in anchors)),
            source_families=5,ordinary_actions=33,new_executions=0,training_updates=0,model_calls=0),
        tool_counts=dict(inventory),anchors=anchors,
        conclusion='Current comparison isolates same-sequence reason conditioning of action, conditional on both groups learning scene-grounded reasons. It does not isolate presence versus absence of project/state association.',
        conditional_evidence=dict(family='osaka_f_house_coilkma',conditions=2,candidates=2,source_supported_condition='shared living',
            opposite_condition='researcher-constructed reduced visual distraction',shared_scene=True,independent_project_count=1,
            professional_labels=0,whole_dwelling=False,not_in_curated_ota=True,
            use='Existing objective conditional review, not extra professional reasons or independent evaluation. Do not regenerate the same layouts.'),
        registry_discrepancies=discrepancies,
        next_required_evidence='Independent project decision tasks with more than one executable response, source-supported conditions, full-room side-effect accounting and honest simple baselines; not more equivalent action rows or repeated loader audits.',
        token_counts_recomputed=False,combined_training_dataset_created=False,new_training_authorized=False)
    write(a.out/'RESULT.json',report);write(a.out/'ACTION_INVENTORY.json',rows)
    print(json.dumps(dict(counts=report['counts'],tool_counts=report['tool_counts'],registry_discrepancies=len(discrepancies))),flush=True)


if __name__=='__main__':
    main()
