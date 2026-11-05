"""Register two sourced knowledge records, preserving zero executed actions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    now=lambda:datetime.now(timezone.utc).isoformat()
    print(now(),'register two authored knowledge decisions; no model or native tools',flush=True)
    cfg=json.loads(a.plan.read_text());p=Path(cfg['public_directory']);root=Path(cfg['repo_root'])
    for row in cfg['inputs']:
        if digest(row['path'])!=row['sha256']:raise ValueError('Input changed: '+row['path'])
    new=json.loads((p/'NEW_KNOWLEDGE.json').read_text())['records']
    corpus_path=root/'research/920/decision_packets_v1/SOURCED_DECISIONS.json';corpus=json.loads(corpus_path.read_text())
    layer_path=root/'research/920/decision_packets_v1/DATA_LAYERS.json';layers=json.loads(layer_path.read_text())
    registry_path=root/'research/920/data_readiness_v1/PROJECT_REGISTRY.json';registry=json.loads(registry_path.read_text())
    coverage_path=root/'research/920/data_readiness_v1/KNOWLEDGE_COVERAGE.json';coverage=json.loads(coverage_path.read_text())
    family='poznan_dlugi_lokal_mili_mlodzi_ludzie_2014'
    project=next(x for x in registry['projects'] if x['family_id']==family)
    old_ids={x['decision_id'] for x in corpus['records']}
    if len(corpus['records'])!=12 or len(new)!=2 or {x['family_id'] for x in new}!={family}:raise ValueError('Unexpected intake scope')
    if any(x['decision_id'] in old_ids for x in new):raise ValueError('Duplicate decision')
    for rec in new:
        if not (project['split_fixed_at']==rec['split_fixed_at']<rec['derived_knowledge_at']):raise ValueError('Split not fixed first')
        if rec['training_eligible'] or rec['training_authorized'] or rec['maturity']['source_grounded_action_replayed']:raise ValueError('Knowledge overpromoted')
        if rec['source_to_scene']['reconstructed_action_sequence'] is not None:raise ValueError('Invented execution')
    before_counts=dict(layers['counts']);stamp=now()
    corpus['records'].extend(new);corpus['updated_at']=stamp
    corpus['counts']['records_reviewed_here']=len(corpus['records'])
    corpus['counts']['source_supported_knowledge']=sum(x['maturity']['source_supported_knowledge'] for x in corpus['records'])
    corpus['counts']['prospectively_train_allocated_knowledge_records']=sum(x['allocation']=='train_source_candidate' for x in corpus['records'])
    layers['updated_at']=stamp
    layers['counts']['professional_knowledge_records']=corpus['counts']['source_supported_knowledge']
    layers['counts']['prospectively_train_allocated_knowledge_records']=corpus['counts']['prospectively_train_allocated_knowledge_records']
    layers['additional_knowledge_intakes']=layers.get('additional_knowledge_intakes',[])+[{
        'run_id':cfg['run_id'],'knowledge_records':2,'source_families':1,'knowledge_QA_candidates':2,
        'native_actions':0,'independent_professional_reviews':0,'training_admitted':0,
        'evidence':str((p/'NEW_KNOWLEDGE.json').relative_to(root))}]
    project.update(derived_knowledge_records=2,source_reviewed_at=stamp,
        knowledge_decision_ids=[x['decision_id'] for x in new],derived_knowledge_QA_candidates=2,
        knowledge_record=str(corpus_path.relative_to(root)),
        action_qualification='Knowledge-only roof/raised-bed mechanisms retained; earlier zoning rejection remains valid.')
    registry['updated_at']=stamp
    categories=Counter(x['knowledge_category'] for x in corpus['records'])
    for row in coverage:
        row['source_supported_knowledge_records_in_current_packet']=categories[row['category']]
        if row['category']=='cost_construction_maintenance':
            row['current_evidence_or_gap']='Długi Lokal architect describes exterior roof insulation to preserve attic volume and existing exposed structure. Knowledge only; no thermal/moisture/structural/cost or native roof-construction validation. Earlier Malaga/source-reuse limitations remain.'
        elif row['category']=='sightlines_and_daylight':
            row['current_evidence_or_gap']='Biombombastic and Długi Lokal raised bed/window-view decisions are attributed knowledge. Bed-height/headroom/eye pose and working drawer geometry remain unknown; no native elevated-bed action. Ono sampled screening is a cross-reference, not another source record or daylight measurement.'
    changed_counts={k:[before_counts[k],v] for k,v in layers['counts'].items() if before_counts.get(k)!=v}
    if set(changed_counts)!={'professional_knowledge_records','prospectively_train_allocated_knowledge_records'}:raise ValueError('Unintended count promotion')
    if sum(categories.values())!=14:raise ValueError('Coverage mismatch')
    # All validation precedes mutation. Original registries are archived by intake.
    for path,value in [(corpus_path,corpus),(layer_path,layers),(registry_path,registry),(coverage_path,coverage)]:write(path,value)
    result={'run_id':cfg['run_id'],'status':'registered_knowledge_only','completed_at':now(),
        'plan_sha256':digest(a.plan),'new_records':2,'new_source_families_registered':0,'source_families_in_this_intake':1,
        'counts_changed':changed_counts,'knowledge_total':14,'coverage_by_primary_category':dict(categories),
        'new_knowledge_QA_candidates':2,'new_native_actions':0,'model_calls':0,'training_updates':0,
        'independent_professional_reviews_added':0,'formal_training_admitted':0,
        'source_qualification_scope':'Assistant checked architect-authored statements; no external designer endorsement or outcome validation.',
        'registry_after_hashes':{str(x.relative_to(root)):digest(x) for x in [corpus_path,layer_path,registry_path,coverage_path]}}
    write(p/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
