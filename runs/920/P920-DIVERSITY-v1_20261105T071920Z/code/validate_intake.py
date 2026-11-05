import json,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
R=Path('/data/wyh/scene_cvpr_20261020/design_decisions_920/scenefunc');P=R/'runs/920/P920-DIVERSITY-v1_20261105T071920Z';now=datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
source=read(R/'research/920/decision_packets_v1/SOURCED_DECISIONS.json');layers=read(R/'research/920/decision_packets_v1/DATA_LAYERS.json');reg=read(R/'research/920/data_readiness_v1/PROJECT_REGISTRY.json');cover=read(R/'research/920/data_readiness_v1/KNOWLEDGE_COVERAGE.json');added=read(P/'NEW_KNOWLEDGE.json')['records'];plan=read(P/'AUDIT_PLAN.json')
counts=Counter(x['knowledge_category'] for x in source['records']);families={x['family_id']:x for x in reg['projects']};before=read(P/'REGISTRY_BEFORE.json')
checks={
'global_knowledge_records_and_ledger_agree':len(source['records'])==source['counts']['source_supported_knowledge']==layers['counts']['professional_knowledge_records']==11,
'category_counts_sum_and_match':sum(x['source_supported_knowledge_records_in_current_packet'] for x in cover)==11 and all(x['source_supported_knowledge_records_in_current_packet']==counts[x['category']] for x in cover),
'old_family_roles_preserved':all(families[x['family_id']]['allocation']==x['allocation'] for x in before['projects']),
'new_families_preallocated':all(families[x['family_id']]['split_fixed_at']==x['split_fixed_at']<x['derived_knowledge_at'] for x in added),
'new_records_identical_in_ledger':all(next(y for y in source['records'] if x['decision_id']==y['decision_id'])==x for x in added),
'pinned_inputs_still_match':all(sha(Path(x['path']))==x['sha256'] for x in plan['inputs'].values()),
'archived_cpu_outputs_exact':sha(P/'TOOL_SCOPE_AUDIT.json')==sha(R.parent/'runtime'/P.name/'audit/TOOL_SCOPE_AUDIT.json'),
'no_new_action_or_expert_label':read(P/'RESULT.json')['new_native_actions']==read(P/'RESULT.json')['new_executed_pairs']==read(P/'RESULT.json')['new_independent_professional_reviews']==0,
'no_new_training_or_loader_claim':not read(P/'RESULT.json')['actual_loader_checked'] and read(P/'RESULT.json')['training_updates']==0,
'executed_unique_decision_count_agrees':len(layers['executed_decision_pairs'])==source['counts']['unique_source_decisions_with_replay']==4,
'variant_count_agrees':sum(len(x.get('variants',[x])) for x in layers['executed_decision_pairs'])==source['counts']['source_grounded_replayed_episodes']==layers['counts']['executed_reconstruction_variants']==7,
'conditional_benefit_count_agrees':sum(bool(x.get('conditional_objective_benefit')) or any(v.get('conditional_objective_benefit',False) for v in x.get('variants',[])) for x in layers['executed_decision_pairs'])==3,
'actual_cpu_exit_zero':read(P/'execution/receipt.json')['exit_code']==0,
'no_sealed_test_relabel':len(reg['sealed_test_projects'])==0,
'new_QA_candidates_exact_and_unadmitted':len(list((P/'episodes').rglob('episode.json')))==2 and all(not read(x)['provenance']['training_eligible'] and not read(x)['provenance']['professional_review'] for x in (P/'episodes').rglob('episode.json')),
}
write(P/'FINAL_CHECKS.json',{'checked_at':now,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'scope':'Data integrity, attribution flags, counts and pinned interface audit only; no human or model outcome evaluation.'})
assert all(checks.values()),checks
files=[{'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(P.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']
write(P/'MANIFEST.json',{'created_at':now,'files':files,'excluded_self':True,'scope':'Committed small run files; source webpages not archived, no mesh/model artifacts created.'})
print(json.dumps({'passed':sum(checks.values()),'total':len(checks),'run_files':len(files),'bytes':sum(x['bytes'] for x in files)}))
