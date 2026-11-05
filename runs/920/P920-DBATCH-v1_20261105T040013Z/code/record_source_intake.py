"""Record the three primary-source readings, without deriving scenes or targets."""
from pathlib import Path
import json
import hashlib
import re
from urllib.parse import urlsplit
from datetime import datetime, timezone

R = Path(__file__).resolve().parents[4]
P = Path(__file__).resolve().parent.parent
now = datetime.now(timezone.utc).isoformat()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')


specs = [
    dict(case='M.S.2', decision_id='schemaa_functional_bands_integrated_storage', category='zoning',
         aliases=['SCHEMAA', 'Appartement C', 'Apartment C'],
         attribution='SCHEMAA architect-owned project description; Paris 75017; delivered 2014',
         locator='PRÉSENTATION DU PROJET: functional bands, stair storage and detail paragraphs',
         condition='巴黎阁楼住宅的整体改造，需要组织入口、起居、厨房与上层寝居，并纳入收纳。',
         reason='用兼具储物的功能带组织空间，使各个使用单元清晰可辨。',
         decision='入口与起居间设置通高厨房带；通往阁楼的楼梯下整合收纳。',
         limit='这是整合建筑与定制家具的设计；移动普通柜子不等于复现楼梯、上下层交通、厨房安装或施工。',
         identity='Primary project Appartement C, SCHEMAA, Paris 75017 and delivered 2014 match M.S.2. Generic Apartment C substring hits are not project identities.',
         primary_year=2014),
    dict(case='M.S.3', decision_id='guilbault_consolidated_storage_for_continuous_living_space', category='storage_and_existing_assets',
         aliases=['Richard Guilbault', '30m2 Flat in Paris', '30 m2 flat in Paris'],
         attribution='Richard Guilbault; ArchDaily explicitly attributes the description to architects; Paris, 2015',
         locator='Architect-provided description paragraphs 1–3 after project metadata',
         condition='年轻伴侣的约30平方米住宅原先分为四个昏暗房间，改造同时追求明亮起居、私密空间与收纳。',
         reason='重组空间并集中技术与储物需求，为连续的起居空间释放位置。',
         decision='沿整套住宅设置多用途储物墙，配合新的空间组织，使起居区能利用四扇窗。',
         limit='原文涉及完整改造；家具集中摆放只能表达局部。论文表面积31.7平方米、项目正文30平方米分别保留；不把有窗等同于实测采光。',
         identity='Primary title, architect, Paris and 2015 match M.S.3. Primary 30m² and indexed 31.7m² remain distinct reported values.',
         primary_year=2015),
    dict(case='M.S.9', decision_id='airbnp_open_clothing_access_for_short_stays', category='storage_and_existing_assets',
         aliases=["AIRB'N'P", 'AIRB’N’P', 'POSITION Collective', 'POSITION Concept'],
         attribution='POSITION Concept/Collective architect-owned first-person explanation; Budapest; date not shown on checked page',
         locator='Full story: short-stay purpose and paragraph explaining express clothing use',
         condition='为两位旅客短期住宿设计的紧凑布达佩斯公寓，衣物需要方便、快速取用。',
         reason='短住时不一定需要将经常使用的衣物收进封闭柜内。',
         decision='设置便于随手取用衣物的金属挂杆，其余物品可放入床下收纳。',
         limit='不能推广成长期居住无需衣柜；挂杆承重、可达高度、床内容量与实际拿取未验。摆放静态替代物尚不构成功能动作示范。',
         identity='Primary AIRB’N’P Budapest two-traveller apartment matches M.S.9. Year 2015 comes from scholarly index, not a displayed primary-page date.',
         primary_year=None),
]

assigned = read(P/'PROSPECTIVE_TRAIN_FAMILIES.json')
registry_path = R/'research/920/data_readiness_v1/PROJECT_REGISTRY.json'
registry = read(registry_path)
index = read(R/'research/920/microapartments_v1/SOURCE_INDEX.json')
entries = {e['source_case_id']: e for e in index['entries']}
metadata = read(R/'research/920/public_sources_v1/FAMILY_METADATA_AUDIT.json')
reference = Path(metadata['reference'])
assert sha(reference) == metadata['reference_sha256']
folders = read(reference)['folders']
metadata_text = {k: json.dumps({a: b for a, b in v.items() if a != 'files'}, ensure_ascii=False)
                 for k, v in folders.items()}
normalize = lambda u: (urlsplit(u).netloc.lower().removeprefix('www.')+urlsplit(u).path.rstrip('/')).lower()
records, audits = [], []
for spec, assignment in zip(specs, assigned['assignments']):
    assert spec['case'] == assignment['source_case_id']
    entry = entries[spec['case']]
    assert sha(Path(entry['source_fragment']['local_fragment'])) == entry['source_fragment']['sha256']
    exact = [k for k, urls in metadata['reference_url_index'].items()
             if normalize(assignment['source_url']) in [normalize('https://'+u) for u in urls]]
    hits = {k: [a for a in spec['aliases'] if re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)', v, re.I)]
            for k, v in metadata_text.items()}
    hits = {k: v for k, v in hits.items() if v}
    assert not exact and not hits, (spec['case'], exact, hits)
    snippets = {k: [v[max(0, m.start()-40):m.end()+60] for m in re.finditer('apartment c', v, re.I)][:5]
                for k, v in metadata_text.items() if 'apartment c' in v.lower()} if spec['case'] == 'M.S.2' else {}
    evidence = {'evidence_kind': 'architect_project_statement',
                'primary_crosscheck': {'url': assignment['source_url'], 'attribution': spec['attribution'],
                                      'locator': spec['locator'], 'checked_at': now},
                'corroborating_project_index': {'doi': index['source_doi'], 'source_case_id': spec['case'],
                    'fragment': entry['source_fragment'], 'scope': 'Authored article CC BY4.0; original primary text and photographs have separate rights.'},
                'raw_resident_brief_available': False,
                'reuse_scope': 'Own brief attributed factual paraphrase for source review; no original images copied. Web access and article license do not establish training reuse clearance.'}
    record = {'decision_id': spec['decision_id'], 'family_id': assignment['family_id'],
              'allocation': 'train_source_candidate', 'knowledge_category': spec['category'],
              'professional_evidence': evidence, 'condition_summary': spec['condition'],
              'condition_origin': 'retrospective professional project account, not a raw resident brief',
              'rationale_summary': spec['reason'], 'rationale_origin': 'researcher paraphrase of cited professional explanation',
              'decision_scope': {'source_decision': spec['decision'], 'unsupported_part': spec['limit']},
              'tool_expression': {'level': 'knowledge_only_pending_subdecision_qualification',
                                  'executable_subdecision': None, 'full_source_decision_supported': False},
              'source_to_scene': {'initial_proxy_ids': None, 'historical_action_sequence': None, 'reconstructed_action_sequence': None},
              'maturity': {'source_supported_knowledge': True, 'source_to_native_objects_bound': False,
                           'source_grounded_action_replayed': False, 'whole_room_consequences_reviewed': False,
                           'independent_professional_preference_label': False, 'paired_episode_ready': False},
              'readiness_by_layer': {'professional_knowledge': {'supported': True,
                    'use': 'Attributed knowledge; a second designer is not required to recognize the original statement.'},
                  'professional_evaluation': {'available': False}, 'scene_action': {'executed': False},
                  'professional_decision_pair': {'executed': False, 'positive_demonstration': False}},
              'split_fixed_at': assignment['split_fixed_at'], 'derived_knowledge_at': now,
              'training_eligible': False, 'training_authorized': False,
              'training_eligible_scope': 'Core training admission is distinct from knowledge usefulness and prospective family allocation.',
              'next_action': 'Qualify a concrete representable subdecision and its verification before deriving an action target; retain knowledge regardless of missing full 3D.'}
    records.append(record)
    audits.append({'family_id': assignment['family_id'], 'source_case_id': spec['case'],
                   'primary_url': assignment['source_url'], 'primary_attribution': spec['attribution'],
                   'primary_identity_review': spec['identity'], 'primary_year': spec['primary_year'],
                   'indexed_year': assignment['reported_year'], 'existing_33_metadata_exact_url_matches': exact,
                   'word_boundary_alias_matches': hits, 'aliases_checked': spec['aliases'],
                   'generic_substring_false_positive_contexts': snippets,
                   'independence_limit': 'Known-metadata check plus primary identity review, not universal duplicate exclusion; shared scholarly-source correlation retained.',
                   'project_role_fixed_before_extraction': assignment['split_fixed_at'] < now,
                   'derived_native_episodes': 0})
    project = next(p for p in registry['projects'] if p['family_id'] == assignment['family_id'])
    project.update(identity_status=spec['identity'], source_reviewed_at=now,
                   knowledge_decision_id=spec['decision_id'], derived_knowledge_records=1,
                   derived_native_episodes=0, derived_training_samples=0,
                   knowledge_record='research/920/decision_packets_v1/SOURCED_DECISIONS.json')

ledger_path = R/'research/920/decision_packets_v1/SOURCED_DECISIONS.json'
ledger = read(ledger_path)
assert len(ledger['records']) == 6 and len(records) == 3
assert not (P/'NEW_KNOWLEDGE.json').exists()
save(P/'PRIMARY_SOURCE_AUDIT.json', {'checked_at': now, 'metadata_reference': {'path': str(reference), 'sha256': sha(reference)},
     'assignments_sha256': sha(P/'PROSPECTIVE_TRAIN_FAMILIES.json'), 'records': audits,
     'new_knowledge_records': 3, 'new_native_episodes': 0, 'professional_reviews': 0,
     'training_admitted': 0, 'original_media_downloaded': 0, 'source_correlation_group': 'buildings11120654'})
save(P/'NEW_KNOWLEDGE.json', {'created_at': now, 'records': records})
ledger['records'] += records
ledger['updated_at'] = now
ledger['counts'].update(records_reviewed_here=9, source_supported_knowledge=9, prospectively_train_allocated_knowledge_records=3)
ledger['count_scope'] = 'Nine knowledge records: six development and three prospective train-source records; no claim of nine qualified action pairs.'
registry['updated_at'] = now
save(ledger_path, ledger)
save(registry_path, registry)
layers_path = R/'research/920/decision_packets_v1/DATA_LAYERS.json'
layers = read(layers_path)
layers['updated_at'] = now
layers['counts'].update(professional_knowledge_records=9, prospectively_train_allocated_knowledge_records=3)
layers['counting_notes'][0] = 'Nine knowledge records are not nine action pairs. Three new source families received a prospective train role before extraction; no action episodes or training admission yet.'
layers['development_batch_audit'] = {'run_id': P.name, 'result': str((P/'LOADER_RESULT.json').relative_to(R)),
    'examples_per_arm': 9, 'supervised_tokens_per_arm': 1522, 'image_instances_per_arm': 20,
    'native_calls_reused': 4, 'new_native_calls': 0, 'training_updates': 0}
save(layers_path, layers)
print(json.dumps({'knowledge_records': 9, 'families': len(registry['projects']), 'new_native_episodes': 0}))
