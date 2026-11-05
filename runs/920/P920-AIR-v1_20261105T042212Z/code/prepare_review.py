"""Export the actual mixed consequence as an objective QA candidate, not praise."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import sys

P = Path(__file__).resolve().parent.parent; R = P.parents[2]; D = R.parent
T = D/'runtime'/P.name; replay = T/'replay'
sys.path.insert(0, str(R/'scripts/920'))
from audit_native_review import export

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ref = lambda p: {'path': str(p.resolve().relative_to(D)), 'sha256': sha(p)}
def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

summary = read(replay/'SUMMARY.json'); frozen = read(P/'FROZEN_PROGRAM.json')
assert [r['before'] for r in summary['fixed_point_access']] == [7,7,7]
assert [r['after'] for r in summary['fixed_point_access']] == [7,6,6]
assert [r['lost'] for r in summary['fixed_point_access']] == [[], ['dining_east'], ['dining_east']]
candidates, measurements, evidence = [], {}, []
for label in ('initial', 'after'):
    obs = read(replay/(label+'_observation.json'))
    candidates.append({'candidate_id': label, 'state': ref(replay/(label+'_state.json')),
        'observation': ref(replay/(label+'_observation.json')), 'context': ref(replay/(label+'_context.json')),
        'binding': ref(Path(obs['state_binding']))})
    m = read(replay/(label+'_measurement.json'))
    measurements[label] = [{'body_width_m': s['square_body_width_m'], 'targets': s['targets'],
                            'reachable_floor_center_area_m2': s['reachable_floor_center_area_m2']} for s in m['scenarios']]
    evidence.append({'evidence_id': label+'_access', 'artifact': ref(replay/(label+'_measurement.json'))})
    save(P/(label.upper()+'_VIEWS.json'), {'state': candidates[-1]['state'], 'binding': candidates[-1]['binding'],
        'images': read(Path(obs['state_binding']))['images'], 'native_ID_annotations': True})
for label, path in [('rail_approach', replay/'SUMMARY.json'), ('actual_action', replay/'STEP.json'),
                    ('asset_visual', P/'ASSET_VISUAL_REVIEW.json')]:
    evidence.append({'evidence_id': label, 'artifact': ref(path)})
measurements['added_asset'] = {'object_id': summary['added_object_id'],
    'dimensions_m': summary['unscaled_dimensions_m'], 'floor_AABB_m2': summary['added_floor_AABB_m2'],
    'rail_approaches': summary['rail_approaches']}
claims = [
    {'candidate_id': 'after', 'object_ids': ['open_clothes_rail_0'], 'activity': '接近新增的开放挂衣构件',
     'assessment': '新增未缩放开放双杆衣架。0.4米方形代理的两侧10/10采样点可达；0.5和0.6米均只有局部+Y侧5/10可达。可达位置不证明拿取速度、挂衣容量、承重或舒适。',
     'evidence_ids': ['actual_action', 'asset_visual', 'rail_approach']},
    {'candidate_id': 'after', 'object_ids': ['open_clothes_rail_0', 'recon_chair_east_0', 'recon_table_0'], 'activity': '从入口接近东侧餐位',
     'assessment': '七个固定点可达数在0.4米下保持7/7，在0.5和0.6米下由7/7降至6/7，失去的都是dining_east。该目标格本身仍为空，但从入口没有代理路径。不能把无家具重叠当作通行已保留。',
     'evidence_ids': ['initial_access', 'after_access']},
    {'candidate_id': 'after', 'object_ids': ['open_clothes_rail_0'], 'activity': '安排衣物存放并保留其他用途',
     'assessment': '原23个对象均保留。落地替代物增加约0.396平方米包围盒占地；它与原项目挂杆的安装方式不同。当前摆位具有局部供给和接近损失，不能标作整体正向建设示范。',
     'evidence_ids': ['actual_action', 'rail_approach']},
]
record = {'review_id': P.name+'_objective_consequences', 'created_at': datetime.now(timezone.utc).isoformat(),
    'provenance': {'kind': 'objective_geometric_review_of_source_guided_reconstruction',
        'project_id': 'budapest_airbnp_position_collective_2015', 'family_id': 'budapest_airbnp_position_collective_2015',
        'allocation': 'train_source_candidate', 'unique_review_unit': 1, 'professional_review': False,
        'training_eligible': False, 'training_authorized': False, 'qualification': 'Objective QA candidate; not admitted training data.'},
    'brief': read(replay/'initial_context.json')['design_request'],
    'source_context': {'url': read(P/'SOURCE_RECORD.json')['professional_evidence']['primary_crosscheck']['url'],
        'supported': '短住时将常用衣物放在开放挂杆以便取用；其余物品可入床下储物。原文未提供本重建尺寸、初态、落地替代物或坐标。'},
    'question': '根据前后五视图、对象和量测，指出新增供给、受损活动和未知项。能否给出整体专业偏好？使用initial/after稳定ID。',
    'assumptions': frozen['assumptions']+['目标格为空与路径存在不同；两种判断都来自给定工具结果。采样点不是人、座位或独立实验。'],
    'candidates': candidates, 'measurement_input': measurements,
    'judgment': {'verdict': 'insufficient_evidence', 'ranking_groups': [], 'claims': claims, 'evidence': evidence,
        'unknowns': ['无独立专业人员对本重建的评价。', '方形身体和包围盒不能确证真实人体可达、拿取或舒适。',
                     '尺寸为假设，原项目安装方式、床内容量和完整住宅设施工作未验证。']}}
save(P/'OBJECTIVE_REVIEW.json', record)
episode = export(record, D, ['initial', 'after'])
save(P/'QA_EPISODE_CANDIDATE.json', episode)
save(P/'REVIEW_EXPORT_CHECKS.json', {'native_state_context_image_bindings_checked': True,
    'named_objects_and_evidence_exist': True, 'verdict': 'insufficient_evidence', 'images': 10,
    'new_official_loader_run': False, 'new_model_calls': 0, 'training_updates': 0,
    'scope': 'Reused validated exporter; this new episode has not been re-tokenized or admitted for training.',
    'episode_sha256': sha(P/'QA_EPISODE_CANDIDATE.json')})
print('one source-linked objective review candidate exported; no positive placement target or training')
