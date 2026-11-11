"""Source-aware audit of all archived Function2Scene public-viewer briefs.

Manual source reading is declared as author self-review, not independent expert
labels. This script checks provenance, selected original/derived spans and a
static box support fact; it does not generate rooms or execute external rewards.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import zipfile

R = Path(__file__).resolve().parents[2]
W = R.parent.parent
SOURCE = W.parent / 'systematic_scene_literature_20261029/scene_designer_context_review_20261102/method_feasibility_review/objective_theory_review/public_functional_case/EMBEDDED_DATA.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def textsha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


# Reviewed against complete ad fields, not the generated constraints. Paragraph
# indices are one-based nonempty lines excluding attribution links/comments.
# Persistent means an explicit in-progress material/build survives sessions;
# ordinary decorative storage or an existing fixture does not count.
REVIEW = [
 ('01_2-boys_bedroom', False, True, 'none', [2,3,6], '兄弟错时阅读和睡眠，个人灯与储物须对应本人；两人游戏需要剩余地面。', '灯光与声学没有由普通碰撞检查证明。'),
 ('02_antique-buyer_guestroom', False, False, 'furniture_reposition', [3,4,7], '可移动沙发和电视；临时客人使用气垫床，允许只移动一两件家具腾地方。', '没有规定家具位移轨迹或床垫尺寸，不得自行补为用户真值。'),
 ('03_antique-dealer-kid_kitchen', False, True, 'none', [3,5,6], '成人做饭时孩子在岛台学习；备餐与就座需同时使用岛台，临时会谈也用该空间。', '儿童可达高度与材料区域大小未量化，不能从家具类别直接判功能通过。'),
 ('04_artist_atelier', True, False, 'none', [2,5], '绘画和陶艺需分配资源；陶艺作品晾干时不能每次绘画都挪动，工作面又需保持清空。', '持久作品、活动工作面和工具储存要区分；未给作品尺寸或数量。'),
 ('05_boy_bedroom', True, False, 'none', [2,4,5,6], '未完成模型须跨制作时段保留组装状态，此外还有学习和朋友游戏空间。', '不拆装不等于永远固定XYZ；原文没有禁止整体转移到项目架。'),
 ('06_business-traveller_studio', False, False, 'fold_articulate', [3,4,5,6,7], '工作和休息需迅速切换，床白天收起，桌面可兼工作和用餐，物品闲置时收纳。', '折叠床与遮藏机构超出当前刚性物体平移的验证范围。'),
 ('07_chef_diningroom', False, True, 'none', [2,3,4,5], '烹饪时保持与客人交流；饮品区与备餐分离；餐桌能正式坐八人。', '一两人时亲密感不是允许任意缩放桌子的数值授权。'),
 ('08_collector_library', False, False, 'furniture_reposition', [3,4,5], '火边阅读、聚会和五六人游戏；游戏桌闲置时应完全让开。', '游戏桌收起方式未指定，不应预设一定为可折叠资产。'),
 ('09_couple_dining', True, False, 'material_relocation', [2,3,4], '摄影设备、进行中陶艺和用餐争用空间；保留项目，待客时允许移到附近临时面而非完全打包。', '这不是始终同一桌面保留所有物体；临时去向须真实可用，人数不能由多个活动名称相加。'),
 ('10_dancer-studio', False, False, 'missing_brief', [], '该别名条目缺少ad；实际完整舞者条目另存为10_dancer_studio。', '不算一个可读空白用户任务，也不从相似ID补写原文。'),
 ('11_designer-couple_bedroom', False, True, 'screen_concealment', [2,3,4,6], '保留固定小凳和两块石头；夫妻错时阅读睡眠；电视只在需要时可见。', '固定装饰物不是跨活动进行中材料，需避免扩大持久项目类别。'),
 ('12_designer_livingroom', False, False, 'none', [3,4,5,6], '独居者的多面景观和光照、陈列轮换、偶尔待客；睡眠时需避免客厅光声干扰。', '没有第二常住人物；不能把同一人的工作和睡眠当两个同时存在的人。'),
 ('13_family_kitchen', False, True, 'none', [2,3,4,6], '两名成人可并行备餐，儿童就座与烹饪安全分区，餐饮服务路线也需并行。', '食材流程有短时占用，但未明说跨制作时段保存进行中项目。'),
 ('14_family-young-kid_kitchen', False, True, 'material_storage_reset', [2,3,4], '做饭与儿童学习手工并行；手工材料需迅速收回专门储物区域。', '清空工作面不等于删除必需材料实例，也没有给出储物容量数字。'),
 ('15_fitness-coach_livingroom', False, False, 'furniture_reposition', [2,3], '一分钟内腾出练习地面，并保留平时家庭起居用途和两只狗的需求。', '实际物理搬运时间无法由一次坐标修改耗时替代。'),
 ('16_jet-setting-fam_bedroom', False, True, 'none', [2,3], '夫妻不同时间归来，另一人可能已睡；床与储物需紧凑整合。', '不同气候衣物不是多个同时到场的人或新的活动站点。'),
 ('17_kids_attic-bedroom', False, False, 'none', [2,3,5,6], '斜顶下睡眠与储物，中央高区用于儿童和朋友游戏。', '原文重复了一段床与天窗描述，不可计为两项独立用户证据。'),
 ('18_parttime_studio', False, False, 'mode_switch_unspecified', [2,3,5,6], '日常独居和孩子夏住两种容量；餐桌用于工作，物品隐藏收纳。', '未要求资料用餐时留在同一面，不能从工作餐桌复用推断占用持续。'),
 ('19_remote_studio', False, False, 'none', [2,3,5], '全职远程工作位置要与进餐位置在空间和朝向上区分，兼顾景观和偶尔客宿。', '空间独立是明确要求，不可为了最少家具把两者合并到同一位置。'),
 ('20_retired-couper_greatroom', False, True, 'no_hosting_reconfiguration', [2,3,4,5,8], '待客不应移动家具；餐桌兼备餐和八至十二人餐饮，烹饪时面向来客。', '不同用途不自动授权场景变换；不能把待客流程简化成桌椅是否无碰撞。'),
 ('21_single-designer_livingroom', False, False, 'furniture_reposition', [3,5,6,8], '阅读座位可随日照移位；陈列需要临时放置面；原有建筑设施不变。', '临时陈列放置与进行中项目跨时段保存不同，后者未明确。'),
 ('22_small_kids_room', False, False, 'none', [3,4,5,6,7], '高床下保留儿童游戏空间，另有独立阅读与学习位置，储物儿童可达。', '多种活动未说明同时发生；高床净空不能按最高表面支撑检查。'),
 ('23_stylist-w-pet_livingroom', False, False, 'none', [3,4,5,6], '两处不同用途的停留区、不同日照时段阅读和晚间电视，陈列兼分区。', '多窗采光和材质耐宠物条件不由对象布局单一分数覆盖。'),
 ('24_teacher_livingroom', False, True, 'none', [3,5,6,7], '阅读者不被通往书房的经过者打扰；楼梯和柱子限制路径。', '经过者不能被无根据地加成常住人口或独立座位需求。'),
 ('25_tech-couple_bedroom', False, True, 'none', [2,3,5,7], '夫妻各自床位和阅读灯，错时阅读睡眠，需给狗与两人穿衣留通行。', '两张床是用户明确指定，不能以节省家具为由合成一张。'),
 ('26_writer_nursary', False, True, 'none', [2,3,4,5,6,7], '婴儿睡眠与照护者安静工作、夜间喂养和单手换尿布，既有高柜保留。', '婴儿及照护者可并行活动，但原文未要求新增专用大办公桌。'),
 ('27_drag-queen_bedroom', False, False, 'storage_open_close', [2,3,4], '服装储藏需完整打开和遮住，梳妆支持出门前的工作流程。', '开合的空间需机构模型；不应把睡眠与化妆理解成两个使用者。'),
 ('28_producer_kitchen', False, True, 'none', [2,3,4], '成人烹饪时三名儿童在副岛台学习陪伴，待客动线与备餐互不阻断。', '每名儿童的并行容量需检查；不能只摆一张凳子后用岛台标签算完成。'),
 ('29_autism-kid_bedroom', False, False, 'none', [2,3,4,5,6], '攀爬和软着陆、睡眠及感官治疗；可复位储物与不同光照状态。', '治疗安全不是普通碰撞即可证实；不能据此给出医学适用性结论。'),
 ('30_youtuber_mezzanine', False, False, 'none', [2,3,4,5], '拍摄和办公可共享区域；练舞保持独立空地；沙发可供客人睡眠。', '共享不表示同时拍摄并做办公动作；不必擅自加第三套工作家具。'),
 ('short-1', False, False, 'mode_switch_unspecified', [1], '工作日办公、周末朋友观影及偶尔留宿，桌面双显示器是明确要求。', '没有指明材料持续、搬运权限或卧具转换方案，保留未知。'),
 ('short-2', False, False, 'none', [1], '面向轻度行动不便老人的卧室，给出房间尺寸。', '年龄、器具和清晰通行阈值均未给出，不虚构唯一量化答案。'),
 ('10_dancer_studio', False, False, 'furniture_reposition', [2,3,4,5,6], '晚归后固定落物点、睡前拉伸、移动一两件家具练舞，餐区视频授课无需每次重布。', '真实搬运次数、可移动资产和视频灯光均需额外执行证据。'),
]


def paragraphs(text):
    result = []
    for m in re.finditer(r'[^\n]+', text):
        t = m.group().strip()
        if t and not t.startswith(('#', 'https:')):
            result.append(dict(start=m.start(), end=m.end(), sha256=textsha(m.group())))
    return result


def native_box_bounds(obj):
    # The publisher's explicit convention: +Y up, facing0 north. This is only
    # a box proxy over the published JSON, not a mesh or physical simulation.
    x, y, z = obj['location']; width, height, depth = obj['dimensions']
    yaw = math.radians(obj.get('facing', 0))
    ex = (abs(math.cos(yaw)) * width + abs(math.sin(yaw)) * depth) / 2
    ez = (abs(math.sin(yaw)) * width + abs(math.cos(yaw)) * depth) / 2
    return [x-ex, y-height/2, z-ez], [x+ex, y+height/2, z+ez]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args(); root = a.root.resolve(); start = time.monotonic()
    assert root.parent == R / 'runs/autodl'
    raw = json.loads(SOURCE.read_text()); folders = raw['folders']
    source_hash = sha(SOURCE)
    assert source_hash == 'acc1fc6b8845c55f053251a21484e6ea75acef16c9f9dda74503331b179d2edf'
    assert len(REVIEW) == len(folders) == 33 and {r[0] for r in REVIEW} == set(folders)
    rows = []
    for key, persistent, concurrent, modes, indices, summary, limits in REVIEW:
        source = folders[key]; text = source.get('ad', ''); parts = paragraphs(text)
        for i in indices:
            assert 1 <= i <= len(parts), (key, i, len(parts))
        assert bool(text) == bool(indices)
        rows.append(dict(case=key, original_brief_available=bool(text), ad_field_present='ad' in source,
                         brief_words=len(text.split()), brief_sha256=textsha(text),
                         source_selector=['folders', key, 'ad'], source_file=str(SOURCE),
                         source_file_sha256=source_hash, evidence_paragraph_indices=indices,
                         evidence_spans=[parts[i-1] for i in indices],
                         persistent_in_progress_explicit=persistent,
                         concurrent_distinct_person_activities_explicit=concurrent,
                         requested_mode_change=modes, interpretation_summary=summary,
                         limitations=limits, full_original_brief_read=bool(text),
                         reviewer='Author self-review of published/adapted task narratives',
                         independent_professional_annotation=False, generated_failure_label=None,
                         inference_or_training_label_admitted=False))
    assert sum(r['original_brief_available'] for r in rows) == 32
    persistent_ids = {r['case'] for r in rows if r['persistent_in_progress_explicit']}
    assert persistent_ids == {'04_artist_atelier', '05_boy_bedroom', '09_couple_dining'}
    dump(root / 'complete_brief_review.json', dict(rows=rows,
         reviewed_entries=33, nonempty_briefs=32, missing_ad_aliases=['10_dancer-studio'],
         independent_annotators=0, independently_observed_model_failures=0,
         population_prevalence_claim=False, corpus_origin='Archived Function2Scene public viewer: author-adapted narratives, not resident conversation transcripts',
         raw_data_redistributed=False, viewer_training_license_established=False))
    with (root / 'brief_review.csv').open('w', newline='') as f:
        fields = ['case','original_brief_available','brief_words','persistent_in_progress_explicit',
                  'concurrent_distinct_person_activities_explicit','requested_mode_change',
                  'interpretation_summary','limitations','brief_sha256']
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); wr.writeheader()
        wr.writerows({k:r[k] for k in fields} for r in rows)
    # Targeted original/parsed comparison. The long full constraint documents
    # are NOT claimed fully read; offsets here refer to actually reviewed rules.
    selected = {
        '04_artist_atelier': [(13816,14403), (22647,23139)],
        '05_boy_bedroom': [(13536,13988), (21871,22423), (22780,23330), (25590,26096), (28995,29794)],
        '09_couple_dining': [(29539,29993), (31078,31572), (34103,34731), (36236,36960)],
    }
    captures = []
    for key, spans in selected.items():
        rules = folders[key]['constraints']
        final = folders[key]['files']['final.json']
        names = [o['name'] for o in final['furniture']]
        refs = []
        for lo, hi in spans:
            chunk = rules[lo:hi]; assert re.match(r'^[A-Z][0-9]\s+RULE:', chunk)
            refs.append(dict(start=lo, end=hi, sha256=textsha(chunk)))
        captures.append(dict(case=key, derived_constraints_sha256=textsha(rules),
             reviewed_rule_spans=refs, final_furniture_count=len(names), final_furniture_names=names,
             persistence_preserved_in_reviewed_derived_rules=True,
             static_final_state_sha256=textsha(json.dumps(final, ensure_ascii=False, sort_keys=True)),
             complete_constraints_read=False, activity_mode_trace_available_in_reviewed_fields=False,
             intermediate_iters_are_optimization_not_activity_modes=True,
             semantic_success_or_failure_of_complete_scene='Not established by this source inspection'))
    dump(root / 'original_to_derived_review.json', dict(rows=captures,
         all_three_have_explicit_persistence_rules=True,
         absence_of_persistence_understanding_in_published_pipeline=False,
         native_tool_execution_reproduced=False, no_generated_failure_rate_computed=True))
    effects = [
        dict(case='04_artist_atelier',
             activity_modes=['painting', 'ceramics_making'], passive_process='ceramics_drying',
             participants='One maker; no extra person inferred from the drying activity',
             resource_effects=[dict(role='in_progress_ceramic', effect='avoid_relocating_for_each_painting_session',
                                    interval='while_drying', geometry_footprint_m=None,
                                    source_paragraph=2, source_field='ad',
                                    locked_world_position=None)],
             proposed_bindings=[dict(role='drying_support', publisher_object='drying_surface',
                                     binding_origin='published generated final scene, not user-specified ID')]),
        dict(case='05_boy_bedroom',
             activity_modes=['model_making', 'homework', 'play'], passive_process='retained_build',
             participants='Child; a friend is optional for play, not an additional homework user',
             resource_effects=[dict(role='ongoing_build', effect='preserve_assembly_between_sessions',
                                    source_paragraph=4, source_field='ad',
                                    locked_world_position=None,
                                    relocation_permission='Assembly clause alone does not prohibit whole-object movement')],
             proposed_bindings=[dict(role='ongoing_build', publisher_object='in_progress_build',
                                     binding_origin='published generated final scene'),
                                dict(role='project_storage', publisher_object='project_shelf',
                                     binding_origin='published generated choice, not unique user instruction')]),
        dict(case='09_couple_dining',
             activity_modes=['work_or_ceramics', 'ordinary_meal', 'hosting'], passive_process='work_in_progress',
             participants='Couple; number of visiting friends not explicitly quantified',
             resource_effects=[dict(role='ongoing_projects', effect='preserve_progress_without_full_packing',
                                    source_paragraphs=[3,4], source_field='ad',
                                    locked_world_position=None),
                               dict(role='table_items', effect='allow_temporary_nearby_relocation_for_hosting',
                                    source_paragraph=4, source_field='ad',
                                    geometry_footprint_m=None, required_temporary_capacity=None)],
             proposed_bindings=[dict(role='main_shared_surface', publisher_object='dining_table',
                                     binding_origin='published generated final scene'),
                                dict(role='temporary_destination', publisher_object=None,
                                     binding_origin='Must be constructed/checked; a named shelf alone is not a capacity certificate')]),
    ]
    for effect in effects:
        source = folders[effect['case']]; effect.update(
            original_brief_sha256=textsha(source['ad']),
            source_selector=['folders',effect['case'],'ad'],
            annotation_origin='Author self-review of explicit source clauses',
            independently_annotated=False, complete_numeric_contract=False,
            native_action_trace=None, trained=False,
            unknowns=['No activity-mode transition trace', 'Not all loads have sourced dimensions',
                      'Source final box labels are not mesh-verified semantic IDs'])
    dump(root / 'grounded_effect_contracts.json', dict(records=effects,
         role='Partial source-grounded design constraints, not actual model outputs or gold native actions',
         admitted_for_native_training=False, no_user_preference_labels_invented=True))
    boy = folders['05_boy_bedroom']['files']['final.json']
    furniture = {r['name']: r for r in boy['furniture']}
    project = furniture['in_progress_build']; shelf = furniture['project_shelf']
    lo, hi = native_box_bounds(project); slo, shi = native_box_bounds(shelf)
    bottom_gap = lo[1] - shi[1]
    inside = all(lo[i] >= slo[i]-1e-12 and hi[i] <= shi[i]+1e-12 for i in [0,2])
    assert inside and abs(bottom_gap) < 1e-12
    dump(root / 'published_positive_binding.json', dict(case='05_boy_bedroom',
         project=project, shelf=shelf, project_bounds=[lo,hi], shelf_bounds=[slo,shi],
         horizontal_box_containment=inside, bottom_to_top_gap_m=bottom_gap,
         status='Static published box proxy has project resting on the project shelf',
         native_axis='+Y up; no conversion to another pipeline was performed',
         semantic_identity_by_source_names_only=True, bound_to_mesh_assets=False,
         assembly_conservation_across_activities='unknown', transition_path='unknown',
         complete_room_valid=None, positive_case_not_omitted=True))
    fetch = json.loads(Path('/dev/shm/iarcs_source_20261111/fetch_manifest.json').read_text())
    for row in fetch['files']:
        assert sha(Path('/dev/shm/iarcs_source_20261111') / row['path']) == row['sha256']
    dump(root / 'external_source_receipt.json', fetch)
    reflection = (Path('/dev/shm/iarcs_source_20261111')/'iarcs/reflection.py').read_text()
    tree = ast.parse(reflection)
    validator = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_validate_reward_source')
    assert 'task_prompt' not in {n.id for n in ast.walk(validator) if isinstance(n,ast.Name)}
    segments=[]
    for label in ['_validate_reward_source','_dummy_scene']:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==label)
        segments.append(dict(symbol=label,line_start=node.lineno,line_end=node.end_lineno,
                             ast_sha256=textsha(ast.dump(node,include_attributes=False))))
    dump(root/'external_branch_review.json',dict(revision=fetch['revision'],
         selected_functions=segments, external_source_executed=False,
         findings=[
             'Pinned reward validator compiles/calls a proposed function on synthetic input and checks output shape and finiteness; this function does not inspect task_prompt.',
             'This is evidence about the selected validator only, not proof that every part of the authors system lacks semantic checking.',
             'Reflection source already records reward-code history and training statistics; logging and code feedback are not new mechanisms.',
             'Training source uses clipped importance-ratio policy gradient over saved denoising trajectories; no training was run here.',
             'Native coordinate conventions differ: this reward API uses XZ floor and HALF extents; public Function2Scene uses full dimensions and +Y-up. No data was passed between them.'
         ], novel_learning_method_established=False))
    counts = dict(entries=33, complete_nonempty_briefs_read=32, missing_briefs=1,
                  explicit_in_progress_persistence=sum(r['persistent_in_progress_explicit'] for r in rows),
                  explicit_distinct_person_concurrency=sum(r['concurrent_distinct_person_activities_explicit'] for r in rows),
                  subset_with_targeted_parsed_rule_review=3,
                  parsed_persistence_present=3, static_positive_box_binding=1,
                  new_inference_calls=0, paid_API_calls=0, training_updates=0, new_whole_scenes=0,
                  audit_wall_seconds=time.monotonic()-start,
                  full_research_and_source_reading_seconds='Not measured by this script',
                  no_statistical_population_or_generator_failure_rate=True)
    dump(root / 'summary.json', counts)
    protected = json.loads((R / 'runs/autodl/20261111_feedback_information_control/source_before_after.json').read_text())
    assert len(protected['original_inputs']) == 579
    for row in protected['original_inputs']:
        assert sha(row['path']) == row['before'], row['path']
    dump(root / 'protected_sources.json', dict(count=579, all_rehashed_unchanged=True,
         original_ledger='runs/autodl/20261111_feedback_information_control/source_before_after.json',
         original_ledger_sha256=sha(R / 'runs/autodl/20261111_feedback_information_control/source_before_after.json'),
         publisher_source_unchanged=sha(SOURCE)==source_hash))
    required = ['reading_ledger.json','algorithm_scope.txt','stage_report.html','reproduce.txt']
    for name in required:
        assert (root / name).is_file(), name
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in
                   {'evidence.zip','delivery_manifest.json','delivery_verification.json'})
    files.append(Path(__file__).resolve())
    manifest = [dict(path=str(p.relative_to(R)), sha256=sha(p), bytes=p.stat().st_size) for p in files]
    dump(root / 'delivery_manifest.json', dict(files=manifest, no_viewer_raw_data_or_external_code_redistributed=True))
    with zipfile.ZipFile(root / 'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'delivery_manifest.json']:
            z.write(p, str(p.relative_to(R)))
    with zipfile.ZipFile(root / 'evidence.zip') as z:
        assert z.testzip() is None
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    dump(root / 'delivery_verification.json', dict(status='verified', payload_files=len(manifest),
         evidence_zip_sha256=sha(root/'evidence.zip'), zip_crc_and_hashes_ok=True,
         source_count=579, source_and_original_corpus_unchanged=True))
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
