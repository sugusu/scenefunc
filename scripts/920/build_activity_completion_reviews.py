"""Single-state objective reviews from a frozen executed activity sequence.

No new geometric simulation, expert preference, action target or model run.
This retains measurement-defined seat assignments, not all possible layouts.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def aggregate(checks):
    """Unknown is never silently promoted to a successful requirement."""
    if not checks:
        return 'unknown'
    values = [x['pass'] for x in checks]
    require(all(x is None or type(x) is bool for x in values), 'Non-boolean predicate')
    return 'fail' if False in values else 'unknown' if None in values else 'pass'


def review(mode, state, initial, measured, geometry, program):
    rows = []
    tables = program['table_ids']
    require(mode in ('separate_work', 'joined_four_place_hosting'), 'Unknown activity')
    assigned = {x['chair_id'] for x in program['seat_assignments']}
    require(set(measured['seat_alignment']) == assigned, 'Missing or extra seat measurements')
    widths = program['access_settings']['square_body_widths_m']
    fixed = set(program['access_settings']['approach_points_xy_m'])
    for scenarios, targets in ((measured['fixed_access']['scenarios'], fixed),
                               (measured['seat_approaches'], assigned)):
        require(len(scenarios) == len(widths) and
                {x['square_body_width_m'] for x in scenarios} == set(widths),
                'Incomplete body-width coverage')
        require(all(set(x['targets']) == targets for x in scenarios), 'Incomplete target coverage')
    needed = program['work_chair_ids'] if mode == 'separate_work' else sorted(assigned)
    def add(name, value, ids, evidence):
        rows.append(dict(requirement=name, pass_=value, object_ids=ids, evidence=evidence))
        rows[-1]['pass'] = rows[-1].pop('pass_')
    ids = set(initial['objects'])
    movable = {'recon_table_b_0', 'recon_chair_b_0'}
    add('all_original_instances_and_assets', set(state['objects']) == ids and all(
        {k:v for k,v in state['objects'][oid].items() if k != 'transform'} ==
        {k:v for k,v in initial['objects'][oid].items() if k != 'transform'} for oid in ids & set(state['objects'])),
        sorted(ids), 'native_state.objects compared to recorded initial state')
    add('fixed_objects_unchanged', all(state['objects'].get(oid) == initial['objects'][oid] for oid in ids-movable),
        sorted(ids-movable), 'native_state.objects, excluding two permitted movable instances')
    add('no_recorded_envelope_overlap_or_boundary_violation',
        not geometry['positive_3d_AABB_pairs'] and not geometry['outside_declared_wall_inner_faces'],
        sorted(ids), 'recorded summary.actual_geometry; conservative AABB proxy')
    for scenario in measured['fixed_access']['scenarios']:
        for name, target in scenario['targets'].items():
            add(f'access_{name}_width_{scenario["square_body_width_m"]}', target['reachable_from_entry'], [],
                f'measured.fixed_access.scenarios width={scenario["square_body_width_m"]} target={name}')
    if mode == 'separate_work':
        add('separate_table_gap', measured['table_geometry']['footprint_gap_m'] >= program['separate_min_gap_m'],
            tables, 'measured.table_geometry.footprint_gap_m >= program.separate_min_gap_m')
    else:
        add('joined_table_geometry', measured['table_geometry']['joined_under_declared_threshold'], tables,
            'measured.table_geometry.joined_under_declared_threshold (gap, alignment, top-height)')
        add('nonoverlapping_declared_place_rectangles', not measured['place_rectangles_overlap'], needed,
            'measured.place_rectangles_overlap')
    for oid in needed:
        add('assigned_seat_alignment_'+oid, measured['seat_alignment'][oid]['aligned_under_declared_proxy'],
            [oid, measured['seat_alignment'][oid]['table_id']], 'measured.seat_alignment.'+oid)
        for scenario in measured['seat_approaches']:
            add(f'seat_access_{oid}_width_{scenario["square_body_width_m"]}',
                scenario['targets'][oid]['reachable_from_entry'], [oid],
                f'measured.seat_approaches width={scenario["square_body_width_m"]} target={oid}')
    status = aggregate(rows)
    return dict(requested_mode=mode, measured_requirements_status=status,
        further_edit_under_measured_scope='no_measured_need' if status == 'pass' else 'verified_deficit' if status == 'fail' else 'unresolved',
        failed_requirements=[x for x in rows if x['pass'] is False],
        passed_requirements=[x['requirement'] for x in rows if x['pass'] is True],
        whole_room_success='unknown', professional_preference='not_assessed',
        stop_action_training_admitted=False, checks=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); root = a.repo.resolve(); out = a.out.resolve()
    require(not out.exists(), 'Use fresh output directory')
    refs = {}
    def read(path):
        path = Path(path)
        if not path.is_absolute(): path = root / path
        refs[str(path)] = sha(path)
        return json.loads(path.read_text())
    protocol = read(a.protocol)
    for p,h in protocol['input_hashes'].items(): require(sha(p) == h, 'Frozen input changed: '+p)
    folder = Path(protocol['source_directory']); source = read(folder/'SOURCE_RECORD.json')
    summary = read(folder/'SUMMARY.json'); program = read(folder/'FROZEN_PROGRAM.json')
    print(datetime.now(timezone.utc).isoformat(), 'build10singleStateActivityReviewsCPU; no model', flush=True)
    initial = read(folder/'initial_state.json'); records = []
    qa_source = Path(protocol['official_source']) / 'src'
    sys.path.insert(0, str(qa_source))
    spec = importlib.util.spec_from_file_location('completion_official', qa_source/'sft_dataset.py')
    official = importlib.util.module_from_spec(spec);sys.modules[spec.name]=official;spec.loader.exec_module(official)
    for summary_row in summary['rows']:
        name = summary_row['state']
        state = read(folder/f'{name}_state.json'); measured = read(folder/f'{name}_measurement.json')
        archive_state = root.parent/summary_row['state_ref']['path']
        binding_path = root.parent/summary_row['binding_ref']['path']
        require(sha(archive_state) == summary_row['state_ref']['sha256'], 'Archived state mismatch')
        require(sha(binding_path) == summary_row['binding_ref']['sha256'], 'Binding mismatch')
        require(read(archive_state) == state, 'Git and runtime state differ')
        binding = read(binding_path)
        require(binding['native_state'] == state and binding['state_sha256'] == canonical(state), 'Image state mismatch')
        images = binding['images']; require(len(images) == 5, 'Need five views')
        for im in images:
            require(sha(im['path']) == im['sha256'], 'Image changed');refs[im['path']] = im['sha256']
        require(summary_row['mode_geometry'] == measured['mode_geometry'], 'Summary and measurement mismatch')
        for mode in ('separate_work','joined_four_place_hosting'):
            answer = review(mode,state,initial,measured,summary_row['actual_geometry'],program)
            # The independent conjunction must agree with the recorded endpoint
            # label on this bounded, verified sequence, including both intermediates.
            require((answer['measured_requirements_status'] == 'pass') == measured['mode_geometry'][mode], 'Predicate/recorded outcome disagreement')
            unknowns = [*summary['unknowns'], 'Fixed named seat assignments are checked; this is not a search over every feasible arrangement.']
            answer['unknowns'] = unknowns
            observation = dict(current_state=state, requested_mode=mode,
                source_principle='两张可分可合的桌子允许根据活动重新组织空间；不意味着每次都应移动。前半句为项目原则转述，后半句为研究者解释。',
                source_url='https://www.nevertoosmall.com/post/waterloo-street-singapore',
                requirements=dict(retain_all_original_instances=True,movable_ids=sorted({'recon_table_b_0','recon_chair_b_0'}),
                    fixed_access_points=program['access_settings']['approach_points_xy_m'],
                    separate_min_gap_m=program['separate_min_gap_m'],joined_max_gap_m=program['joined_max_gap_m'],
                    work_chair_ids=program['work_chair_ids'],seat_assignments=program['seat_assignments']),
                measurement=copy.deepcopy(measured),unknowns=unknowns)
            # Hide the high-level mode verdict. Raw pre-action tool measurements,
            # including individual alignment checks, are deliberately available.
            observation['measurement'].pop('mode_geometry')
            observation['recorded_geometry_check'] = summary_row['actual_geometry']
            observation['identity_comparison'] = dict(
                provenance='Computed by this data builder from current and initial native states; not a historical tool response.',
                checks=answer['checks'][:2])
            prompt = ('按当前活动评审现状，列出具体对象、已满足与未满足的声明条件及未知项。仅判断当前测量范围内是否仍有修改依据；不得宣称真实整屋完成、最优布局或唯一修改。提供的量测是已取得的工具证据。\n'+
                      json.dumps(observation,ensure_ascii=False,separators=(',',':')))
            identifier = name+'__'+mode
            doc = dict(stage='qa', system='Assess the current requested activity from supplied measurements and state; do not invent expert approval.',
                provenance=dict(family_id=source['family_id'],kind='researcher_derived_objective_single_state_review',
                    training_eligible=False,training_authorized=False,new_independent_project=False,
                    original_source_statement_distinct_from_derived_review=True),
                turns=[dict(role='user',content=prompt,image=[x['path'] for x in images]),
                       dict(role='assistant',content='',answer=json.dumps(answer,ensure_ascii=False,separators=(',',':')))])
            path=out/'episodes'/identifier/'episode.json';write(path,doc)
            samples = official.build_step_samples(path,root)
            require(len(samples)==1 and samples[0].qa_answer==doc['turns'][1]['answer'], 'Official QA target mismatch')
            require(samples[0].qa_images==[x['path'] for x in images], 'Official QA views changed')
            records.append(dict(id=identifier,state=name,requested_mode=mode,review=answer,
                document=dict(path=str(path),sha256=sha(path)),state_sha256=canonical(state),
                measurement_path=str(folder/f'{name}_measurement.json')))
    require(len(records)==10,'Unexpected case count')
    # Semantic guard: missing coverage cannot become a completion label; a
    # measured violation remains a violation alongside other unknown fields.
    require(aggregate([])=='unknown' and aggregate([{'pass':None}])=='unknown' and
            aggregate([{'pass':True},{'pass':None}])=='unknown' and
            aggregate([{'pass':False},{'pass':None}])=='fail','Three-valued aggregation failed')
    result=dict(status='completed_objective_review_candidates',recorded_at=datetime.now(timezone.utc).isoformat(),records=records,
        counts=dict(records=10,unique_states=5,source_families=1,
            measured_no_edit_admissible=sum(x['review']['measured_requirements_status']=='pass' for x in records),
            measured_deficit=sum(x['review']['measured_requirements_status']=='fail' for x in records),
            whole_room_success_labels=0,new_sources=0,new_actions=0),
        input_hashes=refs,script_sha256=sha(__file__),official_samples_verified=10,processor_encoding_run=False,
        model_calls=0,training_updates=0,scene_edits=0,
        limits=['Five states from one previously exposed researcher trajectory, not ten independent projects.',
                'Measured no-edit admissibility does not authorize whole-room completion or stop-action supervision.',
                'No future state, reference action, candidate-ranking slot or endpoint coordinate enters a request.',
                'Measured current checks are deliberately visible; this tests evidence interpretation, not unaided visual geometry.'])
    write(out/'RESULT.json',result)
    lines=['# 当前活动是否还需要修改：单状态客观评审候选','',
           '| 已执行状态 | 两人独立工作 | 四人共享待客 |','|---|---|---|']
    for name in [r['state'] for r in summary['rows']]:
        vals=[r['review']['measured_requirements_status'] for r in records if r['state']==name]
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','pass 仅表示冻结的几何、身份、接近与固定坐席分配全部通过；fail 会在 RESULT.json 指出具体条件、对象和证据。所有记录的真实整屋成功、舒适与专业偏好仍未知。',
            '', '十条是同一来源五个状态与两种活动的研究者派生评审，不增加独立项目数。不提供后续动作或参考终态；输入中主动提供当前工具量测，但去除总模式判定。原三状态比较QA保留，本包增加中间步和单状态条件化判断。',
            '', '十份QA均由固定官方样本构建器实际读取，答案及五图顺序精确核对；没有再次运行处理器或模型，也没有训练。当前不把这些标签改写为终止动作示范。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':
    main()
