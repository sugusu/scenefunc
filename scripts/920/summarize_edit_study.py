"""Summarize the frozen study without equating initial success with improvement.

Read-only with respect to runs, metrics, models and scenes. Missing/failed cells
are explicit; complete-case contrasts are descriptive developer comparisons.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_audits(initial, final, role_objects=None):
    def contacts(a):
        return {tuple(sorted(r['objects'])) for r in a['positive_aabb_pairs']
                if r['triangle_surface_intersection_detected']}
    def outside(a):
        return {r['object_id'] for r in a['bounds'] if r['outside_nominal_room_xy']}
    old_contacts,new_contacts=contacts(initial),contacts(final)
    missing=[oid for oid in initial['activity']['retained_ids']
             if not final['activity']['retained_ids'].get(oid,False)]
    altered=[oid for oid in initial['activity']['identity_asset_dimensions_preserved']
             if not final['activity']['identity_asset_dimensions_preserved'].get(oid,False)]
    changed_immutable=[oid for oid in initial['immutable_records_unchanged']
                       if not final['immutable_records_unchanged'].get(oid,False)]
    common=dict(missing_original_instances=sorted(missing),altered_identity_assets_dimensions=sorted(altered),
                changed_immutable_records=sorted(changed_immutable),
                new_surface_intersections=sorted(new_contacts-old_contacts),
                remaining_initial_surface_intersections=sorted(new_contacts&old_contacts),
                resolved_initial_surface_intersections=sorted(old_contacts-new_contacts),
                new_outside_objects=sorted(outside(final)-outside(initial)),
                remaining_initial_outside_objects=sorted(outside(final)&outside(initial)))
    base_by_width={s['width_m']:s for s in initial['activity']['scenarios']}
    final_by_width={s['width_m']:s for s in final['activity']['scenarios']}
    if set(base_by_width)!=set(final_by_width):raise ValueError('Width scenarios changed')
    scenarios=[]
    for width,before in base_by_width.items():
        after=final_by_width[width]
        # Every original role stays in the denominator, including deleted IDs.
        old_roles=before['role_access'];new_roles=after['role_access']
        lost=[name for name,v in old_roles.items() if v and not new_roles.get(name,False)]
        initial_failed=[name for name,v in old_roles.items() if not v]
        remaining_failed=[name for name in initial_failed if not new_roles.get(name,False)]
        initial_pass=before['activity_proxy_pass'];final_pass=after['activity_proxy_pass']
        if type(initial_pass) is not bool or type(final_pass) is not bool:
            raise ValueError('Unknown activity must not be cast into a success/failure boolean')
        geometry_identity_failure=bool(missing or altered or changed_immutable or common['new_surface_intersections'] or common['new_outside_objects'])
        access_scenario=next(s for s in final['activity']['role_and_seat_access']['scenarios'] if s['square_body_width_m']==width)
        lost_context=[]
        for name in lost:
            oid=(role_objects or {}).get(name)
            alternatives=[label for label,measurement in access_scenario['targets'].items()
                          if oid is not None and label.startswith(oid+':') and measurement['reachable_from_entry']]
            lost_context.append(dict(role_probe=name,object_id=oid,other_reachable_seat_probes=alternatives,
                                     seat_square_and_any_approach_pass=oid in after['valid_seat_ids']))
        # Preserve the strict probe outcome. Do not call a failed particular
        # side-point a proven loss of human use when another approach exists.
        proxy_failure=geometry_identity_failure or bool(lost)
        scenarios.append(dict(width_m=width,initial_activity_proxy_pass=initial_pass,final_activity_proxy_pass=final_pass,
            activity_transition=('preserved' if final_pass else 'lost') if initial_pass else ('gained' if final_pass else 'still_missing'),
            original_role_denominator=len(old_roles),initial_failed_roles=initial_failed,
            remaining_initial_failed_roles=remaining_failed,newly_lost_roles=lost,
            lost_role_probe_context=lost_context,new_geometry_or_identity_failure=geometry_identity_failure,
            new_declared_proxy_failure=proxy_failure,proxy_satisfied_without_new_declared_failure=final_pass and not proxy_failure,
            human_activity_loss_from_failed_probe_alone='not_established',
            final_original_roles_reachable=sum(bool(new_roles.get(name,False)) for name in old_roles)))
    return dict(**common,scenarios=scenarios,
                initial_furniture=len(initial['inventory']),final_furniture=len(final['inventory']),
                initial_largest_empty_rectangle_m2=initial['free_floor']['largest_grid_empty_rectangle_m2'],
                final_largest_empty_rectangle_m2=final['free_floor']['largest_grid_empty_rectangle_m2'],
                real_human_use='unknown',professional_whole_room_quality='not_assessed')


def training_summary(root):
    summaries={}
    for arm in ('action_only','decision_action'):
        folder=root/'training'/arm
        if not (folder/'RUN.json').exists():
            summaries[arm]=dict(status='not_started');continue
        record=read(folder/'RUN.json');groups=defaultdict(list)
        if (folder/'MICRO_LOSS.json').exists():
            for row in read(folder/'MICRO_LOSS.json'):
                groups[row['optimizer_step_before']].append(row)
        complete=[]
        for step,rows in sorted(groups.items()):
            if len(rows)!=22:continue
            complete.append(dict(update=step+1,
                **{component:sum(row[component]['nll_sum']*row[component]['weight']/22 for row in rows)
                   for component in ('action','decision')}))
        summaries[arm]=dict(status=record['status'],optimizer_updates=record['optimizer_updates'],
            started_at=record['started_at'],ended_at=record.get('ended_at'),
            macro_batch_losses=complete,peak_allocated_bytes=record.get('peak_allocated_bytes'),
            saved_reloaded_exact=record.get('same_process_adapter_save_zero_reload_exact'),
            loss_scope='Online teacher-forced training objectives. Joint actions are conditioned on gold training D; cross-arm action loss is not a common-input comparison.')
    return summaries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();p=a.study.resolve();a.out.mkdir(parents=True,exist_ok=False)
    root=Path(read(p/'TRAIN_PLAN.json')['runtime_directory'])
    matrix=read(p/'EVALUATION_COMMAND_MATRIX.json')['cells'];rows=[]
    for cell in matrix:
        folder=Path(cell['runtime'])/'rollout';file=folder/'RESULT.json'
        row={k:cell[k] for k in ('cell','task','arm','seed')}
        if not file.exists():
            row.update(status='not_started',comparison=None);rows.append(row);continue
        record=read(file)
        row.update(status=record['status'],stop_reason=record.get('stop_reason'),result_path=str(file),result_sha256=sha(file),
            comparison=None,model_calls=record['model_calls'],tool_attempts=record['tool_attempts'],native_calls=record['native_calls'],
            changed_state_batches=sum(r.get('before_sha256')!=r.get('after_sha256') for r in record['rounds'] if 'after_sha256' in r),
            schema_or_output_rejections=sum('rejection' in r for r in record['rounds']),
            generated_tokens=record['generated_tokens'],gpu_subprocess_seconds=record['gpu_subprocess_seconds'],
            error=record.get('error'),finalization_error=record.get('finalization_error'))
        audit=folder/'final_audit.json'
        if audit.exists():
            initial=root/'evaluation'/cell['task']/'prepare/initial_audit.json'
            plan=read(Path(cell['plan_template']))
            roles=read(plan['evaluation']['plan'])['activity']['roles']
            row['comparison']=compare_audits(read(initial),read(audit),{k:v['object_id'] for k,v in roles.items()})
            row['initial_audit_sha256']=sha(initial);row['final_audit_sha256']=sha(audit)
        rows.append(row)
    paired=[]
    for task in ('Pine','Milan'):
        for seed in (17,29):
            group={r['arm']:r for r in rows if r['task']==task and r['seed']==seed}
            if len(group)!=3 or any(r['comparison'] is None or r['status']!='policy_completed_pending_comparison' for r in group.values()):continue
            outcomes={arm:next(s for s in row['comparison']['scenarios'] if s['width_m']==.5) for arm,row in group.items()}
            paired.append(dict(task=task,seed=seed,outcomes=outcomes))
    result=dict(status='descriptive_snapshot_not_final_review',recorded_at=datetime.now(timezone.utc).isoformat(),
        run_id=p.name,planned_cells=12,terminal_cells=sum(r['status'] in ('failed','policy_completed_pending_comparison') for r in rows),
        cells_with_final_audit=sum(r['comparison'] is not None for r in rows),development_project_count=2,
        rows=rows,complete_matched_three_arm_blocks=paired,training=training_summary(root),
        limits=['Two exposed development projects; two seeds do not make four independent projects.',
                'Milan initial work-material/dining proxies already pass: preserved success is not a learned improvement.',
                'Pine reference is a declared reconstruction, not a professional whole-room gold label.',
                'Static proxies and finite search do not establish real use, comfort, aesthetics or universal infeasibility.',
                'No result in this report alone establishes pairing mechanism or cross-project generalization.'])
    (a.out/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 固定开发试验：初态—终态对照','',
        '每一行是一份预定终态；未运行或未测量保留为空，不能算成功或失败。代理通过不等于整屋人类使用通过。',
        '', '| 项目 | 种子 | 方法 | 状态 | 用途代理（初→末，0.5m） | 新增几何/身份违例 | 新失效接近探针 | 原角色探针可达 | 编辑状态变化次数 | GPU子进程秒 |',
        '|---|---:|---|---|---|---|---|---|---:|---:|']
    for r in rows:
        scenario=next((s for s in r['comparison']['scenarios'] if s['width_m']==.5),None) if r['comparison'] else None
        outcome=f"{int(scenario['initial_activity_proxy_pass'])}→{int(scenario['final_activity_proxy_pass'])} ({scenario['activity_transition']})" if scenario else '未测'
        hard=str(scenario['new_geometry_or_identity_failure']) if scenario else '未测'
        lost=','.join(scenario['newly_lost_roles']) or '无' if scenario else '未测'
        access=f"{scenario['final_original_roles_reachable']}/{scenario['original_role_denominator']}" if scenario else '未测'
        lines.append(f"| {r['task']} | {r['seed']} | {r['arm']} | {r['status']} | {outcome} | {hard} | {lost} | {access} | {r.get('changed_state_batches','—')} | {round(r['gpu_subprocess_seconds'],2) if 'gpu_subprocess_seconds' in r else '—'} |")
    lines+=['','Milan 初态已容纳分开的工作材料与两餐位，其终态通过属于保持；Pine 初态缺第二工作位。0.4/0.6m敏感性、初态原有失败、新增相交/越界、身份与原角色损失分别保存在 SUMMARY.json。','',
            '预设某侧接近点失效与对象完全不可使用分开报告；其他可达侧及座位代理在JSON中保留。严格探针失败不被删除，也不单独证明人类用途受损。', '',
            '本表尚未经过最终场景目视和证据审阅，不自动输出“论文方法成立”。']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('terminal_cells','cells_with_final_audit','planned_cells')}))


if __name__=='__main__':main()
