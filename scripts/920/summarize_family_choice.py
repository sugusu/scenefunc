"""Publish bounded interpretations and update live source maturity without training."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from native_project import sha256


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',required=True);ap.add_argument('--current',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];now=datetime.now(timezone.utc).isoformat()
    for run in [a.previous,a.current]:
        p=repo/'runs/920'/run;t=repo.parent/'runtime'/run;archive=read(p/'ARCHIVE_CHECKS.json');summaries=archive['summaries']
        corrected=run==a.current;visual=read(p/'VISUAL_REVIEW.json')
        last_archive=t/('archive_stage' if corrected else 'archive_v2_stage')
        for name in ['receipt.json','stage.log']:
            target=p/'execution'/last_archive.name/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(last_archive/name,target)
        archive['currently_excluded_own_receipt']=[]
        archive['completed_archive_receipt']=read(last_archive/'receipt.json')
        write(p/'ARCHIVE_CHECKS.json',archive)
        summary=dict(run_id=run,status='conditional_choice_record_completed' if corrected else 'completed_visual_input_rejected',
            updated_at=now,source_families=1,condition_variants=2,actual_candidate_episodes=2,native_tool_calls=2,
            views=20,inspected_final_views=10 if corrected else 4,conditions_are_not_independent_projects=True,
            model_calls=0,training_updates=0,training_eligible=False,professional_labels=0,
            choices={label:dict(clear_rays=x['measured']['clear_rays'],total_rays=15,
                declared_access_all_widths=x['measured']['all_declared_access'],
                ids_preserved=x['measured']['all_ids_preserved'],dimensions_preserved=x['measured']['all_dimensions_preserved'],
                triangle_contact_pairs=x['triangle_contact_pairs'],outside_room=x['outside_room'],
                generic_failed_approaches=x['generic_failed_approaches'],
                native_seconds=x['native_receipt']['elapsed_seconds'],native_exit_code=x['native_receipt']['exit_code'],
                cold_restore_seconds=x['restore_receipt']['elapsed_seconds']) for label,x in summaries.items()},
            visual_review_status=visual['status'],failure_reason=None if corrected else 'Sofa and perimeter storage orientations disagree with intended access faces.',
            failed_postprocessing=archive['failed_postprocessing'],
            next_step='Use this as a source-linked conditional development record; freeze model input without candidate coordinates before any model comparison. New training needs separate approval.')
        write(p/'SUMMARY.json',summary)
        rows='\n'.join(f"| {label} | {x['clear_rays']}/15 | 9×3通过 | {x['triangle_contact_pairs']} | {x['outside_room']} | {x['native_seconds']:.3f}秒/{x['native_exit_code']} |" for label,x in summary['choices'].items())
        state_note=('朝向修订版本。两个终态各五图已实际查看，沙发朝向共同起居区域，柜体声明正面与接近侧一致。'
                    if corrected else '本版本两次执行均退出0，但视觉检查发现沙发朝窗墙，靠墙柜体正面与声明取放侧不一致，因此拒绝作为正向示范。原始结果完整保留；修订见 `'+a.current+'`。')
        text=f'''# F-House来源经验的条件化重建记录

{state_note} 本批0模型调用、0训练更新，不是专业学习或泛化收益。

## 原始依据与研究构造

[F-House项目报道](https://www.nevertoosmall.com/post/f-house-osaka)明确把相对开放的组织与容纳既有家具、优先共同生活联系起来。这里重用已登记的这一条知识，不增加知识条数。原项目为三层住宅；本研究只构造7×4.8×2.7米的共同起居/用餐房间。房壳、八件已有家具、具体位置、观察点及活动条件均为研究假设，非原始测绘或历史家具清单。卧室、厨房与卫浴在本选择区域之外，不宣称完整住宅复现。

两种任务卡分别提出“餐桌处希望看见沙发上的家人”和“希望减少两处的直接视线干扰”，均要求保留家具、四个餐位、起居、收纳和通行，没有指定移动哪个物体、摆到哪个坐标。第一种是源共同生活偏好的研究操作化；第二种是研究者构造的对比条件，不是F-House住户要求或建筑师否决过的方案。

同一初态分别执行两个研究者候选：高柜靠墙或置于两个活动区之间。每个候选只有一次真实move，其余七件家具与房壳不变；八件实例ID和尺寸均保留。候选位置不在任务卡中，存于研究动作程序。两种布局不穷尽所有可行方案，当前还没有模型自主选择。

## 实际后果

| 候选 | 畅通视线 | 声明接近点×身体宽度 | 三角接触对 | 越界 | 原生耗时/退出 |
| --- | ---: | --- | ---: | ---: | --- |
{rows}

观察点和5个沙发目标在3个声明坐姿眼高下产生15条线段。柜体为不透明实心代理，程序测线段与家具包围盒相交；0/15不等于真实隐私或隔音，15/15不等于舒适交流。通行用0.4/0.5/0.6米方形身体与2厘米网格，不是人体工学或规范合格证明。

在这两个指定条件和候选中，按事前固定的指标，“相互看见”偏向靠墙方案，“减少视线干扰”偏向分隔方案。这种条件相关的评价不同于把开放程度越大固定标成越好；它也不是独立专家偏好。高柜的实体性质、位置与活动关系共同决定后果，不能把公开原则当作任意空间的统一坐标答案。

通用局部正面探针失败仍逐项保存在各 `FINAL_AUDIT.json` 和 `SUMMARY.json`。修订版两组均为0.4米下2个失败、0.5/0.6米下3个失败，涉及两张南侧凳及餐桌默认正面；声明的凳后接近点可达。这种差异不能用于删掉不利指标，也不能凭点可达宣称真实坐席可用。柜门、容量、人体取用、搬运轨迹、家具稳定性与整体偏好未知。

## 数据交付及限制

`DECISION_PACKET.json`把来源、条件、研究初态、真实对象/动作、终态和后果连接起来。`REVIEW_CANDIDATES.json`提供两个条件下的程序评价候选，标明非专家标签，评价只放evaluation_only；未接训练加载器、未新增训练准入。只有开放共同生活原则有本项目来源，反向需求及比较规则属于研究构造，不能强行写成另一条专业理由。

最终五图、状态绑定和小日志已归档，全部20张原图逐张核哈希/解码；原生检查点和GLB在服务器，路径/版本/哈希见 `RECOVERY.md`。两候选各已在新进程冷恢复，资产绑定、身份与一条动作回执一致，未做恢复后编辑。通用审计脚本本次仅修正了“同资产多个实例共用名称”的展示问题，旧脚本和历史结果不改。

v1错误与v2朝向修订分别保留；两版本不是独立项目或效果重复验证。v1归档首次把正在运行的自身回执当成未完成实验而退出1；修复只涉及归档生命周期，并保留失败代码/日志，没有因此重跑原生动作。

下一步可在冻结底座上检验需求改变是否改变实际编辑，使用相同房间、工具和有限调用预算，输入不含候选坐标或评价答案。之后须扩展独立来源与空间分布，并与同知识提示/反馈比较；本条记录不够支持论文主张或直接扩大训练。
'''
        (p/'SUMMARY.md').write_text(text,encoding='utf-8')
        status=read(p/'STATUS.json');status.update(status=summary['status'],completed=2,planned=2,updated_at=now,
            started_at=summaries['perimeter']['native_receipt']['started_at'],
            ended_at=summaries['divider']['native_receipt']['ended_at'],actual_native_exit_codes=[0,0],
            first_log=summaries['perimeter']['native_receipt']['first_log'],
            configuration_sha256=sha256(p/'CHOICE_PLAN.json'),input_sha256=read(p/'INPUT_LOCK.json')['manifest_sha256'],
            latest_verifiable_artifact=str(p/'DECISION_PACKET.json'),failure_reason=summary['failure_reason'],
            next_step=summary['next_step']);write(p/'STATUS.json',status)
        shutil.copy2(Path(__file__),p/'code'/Path(__file__).name)
    current=repo/'runs/920'/a.current
    write(repo/'runs/920/latest.json',dict(run_id=a.current,stage='conditional_choice_record_completed',
        updated_at=now,run_directory=str(repo.parent/'runtime'/a.current)))
    indexp=repo/'research/920/data_readiness_v1/NATIVE_SOURCE_EPISODES.json';index=read(indexp)
    for run in [a.previous,a.current]:
        for label in ['perimeter','divider']:
            artifact=repo/'runs/920'/run/label/'native/STEPS.json'
            if not any(x.get('artifact')==str(artifact.relative_to(repo)) for x in index['episodes']):
                index['episodes'].append(dict(run_id=run,candidate=label,artifact=str(artifact.relative_to(repo)),
                    sha256=sha256(artifact),scope='source_principle_transfer' if label=='perimeter' else 'researcher_contrast_to_source_principle',
                    does_not_imply_positive_demonstration=True))
    index.update(updated_at=now,unique_run_count=len({x['run_id'] for x in index['episodes']}),
        native_episode_count=len(index['episodes']),source_project_count=7,
        count_scope='Executed reconstruction episodes, including failed versions and researcher contrasts. Four FCHOICE episodes are one source family, two candidates and two versions; not four professional decisions.')
    write(indexp,index)
    sp=repo/'research/920/decision_packets_v1/SOURCED_DECISIONS.json';data=read(sp)
    source=next(x for x in data['records'] if x['decision_id']=='fhouse_existing_furniture_shared_living')
    source['source_to_scene'].update(initial_proxy_ids=['recon_tall_storage_0'],
        decision_packet=str((current/'DECISION_PACKET.json').relative_to(repo)),
        reconstructed_action_sequence=str((current/'perimeter/native/STEPS.json').relative_to(repo)),
        geometry_provenance='Declared7x4.8x2.7m transfer room and retained research furniture; not surveyed F-House or original instances.',
        contrast_candidate=str((current/'divider/native/STEPS.json').relative_to(repo)))
    source['maturity'].update(source_to_native_objects_bound=True,source_grounded_action_replayed=True,
        whole_room_consequences_reviewed=True,paired_episode_ready=False,
        conditional_choice_comparison_available=True)
    source['readiness_by_layer']['scene_action']=dict(executed=True,tool_calls=4,
        scope='Two candidates in two versions; v1 rejected for orientation. Same source family.')
    source['readiness_by_layer']['professional_decision_pair']=dict(executed=True,positive_demonstration=False,
        conditional_objective_choice_record=True,core_training_ready=False,
        limitation='Programmatic visual/access indicators only; contrast request is researcher-authored, no independent professional preference.')
    source['tool_expression'].update(level='conditional_native_subdecision',full_source_decision_supported=False,
        executable_subdecision='Retained storage placement in one declared shared-living transfer room, with two activity-conditioned objective comparisons.')
    data['counts'].update(unique_source_decisions_with_replay=7,source_grounded_replayed_episodes=13,
        total_native_reconstruction_episodes_including_constructed_contrasts=15)
    data['updated_at']=now
    data['native_episode_count_basis']='NATIVE_SOURCE_EPISODES lists15executions:13 source-linked episodes plus2 researcher contrast candidates; versions and candidates are not independent projects.'
    write(sp,data)
    print(json.dumps(dict(source_decisions_with_replay=7,native_episodes=15,unique_run_count=13,training_updates=0)))


if __name__=='__main__':main()
