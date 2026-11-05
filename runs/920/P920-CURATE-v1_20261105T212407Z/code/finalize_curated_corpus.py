"""Publish candidate roles and control manifests after CPU encoding verification."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    data=read(p/'DATASET.json');enc=read(p/'ENCODING.json');plan=read(p/'PLAN.json');now=datetime.now(timezone.utc).isoformat()
    assert enc['status']=='curated_corpus_official_cpu_encoding_verified'
    assert enc['dataset_sha256']==sha(p/'DATASET.json')
    records={x['path']:x for x in enc['records']}
    action_only=[row['documents'].get('separate',row['documents'].get('shared')) for row in data['rows']]
    action_tokens=sum(records[x['path']]['supervised_tokens'] for x in action_only)
    assert len(action_only)==36 and action_tokens==enc['totals']['separate']['target_tokens']-143
    write(p/'ACTION_ONLY_CONTROL.json',dict(kind='unexecuted_future_control_manifest',root=data['root'],
        references=action_only,actual_encoded_action_tokens=action_tokens,samples=36,
        evaluation='Same executed actions, no source reason targets/prefixes; no optimizer invoked. Future training budget must be reviewed separately.'))
    sources=read(p/'SOURCE_KNOWLEDGE.json');pool={}
    for group,spec in plan['groups'].items():
        if not spec['reason']:continue
        record=sources[group];pool[spec['decision_id']]=dict(decision_id=spec['decision_id'],family_id=record['family_id'],
            allocation=record['allocation'],conditions=record['condition_summary'],source_reason=spec['reason'],
            attribution=record['professional_evidence']['primary_crosscheck'],
            limitations=record['decision_scope']['unsupported_part'],raw_resident_brief_available=False)
    write(p/'KNOWLEDGE_CONTROL_POOL.json',dict(records=list(pool.values()),source_families=4,
        contains_scene_coordinates=False,contains_demonstration_actions=False,
        use='Candidate shared source material for separate learning and future same-knowledge retrieval; retrieval/inference not executed. Never retrieve held-out family material into its test.',training_authorized=False))
    stage_receipts=[]
    for label in ['prepare_execution','encoding_execution','encoding_v2_execution']:
        receipt=read(t/label/'receipt.json');assert receipt['status']!='running'
        stage_receipts.append(receipt)
        for name in ['receipt.json','stage.log']:
            target=p/'execution'/label/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(t/label/name,target)
    assert [r['exit_code'] for r in stage_receipts]==[0,1,0]
    summary=dict(run_id=a.run,status='curated_candidate_corpus_verified_not_training_admitted',updated_at=now,
        counts=data['counts'],encoding=enc['totals'],ordinary_action_control_tokens=action_tokens,
        selected_reason_steps=[x['id'] for x in data['rows'] if x['reason']],
        task_relabelled_steps=[x['id'] for x in data['rows'] if x['input_change']],
        independent_native_replays_added=0,professional_evaluations_added=0,training_updates=0,
        actual_stage_exit_codes=[0,1,0],first_failed_check='Audit v1 treated historical {path,sha256} reference as a string; v2 normalized representation, no data change.')
    write(p/'SUMMARY.json',summary)
    text='''# 来源理由与实际动作的整理版候选语料

本批把已执行的建设和修订记录整理为36个动作、5处理由配对、4个来源家族。两组各41个样本、2166个监督token、205个图像实例；目标token序列和像素相同。配对组输入360598token，分开组360470token，最大样本12031token。输入长度不同，不宣称训练计算完全相等。全部处理在CPU完成，0模型前向、0新场景执行、0训练。

## 改了什么、为什么

| 记录 | 本版处理 | 保留的限制 |
| --- | --- | --- |
| AIR建设 | 仅挂杆添加前使用开放衣物策略来源理由 | 目录只有挂杆，不证明在多种储物方案间判断 |
| AIR修订 | 移衣架修通路仅作普通动作，移除混合计划理由 | 开放储物原则不能直接解释这次平移；另一侧接近损失保留 |
| Guilbault建设/修订 | 各在首个柜体动作前给同一来源理由，其余步骤仅监督动作 | 两段同一家族；理由支持整体集中策略，不验证每个中间动作均改善空间 |
| Paper模式变化 | 理由对应真实移桌动作；任务改写为保持来客预留区可用 | 事后研究任务重标，非新实验；静态沙发未验证睡眠 |
| Waterloo模式变化 | 理由对应第二步移桌，准备性移凳不附来源理由；任务表达活动而非指定拼桌 | 只许两物体调整、固定目录；不是全自由设计或新住户任务书 |

31个普通动作只教已执行的工具操作。5处来源配对有4条不同理由，Guilbault在建设/修订中出现两次，不算两个独立项目。四个家族、六个执行组仍共享部分资产和几何模板，不是新增36条独立专业决定。原始混合理由、原任务、状态和动作链接保存在evaluation_only与逐条清单，历史数据及已训练适配器没有重写。

Paper和Waterloo共14个步骤改了研究任务文字。新文字没有指定目标坐标或操作对象，但现有轨迹是在原冻结实验中执行的。这里只说明已有动作可作为新任务下的候选示范；尚未进行新需求下的模型生成、重新原生执行或独立可行性验收。不能把这种标签整理冒称提高了模型的条件判断能力。

## 对照及验证

`DATASET.json`列明配对/分开两组的精确顺序和哈希。每处来源理由在两组各监督一次，配对动作带有教师强制的同一理由前缀，分开动作没有；31个普通动作文件两组共享。`ACTION_ONLY_CONTROL.json`给出36个相同动作、2023监督token的普通动作基线；它未训练，目标数与混合组不同必须明确计账。`KNOWLEDGE_CONTROL_POOL.json`是四家族同知识控制的候选池，没有示范坐标或动作，尚未执行检索。

真实官方CPU加载器核验46个唯一组件，输入和padding不监督，配对前缀变化确实进入输入，全部动作目标一致。旧研究者计划、被替换的操作指令、未来状态路径和评价哨兵未进入编码文本。图像来自实际执行前状态，哈希与原记录一致，没有生成新的场景图。五个来源理由的语义资格由本次研究者审阅界定，结构断言不能代替专业审阅。

若未来采用每家族token均值再等权平均，理由/来源动作/普通动作的算术损失权重分别约7.30%/16.03%/76.67%。这是从已编码目标推得的系数，不是实际梯度或训练收益，也不是新Trainer验收。

第一次CPU编码审计因旧数据将未来路径保存为对象而发生TypeError，14.567秒/退出1。失败代码与日志保留；仅修正审计路径读取的新版本18.781秒/退出0。候选数据在这两次审计之间没有变。生成候选耗时0.818秒/退出0。

## 研究使用边界

这些是来源支持的局部策略和明确标注的研究重建示范；没有独立专业正向标签。来源复用范围、任务适用条件、真实活动和跨项目评价仍需分别处理。保持原家族分配，不将本批或已暴露开发房间包装为封存测试。新训练须另行审定，原4更新批准已消耗。本版应作为下一次具体训练方案的数据候选，不能在没有独立任务和强对照时扩大性能主张。
'''
    (p/'SUMMARY.md').write_text(text)
    (p/'README.md').write_text('本批入口：[结果与边界](SUMMARY.md)。完整索引 DATASET.json，逐条可审阅样本 dataset/，源条件与引用 SOURCE_KNOWLEDGE.json。服务器 episode 路径与实际五图哈希由索引记录；小型JSON副本可比对恢复，原图/原生资产不复制进Git。运行命令与实际退出在 execution/。\n')
    status=read(p/'STATUS.json');status.update(status=summary['status'],updated_at=now,completed=36,planned=36,
        started_at=stage_receipts[0]['started_at'],ended_at=stage_receipts[-1]['ended_at'],first_log=stage_receipts[0]['first_log'],
        input_sha256=dict(construction_index=plan['construction_index']['sha256'],old_edit_manifest=plan['old_edit_manifest']['sha256']),
        actual_exit_statuses=[0,1,0],failure_reason=None,retained_failed_attempt=summary['first_failed_check'],
        latest_verifiable_artifact=str(p/'ENCODING.json'),next_step='Use as candidate data for a separately reviewed training design and independent evaluation. No further same-data encoding audit or training implied.')
    write(p/'STATUS.json',status);write(repo/'runs/920/latest.json',dict(run_id=a.run,stage=status['status'],updated_at=now,run_directory=str(t)))
    write(repo/'research/920/data_readiness_v1/CURATED_CANDIDATES.json',dict(updated_at=now,run_id=a.run,
        manifest=dict(path=str((p/'DATASET.json').relative_to(repo)),sha256=sha(p/'DATASET.json')),
        encoding=dict(path=str((p/'ENCODING.json').relative_to(repo)),sha256=sha(p/'ENCODING.json')),
        counts=data['counts'],training_authorized=False,training_admitted=False,
        relation='Derived candidate version; does not overwrite source episodes, existing training or historical construction candidate index.'))
    shutil.copy2(Path(__file__),p/'code'/Path(__file__).name)
    print(json.dumps(dict(status=summary['status'],actions=36,reasons=5,families=4,training_updates=0)))


if __name__=='__main__':main()
