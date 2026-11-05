"""Archive the mixed-corpus audit and its bounded interpretation; no training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    public, runtime = repo / 'runs/920' / args.run, repo.parent / 'runtime' / args.run
    plan, result = read(public / 'PLAN.json'), read(runtime / 'ENCODING.json')
    receipt = read(runtime / 'encoding_stage/receipt.json')
    assert receipt['exit_code'] == 0 and result['plan_sha256'] == sha(public / 'PLAN.json')
    now = datetime.now(timezone.utc).isoformat()
    shutil.copy2(runtime / 'ENCODING.json', public / 'ENCODING.json')
    shutil.copytree(runtime / 'encoding_stage', public / 'encoding_stage')
    shutil.copy2(Path(__file__), public / 'code' / Path(__file__).name)
    index = read(repo / plan['index']['path'])
    choice_notes = {
        'clothing': 'The seven-entry catalogue has an open clothes rail and no closed clothes cabinet. This sample does not compare open versus closed storage under different needs.',
        'storage': 'The brief explicitly requires three cabinets and continuous living space. It leaves placement choices open, but the recorded source pairing covers only the first cabinet addition and one declared layout.',
        'paper_table_modes': 'The brief explicitly asks to move the table for guests and restore dining afterwards. The phase and reserve are supplied; this is execution of a specified activity change, not a test of discovering when a movable table is appropriate.'}
    choices = []
    for case in index['cases']:
        data = read(repo / case['candidate_manifest'])
        row = next(row for row in data['rows'] if row['source_principle_target'])
        ref = row['documents']['paired']
        episode = read(Path(case['server_input_root']) / ref['path'])
        context = json.loads(episode['task_context']['scene_description'])
        choices.append(dict(case=case['case'], episode=ref, input_root=case['server_input_root'],
            source_action_step=row['step'], original_request=context['original_request'],
            adapted_design_request=context['design_request'],
            catalogue=[dict(asset_id=a['asset_id'], name=a['name']) for a in context['asset_catalog']],
            activity_context=context.get('activity_context'), reason_target=episode['turns'][0]['content'],
            actual_action=row['command'], assessment=choice_notes[case['case']],
            assessment_origin='Researcher inspection of the locked input; not a professional preference label.'))
    write(public / 'CHOICE_SCOPE.json', dict(created_at=now, cases=choices,
        independent_applicability_counterfactuals=0, limitations='No alternative catalogue or requirement intervention executed in this audit.'))
    totals = result['totals']; single = totals['separate']; masses = single['equal_source_loss_mass_by_category']
    table = '\n'.join('| ' + ' | '.join(str(v) for v in [c['case'],
        c['arms']['separate']['examples'], c['arms']['separate']['supervised_by_category']['reason'],
        c['arms']['separate']['supervised_by_category']['source_action'],
        c['arms']['separate']['supervised_by_category']['ordinary_action'],
        c['arms']['separate']['input_tokens'], c['arms']['paired']['input_tokens']]) + ' |' for c in result['cases'])
    text = f'''# 三案建设候选：真实监督与决定空间审计

本批核验通过，但尚不足以据此申请以“学会专业取舍”为目标的正式训练。三案可以提供带来源原则的可执行重建示范；它们没有覆盖同一经验何时适用、何时应放弃，也没有新的独立整屋测试。

## 实际执行

运行 `{args.run}`，CPU 官方数据加载器、固定 processor、既有 role mask 和实际 collator。{receipt['started_at']} 至 {receipt['ended_at']}，{receipt['elapsed_seconds']:.6f} 秒，退出 {receipt['exit_code']}。读取32个不同组件文件；两组各29样本、26动作、3理由、145图像实例。0模型加载、0前向、0CUDA初始化、0训练更新。没有重建或重跑任何场景。

| 案例 | 样本/组 | 理由目标 token | 来源对应动作 token | 普通动作 token | 分开输入 token | 配对输入 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}
| 合计 | 29 | 92 | 211 | 1276 | {single['input_tokens']} | {totals['paired']['input_tokens']} |

两组每轮目标序列完全相同，共{single['supervised_tokens']} token；实际图像张量相同。只有三处动作前缀加入来源理由，两组对应输入差{totals['paired']['input_tokens']-single['input_tokens']} token。理由目标包括格式 token，因此92不是纯正文长度。最长样本9932 token，小于冻结的16384上限。

如果后续采用旧试运行的“来源等权、来源内部 token 均值”公式，普通动作损失系数占{masses['ordinary_action']:.4%}、理由占{masses['reason']:.4%}、带来源的动作占{masses['source_action']:.4%}。这是实际计数的算术推导，没有新跑Trainer或测梯度贡献。token质量不同，浮点坐标的冗长程度也影响权重；不能把这个比例当作学习效果或专业知识含量。

## 核验具体覆盖什么

32个组件均核对数据/图片哈希及五图顺序；原始需求字段、改写任务、对象表、目录、最近回执和未知项完整进入实际输入。原始住户任务字段三案均为null，任务为明确标注的研究重建。Paper的布置/来客预留/恢复模式按步骤进入输入。评价哨兵与未来状态路径不进入序列，prompt及padding不监督。三处配对的共同prompt完全相同，动作目标相同，理由前缀有差异。这里的排除检查不等于对所有可能间接泄漏做了证明。

两组理由样本都看得到场景；检验操作化仍为“预测动作时有无教师强制理由前缀”，不是分开组完全没有场景知识。推理时不得提供该项目事后理由。相同监督token和图像不保证相同FLOPs。

## 比 token 更重要的决定空间

- **衣物**：七项目录里只有开放挂杆能承担衣物收纳，没有封闭衣柜备选。原步骤可以示范添加及放置，不能区分模型是否根据短住条件选择开放收纳。
- **集中收纳**：输入明确要三件柜体及连续起居空间，位置仍可选择；目前只在第一柜添加时配理由，只有一个研究布局，没有变化条件下的取舍记录。不能因资产固定就说完全没有空间决定，也不能把一处对应算成完整设施组织复现。
- **活动桌**：任务已经写“需要移开桌子”，模式和预留区也给出。此样本教执行指定转换，尚未测试模型自行选择移动、共用或保留的适用条件。沙发不能展开，留空不是睡眠功能验收。

逐条依据、实际任务、目录和动作见 `CHOICE_SCOPE.json`；这是研究者对输入的审计，不是新增专家评价。三来源仍只有两种声明空间模板，26动作不等于26条专业决定。

## 下一步的具体研究决定

保留23个普通动作作为共同建设材料，保留3条来源原则与可执行对应；不删减材料去人为放大配对效果，也不靠再增加几次更新弥补选择空间不足。不重跑本审计或旧四更新。

下一批专业决定数据应优先解除**需求已指定操作／目录已限定策略**的问题。在新来源或清楚标注的迁移重建中，输入表达活动与不可变条件，提供实际可用的备选；记录为何某种条件下某项操作能保护活动，并保留其他可行解。改变使用条件或空间阻塞后的候选必须实际执行、核后果，不能把构造反例写成设计师历史否决方案。先完成一组有这种决定空间的可审阅记录，再按项目家族扩展；同一房壳派生条件仍只算一个家族，不能充当独立测试。

主对照继续让分开／配对组获得相同真实理由和动作；错配仅作辅助诊断。明确样本选择、归一化、训练步数与同条件推理后，另行申请训练。原OFFLOAD四更新授权已消耗。本批不提出重复四更新申请，不以许可或缺人评为由抹掉现有知识/动作资料；专业整屋评价和训练使用范围仍单列待补。

## 重现与产物

`PLAN.json`冻结代码、官方加载器、processor、三个数据manifest及输入索引哈希；`ENCODING.json`保存每组件的长度、目标解码、输入/目标/图像hash；`encoding_stage/receipt.json`与`stage.log`保存真实启动、首日志和退出。`COMMANDS.json`为原执行命令。原场景与大资产继续按各来源run的恢复说明留服务器。
'''
    (public / 'SUMMARY.md').write_text(text, encoding='utf-8')
    write(public / 'COMMANDS.json', dict(cwd=str(repo), encoding=receipt['command'],
        runner=['scripts/920/run_stage.py', '--timeout', '240'],
        summary=['scripts/920/summarize_source_construction_audit.py', '--run', args.run],
        environment='/data/wyh/conda/envs/c15_genrec/bin/python',
        recovery_note='For a deliberate reproduction choose a new output/receipt path; do not overwrite the original run.'))
    status = read(public / 'STATUS.json')
    status.update(status='completed_cpu_audit_not_training_admission', completed=32, updated_at=now,
        started_at=receipt['started_at'], ended_at=receipt['ended_at'], exit_code=receipt['exit_code'],
        elapsed_seconds=receipt['elapsed_seconds'], first_log=receipt['first_log'],
        latest_verifiable_artifact=str(public / 'ENCODING.json'),
        results={'supervised_tokens_per_arm':1579, 'source_principles':3, 'actions_per_arm':26},
        failure_reason=None, next_step='Broaden genuine choice/condition coverage; no repeat of this audit or unapproved training.')
    write(public / 'STATUS.json', status)
    write(repo / 'runs/920/latest.json', dict(run_id=args.run, stage=status['status'], updated_at=now,
        run_directory=str(runtime)))
    live_path = repo / 'research/920/data_readiness_v1/CONSTRUCTION_CANDIDATES.json'
    live = read(live_path)
    live.update(updated_at=now, mixed_token_budget_verified=True,
        actual_loader_audit={'path':str((public / 'ENCODING.json').relative_to(repo)),
            'sha256':sha(public / 'ENCODING.json')},
        token_budget_per_arm=1579, trainer_mixed_corpus_verified=False,
        scope='Three source-only construction cases encoded:26actions/3reasons per arm. Actual targets and pixels match; restricted choice and two related geometry templates limit applicability conclusions. No new training.')
    write(live_path, live)
    print(json.dumps(dict(status=status['status'], public_directory=str(public))))


if __name__ == '__main__':
    main()
