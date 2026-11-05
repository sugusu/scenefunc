"""Summarize realized fixed-weight fitting, never independent design gains."""
import argparse,json,shutil,hashlib
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve();plan=load(p/'PLAN.json');r=load(t/'evaluate/RESULT.json');receipt=load(t/'evaluate_stage/receipt.json')
    assert r['status']=='completed' and receipt['exit_code']==0 and len(r['rows'])==30
    assert r['optimizer_updates']==r['backward_calls']==r['generation_calls']==0 and r['all_parameters_frozen_and_versions_unchanged']
    groups={};baseline={x['record_id']:x for x in r['rows'] if x['arm']=='base'};detail=[]
    for arm in ['base','separate','paired']:
        rows=[x for x in r['rows'] if x['arm']==arm];assert len(rows)==10
        families=defaultdict(list)
        for x in rows:
            b=baseline[x['record_id']];assert (x['tensor_sha256'],x['target_ids'],x['input_tokens'])==(b['tensor_sha256'],b['target_ids'],b['input_tokens'])
            families[x['family_id']].append(x);detail.append(dict(arm=arm,record_id=x['record_id'],family_id=x['family_id'],mean_nll=x['mean_nll'],delta_nll_vs_base=x['mean_nll']-b['mean_nll'],teacher_forced_correct=x['teacher_forced_correct'],target_count=x['target_count']))
        per={f:dict(steps=len(xs),targets=sum(x['target_count'] for x in xs),mean_nll=sum(x['nll_sum'] for x in xs)/sum(x['target_count'] for x in xs)) for f,xs in families.items()}
        groups[arm]=dict(families=per,macro_family_nll=sum(x['mean_nll'] for x in per.values())/len(per),pooled_token_nll=sum(x['nll_sum'] for x in rows)/sum(x['target_count'] for x in rows),teacher_forced_correct=sum(x['teacher_forced_correct'] for x in rows),targets=sum(x['target_count'] for x in rows),all_target_top1_steps=sum(x['teacher_forced_correct']==x['target_count'] for x in rows),forwards=10)
    for arm in ['separate','paired']:
        g=groups[arm];g['macro_delta_vs_base']=g['macro_family_nll']-groups['base']['macro_family_nll'];g['families_improved']=sum(v['mean_nll']<groups['base']['families'][f]['mean_nll'] for f,v in g['families'].items());g['steps_improved']=sum(x['delta_nll_vs_base']<0 for x in detail if x['arm']==arm)
    summary=dict(run_id=p.name,status='final_adapter_common_input_fit_evaluated',groups=groups,paired_minus_separate_macro=groups['paired']['macro_family_nll']-groups['separate']['macro_family_nll'],same_inputs_targets_verified=True,total_forwards=30,new_training_updates=0,backward_calls=0,actual_generation_calls=0,scene_edits=0,actual_exit_code=receipt['exit_code'],elapsed_seconds=receipt['elapsed_seconds'],adapter_storage_precision='saved FP32 restored exactly after PEFT construction',inference_precision='common BF16 autocast, SDPA',pre_restore_load_differences=r['pre_restore_peft_load_differences'],independent_projects_evaluated=0,professional_design_gain='not_tested',generalization_gain='not_tested',new_training_authorized=False)
    write(p/'SUMMARY.json',summary);write(p/'PER_STEP.json',detail);shutil.copytree(t/'evaluate_stage',p/'evaluate_stage');shutil.copytree(t/'evaluate',p/'evaluation');shutil.copy2(Path(__file__).parent/'evaluate_final_pilot_actions_v2.py',p/'EVALUATOR.py');shutil.copy2(Path(__file__),p/'REPORTER.py')
    text='''# 最终适配器在相同动作输入下的实际拟合\n\n这是对已批准且完成的四次更新的补充评价，不是新训练。冻结底座与两个最终适配器使用同一10个原训练步骤、同五图、同536个动作目标token，不给任何组理由前缀。每组10次前向，总30次；没有优化器、反向、生成或场景编辑。\n\n| 权重 | 三来源等权动作NLL（越低越好） | 合并token NLL | 教师强制top1正确token | 相对底座等权NLL差 |\n|---|---:|---:|---:|---:|\n'''
    for arm,g in groups.items():text+=f"| {arm} | {g['macro_family_nll']:.9f} | {g['pooled_token_nll']:.9f} | {g['teacher_forced_correct']}/{g['targets']} | {g.get('macro_delta_vs_base',0):+.9f} |\n"
    text+='\n| 来源家族 | base | separate | paired |\n|---|---:|---:|---:|\n'
    for f in groups['base']['families']:text+='| '+f+' | '+' | '.join(f"{groups[arm]['families'][f]['mean_nll']:.9f}" for arm in ['base','separate','paired'])+' |\n'
    text+=f'''\n运行{receipt['started_at']} → {receipt['ended_at']}，{receipt['elapsed_seconds']:.6f}秒/exit0。3个来源家族的10个相关步骤不是10个独立项目；这里不做独立设计能力的显著性检验。逐token NLL和top1、逐步差值完整保留。**教师强制会给模型前面的正确动作token；token命中率不是自由生成合法率，更不是整屋成功率。**\n\n## 精度与失败记录\n\n首版在任何前向前发现加载值不等于存档，47.525221秒/exit1，0更新/0前向；见../P920-FINALFIT-v1_20261105T220422Z/。修订版先记录实际差异，再在PEFT模块转为FP32后从原存档恢复数值；开始/结束逐项验证两组各144张量、3833856参数值与dtype精确一致。底座BF16、适配器保存FP32，三组统一BF16 autocast和SDPA；参数版本不变、无梯度。\n\n加载前差异：{json.dumps(r['pre_restore_peft_load_differences'],ensure_ascii=False)}。这一加载路径差异不能直接推断旧Casa生成会变化多少。原生成记录及其负结果保持不变；新的同输入比较也不能与旧训练模式NLL直接相减。\n\n## 解释范围\n\n可以判断：这两个**实际最终权重**在原训练动作、不提供理由前缀的共同输入下，似然较底座增减多少。不能据此判断专业设计、跨项目泛化，不能把配对组较低的原始训练NLL当因果收益。配对训练时动作前看到理由，这里没有，因此是共同直接动作输入下的评价，不覆盖所有理由生成策略。原训练还有研究者操作说明混入理由和步骤相关的问题；后来的CURATE清理未用于这两个权重。\n\n下一阶段训练申请必须以可回答的新问题、合格数据及一致评价为依据；本结果不自动授权训练，也不把学习曲线/样本拟合替代独立整屋评价。\n\n## 恢复\n\nPLAN.json锁定原ENCODING和每个张量、底座分片、适配器文件及评价代码。原权重/张量留服务器，来源和哈希随计划；本目录仅小日志、逐token输出和统计。无新增权重。完整命令见evaluate_stage/receipt.json；不覆盖已有输出，首错停止记录不删。\n'''
    (p/'README.md').write_text(text)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
