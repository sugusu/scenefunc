"""Summarize frozen conditional likelihood without asserting learning or preference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import torch
from transformers import AutoTokenizer
from native_project import sha256
from run_decision_loss import save


def summary(rows):
    return {'examples':len(rows),'target_tokens':sum(x['target_count'] for x in rows),
        'target_weighted_nll':sum(x['nll_sum'] for x in rows)/sum(x['target_count'] for x in rows),
        'mean_example_nll':sum(x['mean_nll'] for x in rows)/len(rows),
        'measured_forward_seconds':sum(x['elapsed_seconds'] for x in rows),
        'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in rows)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.plan.read_text());p=Path(cfg['public_directory']);t=Path(cfg['runtime_directory'])
    enc=json.loads((t/'prepare_absolute/RESULT.json').read_text());actual=json.loads((t/'forward/RESULT.json').read_text())
    assert actual['status']=='completed' and len(actual['rows'])==30
    assert actual['optimizer_updates']==0 and actual['trainable_parameters']==0 and actual['all_gradients_none']
    assert actual['plan_sha256']==sha256(a.plan) and actual['encoding_sha256']==sha256(t/'prepare_absolute/RESULT.json')
    tok=AutoTokenizer.from_pretrained(cfg['model_directory'],local_files_only=True)
    indexed={x['id']:x for x in actual['rows']};encoded={x['id']:x for x in enc['records']}
    prefixes=[]
    for decision in dict.fromkeys(x['decision_id'] for x in actual['rows']):
        batches={}
        for role in ('reason_shared','action_paired','action_separate'):
            ref=encoded[decision+'_'+role];f=Path(ref['tensor_path']);assert sha256(f)==ref['tensor_sha256']
            b=torch.load(f,map_location='cpu',weights_only=True);pos=(b['labels'][0]!=-100).nonzero().flatten()
            assert len(pos)==ref['target_count'] and b['labels'][0,pos].tolist()==ref['target_ids']
            assert indexed[ref['id']]['target_ids']==ref['target_ids']
            batches[role]=(b['input_ids'][0],int(pos[0]))
        r,rp=batches['reason_shared'];s,sp=batches['action_separate'];pa,pp=batches['action_paired']
        common=torch.equal(r[:rp],s[:rp]) and torch.equal(r[:rp],pa[:rp])
        assert common and sp>rp and pp>sp
        prefixes.append({'decision_id':decision,'common_task_prefix_equal':common,
            'separate_action_extra_masked_prefix':tok.decode(s[rp:sp]),
            'paired_action_extra_masked_prefix':tok.decode(pa[rp:pp]),
            'reason_target_starts_with':tok.decode(r[rp:rp+2]),
            'identical_prefix_conflicting_first_targets':False,
            'scope':'Separate actions condition on empty think; paired actions on full reason. Not a direct same-prefix target conflict.'})
    reason=[x for x in actual['rows'] if x['component']=='reason']
    arms={};by_source={}
    for arm in ('paired','separate'):
        action=[x for x in actual['rows'] if x['component']=='action' and x['arm']==arm]
        arms[arm]={'all_targets':summary(reason+action),'action_targets':summary(action),
                   'input_budget':enc['arms'][arm],'reason_forwards_shared_not_reexecuted':10}
        by_source[arm]={sid:summary([x for x in action if x['source_decision']==sid]) for sid in ('air','storage','waterloo')}
    result={'status':'completed','run_id':cfg['run_id'],'reported_at':datetime.now(timezone.utc).isoformat(),
        'source_decisions':3,'dependent_steps':10,'new_source_decisions':0,'new_scene_actions':0,
        'unique_source_forwards':30,'verification_forwards':2,'optimizer_updates':0,
        'arms':arms,'actions_by_source':by_source,'reason_targets':summary(reason),
        'source_containing_mixed_reason_targets':summary([x for x in reason if x['source_principle_in_reason']]),
        'researcher_plan_only_reason_targets':summary([x for x in reason if not x['source_principle_in_reason']]),
        'reason_scope':'Three mixed source-principle/researcher-plan explanations, seven researcher-only plans. No claim of isolated professional-knowledge-token loss.',
        'unique_measured_forward_seconds':sum(x['elapsed_seconds'] for x in actual['rows']),
        'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in actual['rows']),
        'causal_loss_check':actual['causal_loss_check'],'input_token_delta':enc['arms']['paired']['input_tokens']-enc['arms']['separate']['input_tokens'],
        'actual_forward':str(p/'FORWARD.json'),'prefix_diagnostic':str(p/'PREFIX_DIAGNOSTIC.json'),
        'training_authorized':False,'training_admitted':False,'professional_preference_established':False,
        'compute_parity_proven':False,'independent_test':False,
        'claim':'Observational frozen-model likelihood under two teacher-forced action contexts, with matched targets. Neither learned pairing nor new-room generation tested.'}
    save(p/'PREFIX_DIAGNOSTIC.json',prefixes);save(p/'FORWARD.json',actual);save(p/'RESULT.json',result)
    shutil.copytree(t/'prepare_absolute/episodes',p/'episodes',dirs_exist_ok=False)
    shutil.copy2(t/'prepare_absolute/MANIFEST.json',p/'MANIFEST.json')
    receipts={}
    for folder in ('prepare','prepare_absolute','forward'):
        src=t/'execution'/folder;dst=p/'execution'/folder;shutil.copytree(src,dst,dirs_exist_ok=False)
        receipts[folder]=json.loads((dst/'receipt.json').read_text())
    save(p/'EXECUTION.json',receipts)
    code=p/'code';code.mkdir()
    for name in list(cfg['scripts'])+['report_decision_loss.py']:shutil.copy2(Path(__file__).parent/name,code/name)
    save(p/'CODE.json',{f.name:sha256(f) for f in code.iterdir()})
    registry=json.loads((p.parents[2]/'research/920/decision_packets_v1/SOURCED_DECISIONS.json').read_text())
    families={x['family_id'] for x in cfg['sources']}
    save(p/'SOURCE_RECORDS.json',[x for x in registry['records'] if x['family_id'] in families])
    lines=['# 三来源决定的冻结模型条件损失','',
        '本批重用已有来源与真实重建回放：开放挂衣、集中收纳、可分合桌，共3条来源决定、10个相关步骤。没有新增专业经验、场景动作或训练。',
        '三段解释含专业来源原则及研究者计划，另七段仅为研究者操作计划；982个目标token不能全算专业知识。',
        '', '| 动作目标（每token负对数似然，越低表示更易预测） | 有配对理由 | 空理由 |','|---|---:|---:|']
    for sid in ('air','storage','waterloo'):
        lines.append(f"| {sid} | {by_source['paired'][sid]['target_weighted_nll']:.6f} | {by_source['separate'][sid]['target_weighted_nll']:.6f} |")
    lines += ['',f"两组各20个概念样本、982个目标token、100次图像使用；有效输入分别212672和212256token，配对多416。十次相同理由前向共享，只实际执行30次来源样本前向，另两次短文本核对。峰值分配显存{result['peak_allocated_bytes']/1024**3:.3f}GiB。",'',
        '这里的“配对/分开”是同一冻结模型看到不同动作上下文。较低损失不能证明训练有效、专业选择正确或房间改善；正确坐标/对象规划本身也会提示下一动作。不得用三个相关来源和共用重建模板推断独立项目泛化。', '',
        '实际模板核对修正了“同前缀冲突监督”的初步猜测：分开动作有被屏蔽的空<think>段，配对动作有完整理由段，两者不是相同前缀。后续训练对照必须事先规定动作时是否有理由，并交叉报告共同推理条件，避免把上下文分布差异写成专业配对收益。见PREFIX_DIAGNOSTIC.json。', '',
        'RESULT.json同时报告按目标token汇总和先求每样本均值再平均；这两种权重不同。此脚本未运行Trainer、梯度累积、反向传播或优化器，前向显存不证明LoRA训练装得下。使用BF16/SDPA，与前序生成基线的FP16/eager设置不同，不直接比较速度。', '',
        '首次CPU准备因相对输出路径失败（1.62秒，未编码/加载模型）。AMENDMENT_01在修复运行前明确增加一次CPU准备；同代码改用绝对输出路径后通过。保留两份真实启动、首日志和退出回执。没有重试模型输出。', '',
        'PLAN锁定全部原始状态、五视图、动作回执、源文件和模型。STATE_ACTION_BINDINGS保留AIR四元数≤1e-12恢复差异；未冒称状态字节一致。SOURCE_RECORDS保留来源使用范围、替代物、局部收益、过渡代价及未知项。', '',
        '小型输入episodes、指标、代码与日志提交Git；模型权重、张量缓存和大型场景留服务器。模型来源/版本/哈希见PLAN；MANIFEST中的路径以PLAN.input_root为根。使用code/prepare_decision_loss.py在新的绝对输出目录重建张量，然后运行code/run_decision_loss.py；精确命令见EXECUTION。执行前核对同版本环境、模型与已有三来源run图像。不能把已完成目录覆盖或因网络失败重跑。', '',
        '下一步：把明确任务格式与推理协议写入可审定训练设计，同时补独立房间/来源覆盖。本批不申请自动解除训练门禁，也不要求再找专业人员签字来证明原作者确实说过这些理由。']
    (p/'README.md').write_text('\n'.join(lines)+'\n')
    # Preserve the final model stage status; append actual process receipts.
    status=json.loads((p/'STATUS.json').read_text());status.update(stage='final_evidence_review',updated_at=result['reported_at'],
        latest_verifiable_artifact=str(p/'RESULT.json'),actual_process_exit_code=receipts['forward']['exit_code'],
        first_log=receipts['forward']['first_log'],preparation_failure_preserved=True,
        next_step='Fix the prospective training/inference protocol and expand independent source/task evidence; no unauthorized optimization.')
    save(p/'STATUS.json',status)
    latest=json.loads((p.parent/'latest.json').read_text());latest.update(stage=status['stage'],updated_at=status['updated_at']);save(p.parent/'latest.json',latest)
    print(json.dumps({'status':result['status'],'arms':arms,'actions_by_source':by_source},ensure_ascii=False))


if __name__=='__main__':main()
