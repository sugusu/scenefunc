"""Summarize the terminal bounded shared-model run and source implications."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    result = json.loads((ROOT / 'actual_v1/result.json').read_text())
    ledger = json.loads((ROOT / 'actual_v1/actor/shared_model_ledger.json').read_text())
    diagnostic = json.loads((ROOT / 'token_parser_diagnostic.json').read_text())
    cache = json.loads((ROOT / 'prefix_cache_audit.json').read_text())
    clients = result['client_conditions']
    forwards = ledger['forwards']
    phase_results = [x for x in clients if 'result' in x]
    valid = [x for x in phase_results if x['policy_status'] in ('tool_proposed', 'stop_proposed')]
    assert result['status'] == 'bounded_runtime_probe_completed'
    assert len(ledger['model_loads']) == 1 and len(forwards) == 3
    assert result['actor_process_terminal']
    summary = {
        'status': 'real_single_frozen_model_cross_runtime_inference_completed',
        'local_model_forwards': len(forwards), 'configured_forward_cap': 4,
        'model_loads': 1, 'same_model_object_and_parameter_pointer': True,
        'all_parameters_frozen': ledger['model_loads'][0]['all_parameters_frozen'],
        'actual_output_tokens': ledger['total_output_tokens'], 'output_cap': 1536,
        'native_tool_structured_response_returned': clients[1]['status'] == 'returned',
        'strict_policy_contract_valid': len(valid), 'policy_conditions': len(phase_results),
        'direct_policy_status': clients[0]['policy_status'],
        'paired_policy_status': clients[2]['policy_status'],
        'paired_action_stage_executed': any(s['stage'] == 'action' for s in clients[2]['result']['stages']),
        'exact_input_hashes_reencoded_and_matched': True,
        'decision_action_common_prefix_tokens': cache['longest_common_prefix_tokens'],
        'entire_decision_KV_reuse_qualified': False,
        'actor_peak_cuda_allocated_GiB': result['actor_peak_cuda_allocated_bytes'] / 1024**3,
        'actor_peak_cuda_reserved_GiB': result['actor_peak_cuda_reserved_bytes'] / 1024**3,
        'parent_seconds_excluding_preparation': result['total_parent_seconds'],
        'sum_forward_body_seconds_excluding_load': sum(r['seconds'] for r in forwards),
        'full_weight_preflight_sha_verification_seconds': protocol['weight_verification_seconds'],
        'actor_terminal': True, 'actor_exit_code_expected_cleanup_SIGTERM': result['actor_exit_code'],
        'new_complete_scenes': 0, 'policy_actions_dispatched': 0,
        'paid_API_calls': 0, 'training_steps': 0, 'new_preference_labels': 0,
        'whole_runtime_ready': False, 'trained_method_benefit_established': False,
        'goal_achieved': False, 'original_inputs_unchanged': result['original_inputs_unchanged'],
        'disk_free_bytes_directory_measurement': shutil.disk_usage(ROOT).free,
        'limits': ['One fixed existing source-room observation, not an independent test set',
                   'Same frozen base has not been trained for this prospective split contract',
                   'Native tool reuses an old actual input but runs a NEW model forward; no asset-manager rerun',
                   'No phase-policy action was dispatched; no geometry or whole-task result measured',
                   'Current host has no trained-policy adapter role switch; this qualification is frozen-base only',
                   'First policy client time includes model loading; paired action never ran, so timing is not a policy-efficiency comparison']}
    dump(ROOT / 'summary.json', summary)
    dump(ROOT / 'CLOSED.json', {'status': 'closed_after_terminal_run', 'actual_forwards': 3,
        'unused_forward_allowance': 1, 'allowance_carries_forward': False,
        'automatic_retry_allowed': False, 'protocol_sha256': sha(ROOT / 'protocol.json')})
    with (ROOT / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['condition', 'transport', 'policy_status', 'forwards', 'output_tokens',
                    'client_seconds', 'action_dispatched'])
        for row in clients:
            matching = [r for r in forwards if r['role'] == row['condition'] or
                        row['condition'] == 'native_tool' and r['role'] == 'native_tool']
            w.writerow([row['condition'], row['status'], row.get('policy_status', 'not_a_policy'),
                        len(matching), sum(r['output_tokens'] for r in matching), row['seconds'], False])
    readings = [
        {'title': 'Fast-ThinkAct', 'url': 'https://arxiv.org/html/2601.09708v1',
         'primary_project': 'https://jasper0314-huang.github.io/fast-thinkact/',
         'read_now': 'Methods3.1–3.4, selected setup4.1, algorithm1 and B.3 objective/conditioning ablations; not full paper or all figures',
         'finding': 'Compact reasoning, reward-guided distillation and latent conditioning are direct prior work; '
                    'robot trajectory targets are not available merely from scene-edit logs.',
         'not_done': 'No official implementation/weights loaded or reproduced. CVF PDF fetch403; '
                     'technical reading used arXivv1, not a verified identical proceedings revision.',
         'project_code_scope': 'Opened project page exposes paper/post links; no implementation inspected. '
                               'This is not proof that code is unavailable elsewhere.'},
        {'title': 'Existing ThinkAct and ECoT reading',
         'source_ledgers': ['runs/autodl/20261110_exact_view_recovery/reading_ledger.json',
                            'runs/autodl/20261110_reasoning_execution_review/reading_ledger.json'],
         'read_now': 'Checked existing reading ledger to avoid presenting old reading as new',
         'finding': 'Reasoning-to-action and embodied reasoning are established prior; prior readings reused'},
        {'title': 'Current local and native source',
         'read_now': 'phase_deployment, PhasePolicy/TorchPhaseBackend, LocalQwenEngine/LocalVLMService, '
                     'native parse_assistant_text and shared host/client code',
         'finding': 'Actual generated control IDs violate prospective split contract. Native permissive '
                    'parser extracts a tool substring without a completed think block.',
         'not_done': 'No native scene/tool execution or old-source edits'}]
    dump(ROOT / 'reading_ledger.json', readings)
    with (ROOT / 'method_differences.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['condition', 'what_changes', 'status_in_this_stage', 'what_it_does_not_prove'])
        w.writerows([
            ['Shared local host', 'One frozen weight instance for policy and native tool inputs',
             'Real cross-runtime GPU inference completed', 'Not a new learned method or full native scene run'],
            ['Native joint parser', 'Extracts tool blocks even with malformed thought text',
             'Exact source parser replay on actual outputs', 'Does not make a malformed decision stage valid'],
            ['Direct action prospective contract', 'Action only; no generated decision prefix',
             'One real forward; invalid action format retained', 'Not a trained direct-policy result'],
            ['Self-generated D then A contract', 'Own valid decision required before action',
             'One real D forward; A not reached', 'No D-conditioned action effect measured'],
            ['Same-D auxiliary direct action', 'Same decision labels during learning, direct action at deployment',
             'Required future control; no trained adapter exists in this stage', 'Cannot label the frozen base as this trained arm'],
            ['Fast-ThinkAct', 'Preference-guided latent/verbalizer and visual-plan distillation to action model',
             'Selected primary methods read; not implemented or reproduced', 'Robot trajectory targets and reported speedups do not transfer automatically']])
    theory = '''共享模型实跑后：算法、训练目标与可检验结论
2026-11-10；完整可编辑室内场景生成目标保持进行。

1. 新的真实执行证据

Python3.11 客户端和 Python3.12 模型宿主已真实连通。现有公开 SceneBuilder8B
权重只加载一次，全部参数冻结；策略和工具的三个实际前向都指向同一模型对象
和同一首参数指针。没有另载一个工具模型。实际输出464token，共用上限1536。
父进程运行59.198秒，准备及完整分片SHA校验不在此计时内。
现有V10032GB上的峰值CUDA allocated约18.378GiB、reserved约18.961GiB。
这只证明该固定输入的推理足迹，不是训练或整场的最低资源要求。

两种部署条件来自同一原始观察、五视图及原生状态。输入来自已审计组件的原始
messages，明确去掉此前的action阶段后缀，再由原部署代码构造阶段输入。
没有复制target_text、target_token_ids、教师D或assistant_prefix到模型请求。
工具条件复用真实旧请求的输入图文，重新执行本地VLM前向；没有复用旧答案，
没有再次执行完整AssetManager、检索或SDF生成。

2. 未把格式失败藏起来

直接动作产生205token，其中有未完成的think块和工具块；严格动作阶段拒绝。
自生成决策产生157token，未产生决策结束符，且包含工具块；严格决策阶段拒绝，
没有进入后续动作阶段。两者均在自然EOS结束，未触及各自512/192token上限。
工具条件产生102token，返回结构化物理属性估计；材质、质量和朝向仍不是测量真值。

实测token映射区分 </think>=151668 与 </tool_call>=151658。返回序列没有151668，
而有151658；用同一公开tokenizer和原分类函数复放得到相同拒绝。不能用改解析器、
补结束符或额外重试把决策算作成功。原生宽松正则能提取get_current_scene_state
子串，但不因此获得有效决策阶段、完整任务成功或已执行动作。

这不是经过本次目标训练的两个政策之间的效果实验。公开冻结基座没有针对这套
前瞻D/A合同微调；格式适配、决策条件化和三维质量必须分开判断。本例不能证明
训练后方法无效，也没有证明显式D有效。预先给定的4前向上限实际只用3次，剩余
一次不补跑、不转成新的实验额度。进程已经回收，GPU权重不留后台占用。

3. 补入的直接近邻

Fast-ThinkAct已经把教师文本推理压成连续latent，通过偏好监督及视觉计划对齐，
再条件化动作模型。其视觉计划监督来自机器人轨迹；动作模型与推理模型分阶段
训练，不能把这些目标直接换成目前的少量房间日志，就称作同一算法或新理论。
https://arxiv.org/html/2601.09708v1
本轮读方法3.1–3.4、算法1和指定消融，没有运行实现。CVF PDF访问失败，未假装
审阅了出版版本全部内容，也不把论文报告的加速当成本项目的预期收益。

这一前作限制了主张：reason->action、latent压缩和偏好蒸馏本身都不是新贡献。
当前仍优先检验有原始要求依据的决策能否选择更合适的上游编辑和重执行范围，
并改善完整生成；不因为发现latent先例就立即换一套更大的训练架构。

4. 对训练目标的具体约束

固定任务U、实际历史H和共同工具T，候选仍为 pi(D,A|U,H)=pi(D|U,H)pi(A|U,H,D)。
同源控制是：原始A监督直接动作；相同D辅助监督但部署直出A；相同D/A监督且部署
使用自己生成的D。三组都必须完成相同输出合同的适配，不能让决策组接受格式训练
却拿未经适配的直接动作当弱对手。动作与决策标签、家族和组件总权重按原来源保持。
当前原始24样本的覆盖与缺失限制不因本轮推理而扩大；新生成文本不自动变成真值。

若考虑偏好蒸馏，只有可信的质量差别才能形成正负对。对于当前已有某些代理奖励
全部相同的集合，A=(R-meanR)/(stdR+epsilon)=0。按索引挑max/min不提供严格偏好
证据；不得借用前作的loss名字，把无差别代理转成有根据的专家计划。该恒等性是
普通条件判断，不是对Fast-ThinkAct实际训练数据的失败指控。

未来学习收益必须在相同信息和原生工具上确认；上一轮原任务到Critic和父家具选择
的透传应作为共同基线。函数成功、格式通过、说得合理均不能替代原始要求完成。
同一房间的一次部署观察、两个输出和一个工具请求不构成泛化样本或因果优势证据。

5. 参数所有权与预算

本次只有冻结基座，没有训练adapter，也没有比较不同微调checkpoint。当前宿主
没有实现训练adapter的角色切换；不能把未来更新后的整个政策权重同时用于工具
分析，再声称工具基座相同。后续若使用可逆adapter，需要政策启用、共同工具禁用
且状态已核验；若更新完整基座，则要另行保持真正相同的工具模型并计入全部开销。

公平预算应计入两阶段重复的上下文prefill、生成token、工具VLM、图像和场景执行，
而非只数最终动作。直接条件客户端时间含首次权重加载，决策条件没有动作阶段，
不能从本次时间差宣传两阶段更快。文件中同时保留模型前向体时间和完整父进程时间。

对实际模型输入做CPU重编码，全部tensor哈希与真实前向收据一致。动作输入8487
token、决策输入8459token，最长公共前缀只有143token，后面的system任务模式不同。
因此不能直接把整个D阶段KV当成A阶段前缀缓存使用。共同前缀或视觉特征可能另有
复用空间，但本轮未实现，也没有证明它们的数值等价；没有插入虚构有效D来测A。
当前代码每次generate都新建完整输入，并未在D/A之间传递KV。

在此前提下，若未来D合法且继续A，计算账本应包含
 C_DA = C_prefill(H_D) + C_decode(D) + C_prefill(H_A,D) + C_decode(A)，
而直接动作是 C_A = C_prefill(H_A) + C_decode(A)。这不是FLOP/时延的数值定理，
但足以说明相同输出token上限不等于相同实际计算。普通方法应能在共同预算内
使用省下的计算作候选搜索；比较质量随预算的曲线，不能只比较一次动作字符串。
Fast-ThinkAct的空间latent KV条件化也不是当前两个不同system提示间缓存的替代品。

6. 下一项应完成什么

先让同一宿主承接原生joint输出合同下的完整生成控制和各阶段工具，保持本次严格
分阶段失败记录。原生ModelPolicy仍不能在CPU版3.11环境直接加载模型，整场桥接
需要显式使用RPC代理；当前RemotePhasePolicy只接入阶段接口，不等于整场原生
驱动已经运行。材料、阶段资产、共同预算和完整任务验收仍需实际贯通。

之后才在同一完整基线上运行已适配的三政策，以及同预算普通搜索。若改善都能由
原任务透传、输出格式适配或普通搜索解释，收益归这些普通组件。只有同源辅助监督
与普通搜索无法解释的整场收益，才支持继续扩大候选决策机制。

本轮有真实推理与参数复用证据，但没有新完整场景、梯度更新、独立偏好标签或
CVPR级贡献证明。目标保持进行，不因这个单例扩大实验范围或追加付费API。
'''
    (ROOT / 'algorithm_and_theory.txt').write_text(theory)
    (ROOT / 'COMMANDS.txt').write_text('''This stage's inference budget is CLOSED (3 of at most4 forwards used).
Do not rerun inference to regenerate the report. No prompt/sample repair retry.
Read-only postprocessing actually executed:
  python -B runs/autodl/20261110_shared_model_runtime/build_report.py
  python -B runs/autodl/20261110_shared_model_runtime/verify_stage.py

Historical entrypoints, NOT an instruction to reopen this protocol:
  model-Python3.12: scripts/autodl/prepare_shared_model_probe.py --stage <new_stage>
  native-Python3.11: scripts/autodl/run_shared_model_probe.py --stage <that_stage>
The preparation verifies all4 checkpoint shard hashes, removes only the exact
known action-mode suffix from the source observation, and encodes real inputs.
Actor requests are loopback only. Policy actions were NOT dispatched.
The token parser diagnostic uses local CPU tokenizer and exact native AST;
it performs no additional model forwards and changes no parser thresholds.
''')
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in
        [r['condition'], r.get('policy_status', 'structured estimate'),
         len([f for f in forwards if f['role'] == r['condition']]),
         sum(f['output_tokens'] for f in forwards if f['role'] == r['condition']),
         f'{r["seconds"]:.3f}']) + '</tr>' for r in clients)
    (ROOT / 'stage_report.html').write_text('''<!doctype html><meta charset="utf-8"><title>共享模型真实推理</title>
<style>body{max-width:1050px;margin:35px auto;padding:0 20px;font:16px/1.7 system-ui}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #aaa}pre{white-space:pre-wrap;background:#f3f5f6;padding:20px}</style>
<h1>两套Python已共用一个真实本地模型；分阶段格式尚未通过</h1>
<p>实际执行3次本地模型前向，只加载一次冻结权重。相同模型对象与首参数指针用于策略和原生工具输入；没有额外API、训练或场景修改。用时59.198秒，峰值CUDA allocated18.378GiB。准备/校验时间另计。</p>
<table><tr><th>条件</th><th>实际状态</th><th>前向</th><th>输出token</th><th>客户端秒</th></tr>''' + table + '''</table>
<p>自生成决策未结束于正确决策分隔符，所以没有进入动作阶段。剩余前向额度关闭；不补跑到成功。原生宽松解析可抽取工具子串，严格阶段合同仍拒绝；原始返回保留。</p>
<pre>''' + html.escape(theory) + '''</pre>
<p><a href="actual_v1/result.json">完整运行结果</a> · <a href="actual_v1/actor/shared_model_ledger.json">每次真实前向账本</a> · <a href="token_parser_diagnostic.json">实际token及解析器复放</a> · <a href="prefix_cache_audit.json">真实输入缓存条件核查</a> · <a href="protocol.json">运行前固定协议</a> · <a href="reading_ledger.json">阅读记录</a> · <a href="summary.csv">逐条件统计</a> · <a href="method_differences.csv">方法差异</a></p>
<p>完整原生生成尚未接通；此页不把模型服务连通、物理属性估计或格式诊断当作生成成功或新方法效果。</p>''')
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in
             {'artifact_manifest.json', 'evidence.zip', 'delivery_audit.json'}]
    files += [REPO / 'scripts/autodl' / n for n in ('shared_native_model_host.py',
        'prepare_shared_model_probe.py', 'run_shared_model_probe.py', 'diagnose_shared_model_output.py',
        'audit_phase_cache_identity.py')]
    manifest = [{'path': str(p.relative_to(REPO)), 'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(files)]
    dump(ROOT / 'artifact_manifest.json', manifest)
    with zipfile.ZipFile(ROOT / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / 'artifact_manifest.json']:
            z.write(p, str(p.relative_to(REPO)))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
