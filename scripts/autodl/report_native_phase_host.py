"""Publish the actual host integration evidence and remaining research gaps."""
import argparse
import hashlib
import html
import json
from pathlib import Path
from datetime import datetime, timezone


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--work-base', type=Path, required=True)
    args = ap.parse_args()
    root, w = args.root.resolve(), args.work_base.resolve()
    current = read(root/'actual_v3_stop/summary.json')
    first = read(root/'actual_v1/native_stage/phase_host/result.json')
    resumed = read(root/'actual_v2_resume/native_stage/phase_host/result.json')
    stopped = read(root/'actual_v3_stop/native_stage/phase_host/result.json')
    failed_render = read(root/'actual_v1/native_000/render_receipt.json')
    recovered = read(root/'recovered_observation_v1/receipt.json')
    recovered2 = read(root/'recovered_observation_v2/receipt.json')
    failed_render2 = read(root/'actual_v2_resume/native_000/views/receipt.json')
    processor = read(root/'processor_v1/summary.json')
    action_compatibility = read(root/'native_action_compatibility.json')
    failures = read(root/'failures_v2/summary.json')
    coverage = read(root/'deployment_coverage_v2/summary.json')
    before = read(root/'actual_v1/source_hashes_before.json')
    assert all(sha(p) == expected for p, expected in before.items())
    assert current['status'] == 'passed_native_integration_fixture'
    assert recovered['status'] == recovered2['status'] == 'completed'
    assert sum(x['native_calls'] for x in (first,resumed,stopped)) == 2
    assert sum(x['generation_attempts'] for x in (first,resumed,stopped)) == 6
    assert first['final_state_sha256'] == resumed['initial_state_sha256']
    assert resumed['final_state_sha256'] == stopped['initial_state_sha256'] == stopped['final_state_sha256']
    assert stopped['status'] == 'model_stop_proposed'
    sources = [w/'native_execution_inventory_20261104/source/src/agent_rollout_decoupled.py',
               w/'native_execution_inventory_20261104/source/Designer/agent_rollout.py',
               w/'native_execution_inventory_20261104/source/Designer/loop.py',
               w/'memory_replay_source_20261103/src/memory.py',
               w/'paris_local_run_results_20261104/dispatch_corrected.py',
               w/'paris_local_run_results_20261104/render_state.py',
               w/'native_observation_bridge_20261104/source/tools/render_five_views.py',
               w/'native_observation_bridge_20261104/source/tools/render_cardinal_views.py']
    ledger = {'date': datetime.now(timezone.utc).isoformat(),
        'source_pin_scene_designer': 'a0565e50ed7d32c0e09132b9311c0b9b91165405',
        'read': [{'path': str(p), 'sha256': sha(p)} for p in sources],
        'coverage': {
            'agent_rollout_decoupled.py': 'Complete ToolEnvironment, RolloutSession, ModelPolicy, driver_loop, main_async/CLI text read. Only ToolEnvironment class body selectively loaded for the CPU fixture.',
            'Designer/agent_rollout.py': 'Complete shared-helper file read; the apparent stage/main headers contain no driver implementation.',
            'Designer/loop.py': 'Floor-plan service construction and dispatch read; SDK/services not imported.',
            'memory.py': 'Success tracking, furniture add/remove/update, library vs live IDs, handler map and recent calls inspected; not full file reread.',
            'dispatch_corrected.py': 'Complete direct-kernel bridge read and actually run in the saved environment.',
            'render_state.py': 'Complete file read. Preparation cut preserves mesh/axis checks; renderer runs after preparer exits.',
            'render_five_views.py': 'Complete read and executed. Second recovery invokes unchanged entry after four existing images, generating topdown only.',
            'render_cardinal_views.py': 'Complete read and executed; camera/settings unchanged. Second recovery runs missing cardinal views separately.'},
        'not_done': ['No new paper-wide novelty survey', 'No original full SDK/service initialization',
                     'No model inference or trained-policy test', 'No full generation trial'],
        'targeted_theory_reading': {
            'paper': 'Ross, Gordon, Bagnell (AISTATS 2011), A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning',
            'url': 'https://proceedings.mlr.press/v15/ross11a/ross11a.pdf',
            'read': 'Introduction; section 2 notation and Theorems 2.1/2.2; section 3 Algorithm 3.1 and assumptions; section 4.1/4.2 through Theorem 4.1 proof and finite-sample statement.',
            'not_read': 'Not a complete nine-page reread; experiments and later proof material not reviewed here.',
            'not_executed': 'No DAgger implementation, expert relabeling, or model updates.'},
        'primary_reference': 'https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/src/agent_rollout_decoupled.py'}
    dump(root/'reading_ledger.json', ledger)
    record = {'native_calls_total': 2, 'scripted_backend_attempts_total': 6,
        'real_model_calls': 0, 'training_updates': 0, 'new_generated_scenes': 0,
        'counts_across_interruption': [9, 8, 9], 'full_scene_success': None,
        'original_sources_unchanged': len(before),
        'render_failures_retained': [failed_render, failed_render2],
        'recovered_render_seconds': [recovered['elapsed_seconds'],recovered2['elapsed_seconds']],
        'native_host_segment_seconds': [x['elapsed_seconds'] for x in (first,resumed,stopped)],
        'serial_measured_host_and_recovery_seconds': sum(x['elapsed_seconds'] for x in (first,resumed,stopped,recovered,recovered2)),
        'terminal_observation_stop_seconds': current['elapsed_seconds'],
        'actual_cpu_encodings': processor['actual_cpu_encodings'],
        'native_action_compatibility': action_compatibility,
        'new_host_failures': failures['cases'],
        'deployment_coverage': {k:v for k,v in coverage.items() if isinstance(v,(int,bool))},
        'code_execution_note': 'actual_v1 started before ActionRejected refinement. actual_v2_resume exercised compact visible_scene_state and actual add, then failed rendering. actual_v3_stop separately observes the same resulting state and emits scripted STOP. This interrupted chain is NOT an uninterrupted free-model trajectory. Failure behavior exercised by failures_v2. Old artifacts retained.',
        'last_scope': 'New source integration, actual native tools and actual refreshed render observations; no algorithm-effect evidence.'}
    dump(root/'summary.json', record)
    comparison = {
        'candidate': 'Explicit action conditioning on a self-generated scene-bound rationale',
        'shared': ['Native assets and allowed tools', 'Actual state and tool feedback', 'Training source units and token coefficients', 'Acceptance definitions'],
        'strong_controls': ['Same-knowledge frozen base', 'Ordinary rule/search where computable',
                            'Auxiliary D supervision with direct A deployment',
                            'Auxiliary D supervision with D discarded at deployment as a compute diagnostic'],
        'changed_now': ['Native furniture-driver integration', 'Dynamic native object set', 'Cumulative D/A failure accounting'],
        'not_a_method_contribution': ['JSON validation', 'EOS handling', 'Sequential rendering', 'Checkpoint integrity'],
        'open_requirements': ['Recover existing LEARN final receipts before duplicating training',
            'Complete or explicitly preserve missing-view denominator',
            'Match NEW native training and deployment request construction',
            'Evaluate free D/A output through actual full-stage scene generation',
            'Independent source/template-separated rooms with full requirement and cost accounting'],
        'no_new_objective_or_training_authorized_by_file': True}
    dump(root/'method_difference.json', comparison)
    dump(root/'theory_boundary.json', {
        'claim': 'Teacher-input loss and live-scene trajectory quality are different evaluation objects.',
        'existing_result': 'DAgger studies learner-induced state distributions; its guarantees have explicit surrogate/no-regret and expert-feedback assumptions.',
        'our_inference': 'A new action changes the next physical scene. Saved teacher observations cannot be spliced in after a different action and called a closed-loop rollout.',
        'missing_for_theorem_application': ['An appropriate expert/corrective oracle on encountered scene states',
            'A justified relation from token-weighted CE to the task cost or required action error',
            'No-regret or relevant convergence assumptions for nonconvex adapter optimization',
            'Sampling and independent-evaluation assumptions'],
        'not_claimed': ['DAgger bound applies numerically to these 24 units', 'rationale conditioning guarantees improvement',
                        'a new theoretical result', 'the current host is a DAgger algorithm'],
        'source': 'https://proceedings.mlr.press/v15/ross11a.html'})
    def esc(x): return html.escape(str(x))
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>两阶段策略接入原生家具执行</title>
<style>body{{max-width:1050px;margin:36px auto;padding:0 22px;font:17px/1.75 system-ui;color:#192734}}td,th{{border:1px solid #ccd4dc;padding:9px}}table{{border-collapse:collapse;width:100%}}a{{color:#175b96}}code{{background:#edf1f5}}</style>
<h1>宿主接口与原生工具分段核验，尚未运行学习策略</h1>
<p>本轮实现可包装 Scene-Designer 家具 driver 的两阶段策略接口；其他阶段委托原 driver 的分支仅用回调夹具验证，未运行完整 main_async。实际 CPU 测试选择加载原 ToolEnvironment.step 与记忆处理器，使用检查点场景门面和真实网格视图覆盖 observation；没有启动原 SDK 服务或原完整提示构造。原生增删不再受旧 920 编辑入口的固定对象集合限制。这个连接是实验基础，不是新算法。</p>
<p>实际 CPU 验证使用已有巴黎场景副本：9 件家具 → 移除一件后 8 件 → 使用已有资产重新加入后 9 件。动作是明确的脚本夹具，6 次后端调用都是预定 token 输出；真实模型调用、训练更新、新生成房间均为 0。不能从“恢复数量”推出恢复了全部身份、用途或设计要求。</p>
<table><tr><th>实际内容</th><th>结果与边界</th></tr>
<tr><td>原生工具</td><td>2 次真实 remove/add，使用只读直接内核桥。ToolEnvironment.step 与记忆 handler 函数体保留；未启动完整 SDK 服务。</td></tr>
<tr><td>新观察</td><td>实际修改后的两套五视图完成，保留网格导入轴向／顶点检查、原相机和采样设置。两次渲染中断后显式续接；不是一次无中断轨迹，未以旧图代替新状态。</td></tr>
<tr><td>失败计费</td><td>无效 D、后端失败、未授权工具、过期观察、错误 schema 均不冒充 STOP；D+A 累计计数，未知部分 token 记为不完整。</td></tr>
<tr><td>整屋成功</td><td>始终未知。阶段结束、工具成功、物体数量恢复不是完整生成成功。</td></tr></table>
<h2>首次失败与有边界的续接</h2>
<p>首次移除已经执行成功，渲染包装进程退出码 {failed_render['exit_code']}（SIGKILL）。原因未确认为 OOM。原包装在保留 Blender 几何准备状态时启动第二个 Blender；新适配仅把这两个过程串行执行，没有改相机、图像分辨率、24 samples 或场景坐标。失败目录未覆盖，已完成的移除没有重跑。</p>
<p>恢复被移除状态的五视图耗时 {recovered['elapsed_seconds']:.2f} 秒。加入后，串行准备／渲染仍在第三方向收到 SIGKILL，故不能宣称第一次适配已解决资源问题。第二次明确恢复保留已完成两图，将缺失方向分进程运行，最后用原入口完成俯视；耗时 {recovered2['elapsed_seconds']:.2f} 秒。最后仅观察终态并输出脚本 STOP，耗时 {current['elapsed_seconds']:.2f} 秒，没有第三次几何动作。各中断点状态哈希一致。累计原生宿主分段与两次恢复实测 {record['serial_measured_host_and_recovery_seconds']:.2f} 秒，包含失败开销；这不是算法效率成绩。</p>
<p>另实测两个新观察、三种部署前缀共 {processor['actual_cpu_encodings']} 次 CPU 多模态编码，各条件的视觉张量一致，耗时 {processor['wall_seconds']:.2f} 秒。没有模型前向或训练。这只证明新观察可编码，仍不证明新旧训练提示分布匹配。8 件非目标家具的序列化状态逐项未变。22 条旧动作均通过原生参数 schema，另有 2 条停止；参数合法仍不证明动作能完成设计。</p>
<h2>训练与部署还不能直接画等号</h2>
<p>逐条核对当前 24 单元：10 个动作在训练时有 D 前缀，14 个没有；部署的配对策略则每轮都先生成 D 再生成 A。它是部分理由监督下的部署，并非全部材料都做了相同形式的条件训练。这不证明策略无效，但必须保留分母，并实际测试自由生成的 D；不能按测试时教师理由是否存在来路由。</p>
<p>新宿主使用原生观察＋实际对象状态＋真实执行反馈，尚未验证它与旧 920 训练请求匹配。之前的前缀检查只支持旧请求格式。下一次同材料训练前，必须用共同构造器明确输入，不能把此工程接入宣称为已经完成匹配实验。</p>
<p>理论上，<a href="https://proceedings.mlr.press/v15/ross11a.html">DAgger</a> 已研究由策略自身行动引起的状态分布变化，并在明确假设下给出保证。这里据此要求评价模型自己造成的后续状态，不能在动作改变后接回教师轨迹。当前加权 token 损失与真实任务代价的对应、纠正专家及优化保证均未建立，不能直接套用该定理或宣称新理论。</p>
<h2>研究位置与后续</h2>
<p>候选仍是检验自生成理由对真实设计行动的额外价值。现有辅助理由监督、普通规则／搜索及同知识底座必须保留。任何新训练还需要已有 LEARN 终态、完整输入及独立场景；当前没有必要因宿主接口通过就立即开 GPU。</p>
<p>本轮读取了原生四阶段源码与记忆／执行／渲染路径，未做新一轮全面论文排重。没有重新定义 CVPR 成果，也没有证明相同监督下配对方式有效。</p>
<p><a href="summary.json">实际汇总</a> · <a href="reading_ledger.json">源码阅读范围</a> · <a href="method_difference.json">方法和工程边界</a> · <a href="deployment_coverage_v2/units.csv">24 单元覆盖</a> · <a href="actual_v1/native_stage/phase_host/result.json">首次失败</a> · <a href="actual_v2_resume/native_stage/phase_host/result.json">加入后中断</a> · <a href="actual_v3_stop/native_stage/phase_host/result.json">终态观察与脚本 STOP</a> · <a href="processor_v1/summary.json">真实 CPU 编码</a> · <a href="failures_v2/summary.json">失败路径检查</a></p></html>'''
    (root/'stage_report.html').write_text(body)
    print(json.dumps({k:v for k,v in record.items() if isinstance(v,(int,float))}))


if __name__ == '__main__':
    main()
