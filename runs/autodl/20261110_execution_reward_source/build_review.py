"""Package targeted primary-paper/source findings without running upstream tools."""
import hashlib
import html
import json
from pathlib import Path
import zipfile


stage = Path(__file__).resolve().parent
spec = json.loads((stage / 'source_flow_and_theory.json').read_text())
# Correct locations against the fetched pin, not the currently changing main.
spec['source_flow'][0]['location'] = 'task.py:80-84'
spec['source_flow'][1]['location'] = 'task.py:86-103'
spec['source_flow'][2]['location'] = 'task.py:105-141'
spec['source_flow'][3]['location'] = 'task.py:142-150'
(stage / 'source_flow_and_theory.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2) + '\n')
manifest = json.loads((stage / 'source_manifest.json').read_text())
verified = []
for row in manifest['files']:
    if 'error' not in row:
        p = stage / 'source_local_only' / row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        verified.append(row['path'])
pin = manifest['pin']
root = f'https://github.com/CYWang735/Video-Coder/blob/{pin}/Video-Coder-RL/uni-agent/'
report = '''<!doctype html><meta charset="utf-8"><title>新近训练近邻：Video-Coder</title>
<style>body{max-width:1000px;margin:35px auto;font:17px/1.7 sans-serif}pre{white-space:pre-wrap;background:#f2f5f8;padding:15px}</style>
<h1>执行奖励不能代替内容目标：新近方法与源码边界</h1>
<p><a href="https://arxiv.org/html/2611.09894v1">Video-Coder，2026-11-06，v1</a>已把代码执行反馈、最终内容奖励和奖励退火用于三维场景代码智能体。其终点是视频，不能据此声称解决完整可编辑室内场景。当前阅读为定向方法、配置与奖励分支，没有运行其系统、训练、视频服务或付费评价。</p>
<h2>已核对的分支</h2>
<p>固定源码提交PIN。<a href="TASKURL">task.py</a>完整阅读；<a href="REWARDURL">video_reward.py</a>阅读配置、标量奖励、门控、解析及score_rollout流程。奖励把环境接受与内容分数分开，accuracy字段记录接受状态。缓存跳过分支也返回1；这只是代码事实，未证明论文评测实际走过该分支。</p>
<p><a href="CFGURL">发布配置</a>调用独立生成器和外部VLM评价，评测失败默认抛错。我们没有启动这些服务。不能用本地工具返回success替代完整要求；也不能把外部评分失败当作场景语义失败而不报告。</p>
<h2>可确定的奖励推导</h2>
<pre>同一固定组内，全部接受且未触发faithfulness门控：
R_i = λ + α r_i, α>0
R_i−mean(R) = α(r_i−mean(r))
A_i = (r_i−mean(r))/std(r)  （非零标准差，无epsilon）</pre>
<p>因此λ在该组的归一化优势中抵消；固定epsilon也不改变λ的抵消。混合接受/门控组、后续策略分布及采样会改变，所以这不证明退火整体无用，也不是新的定理。它要求我们的对照分别记录执行结果与内容完成，不能只解释一个汇总奖励。</p>
<h2>尚未核实与研究影响</h2>
<p>论文方法段与实验段的样本数量不一致；仓库与论文的训练配置也有差别，当前不选一个数字当作已核实规模。未完整审阅最终复执行的harness实现。源码仅本地保存，下载的许可文件失败，不把未确认许可的源文件放入公开包。</p>
<p>我们的受控比较仍应冻结共同工具侧模型，比较同源辅助决策监督、直接动作与自身决策条件动作，并以原始要求下完整场景与总成本判断。执行/内容双奖励与退火已经有相关先例，不能作为新颖性依据。本轮没有新损失、新训练或新完整场景结果。</p>
<p><a href="source_flow_and_theory.json">精确分支、推导假设与阅读边界</a> · <a href="source_manifest.json">源码固定与下载记录</a> · <a href="../20261110_execution_supervision_review/algorithm_review.html">完整训练目标审查</a></p>'''
report = (report.replace('PIN', pin)
          .replace('TASKURL', root + 'uni_agent/tasks/video_coder/task.py')
          .replace('REWARDURL', root + 'uni_agent/tasks/video_coder/video_reward.py')
          .replace('CFGURL', root + 'examples/quickstart/training/task_config_video_coder_video_reward_apptainer.yaml'))
(stage / 'stage_report.html').write_text(report)
summary = {'status': 'targeted_recent_method_and_reward_source_read',
           'paper': spec['paper'], 'source_pin': pin,
           'verified_local_source_files': verified,
           'source_redistributed': False, 'no_source_runtime_executed': True,
           'new_model_calls': 0, 'paid_api_calls': 0, 'training_updates': 0,
           'new_complete_scenes': 0, 'new_method_claim': False, 'goal_achieved': False}
(stage / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
files = sorted(p for p in stage.iterdir() if p.is_file()
               and p.name not in ('artifact_manifest.json', 'evidence.zip'))
artifacts = [{'path': p.name, 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(stage / 'artifact_manifest.json').write_text(json.dumps({'files': artifacts,
    'excluded': 'source_local_only: license not verified; no raw upstream source redistribution'}, indent=2) + '\n')
with zipfile.ZipFile(stage / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.name)
    z.write(stage / 'artifact_manifest.json', 'artifact_manifest.json')
print(json.dumps({'report': str(stage / 'stage_report.html'), 'files': len(files),
                  'source_pin': pin, 'new_model_calls': 0}))
