"""Build a bounded source/processor receipt; never model quality evidence."""
import ast
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
sys.path.insert(0, str(REPO / 'scripts/autodl'))
from capture_critic_task_inputs import native_factory, sha, TRAIN, ADV
from critic_task_context import bind_factory_import


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    capture = json.loads((ROOT / 'input_capture_v2/capture.json').read_text())
    rows = capture['rows']
    factory, _, calls = native_factory()
    import types
    namespace = types.SimpleNamespace()
    cloned = bind_factory_import(factory, namespace)
    assert cloned.__code__ is factory.__code__
    rollout = WORK / 'critic_scope_bridge_20261104/sources/src/grpo_rollout.py'
    lines = rollout.read_text().splitlines()
    assert 'GRPORolloutBuffer(prompt=room_prompt)' in lines[309]
    assert 'scene=scene, room_prompt=prompt,' in TRAIN.read_text().splitlines()[640]
    dump(ROOT / 'source_flow.json', {
        'source_pin': capture['source_pin'],
        'train': {'path': str(TRAIN), 'sha256': sha(TRAIN),
                  'collect_episode_to_rollout_line': 641,
                  'factory_calls': calls},
        'rollout': {'path': str(rollout), 'sha256': sha(rollout),
                    'original_prompt_to_buffer_line': 310},
        'advantage': {'path': str(ADV), 'sha256': sha(ADV),
                      'optional_user_text_line': 257,
                      'default_prompt_line': 283},
        'factory_body_same_code_object_in_private_namespace': True,
        'meaning': 'Original task reaches native buffer; selected Critic callers omit it. '
                   'Only local fixed source execution is established. No author training run audited.',
        'remote_verification': 'This turn train_grpo.py official pinned SHA matched; '
                               'advantage raw re-fetch failed (record retained). '
                               'Local advantage is fixed by recorded SHA, not a new remote match.'})
    ledger = [
        {'title': 'Improving Context-Aware Preference Modeling for Language Models',
         'url': 'https://papers.nips.cc/paper_files/paper/2024/file/82acbbc04435f6c1e7f656b1cbe4ad82-Paper-Conference.pdf',
         'read_now': '§3 theory, §4 construction, §5 setup/results, appendix C and selected D.1; '
                     '35-page official PDF available; not all appendix pages read',
         'finding': 'Explicit context and preference reversal are prior work; error bound '
                    'separates context inference and conditioned prediction. Not a theorem of 3D quality.',
         'not_done': 'No author reward-model training/code execution or dataset download'},
        {'title': 'Scene-Designer', 'url': 'https://arxiv.org/html/2610.23121v1',
         'read_now': 'Targeted §3.2, appendix B/C critic equations and D.3 instruction wording',
         'finding': 'Paper conditions ranking on instruction U; fixed-source calls inspected here '
                    'use default prompt. API supports custom user_text.',
         'not_done': 'No claim of full-paper re-reading, author training reproduction or capability failure'},
        {'title': 'Pinned public reward and rollout source',
         'url': 'https://github.com/ZJU-OmniAI/Scene-Designer/tree/' + capture['source_pin'],
         'read_now': 'make_reward_critic including nested helpers; default prompt, direct ranker, '
                     'legality wrapper; collect_episode prompt flow and rollout buffer creation',
         'finding': 'Three caller sites omit task text. Six CPU input captures exercise source paths.',
         'not_done': 'No full training module import, renderer, native geometry gate or model inference'},
        {'title': 'Existing 2026-11-04 local context probe',
         'url': 'local:' + str(WORK / 'critic_scope_bridge_20261104'),
         'read_now': 'Reused ALGORITHM_AND_THEORY.txt and actual request-construction script',
         'finding': 'Eight model calls already included ordinary task and measurement prompts; '
                    'not repeated or counted as new results.',
         'not_done': 'No new preference truth or independent evaluation'}]
    dump(ROOT / 'reading_ledger.json', ledger)
    differences = [
        ['Task passthrough', 'Original task at final ranker, legal-subset label alignment',
         'Ordinary interface/control repair', 'Implemented and CPU encoded; no quality result'],
        ['RPR conditional learning', 'Condition on external context; fit conditional preference',
         'Direct prior / required learning baseline', 'Read selected official methods; not trained here'],
        ['Native Scene-GRPO', 'Sibling rank and parent-child improvement plus other rewards',
         'Direct generation/critique prior', 'Fixed calls inspected; no reproduction'],
        ['Decision auxiliary direct action', 'Same original D labels, deployment direct A',
         'Strong same-source supervision control', 'Existing objective code reused, no new training'],
        ['Self-generated D-conditioned action', 'Own decision proposal then tool action',
         'Candidate hypothesis, not established novelty', 'No independent complete-room advantage yet'],
        ['Ordinary search', 'Same upstream operations, task evidence and cumulative budget',
         'Required decision/search control', 'Existing preview evidence reused; not rerun here']]
    with (ROOT / 'method_differences.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['method', 'difference', 'research_status', 'evidence'])
        w.writerows(differences)
    with (ROOT / 'input_summary.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['condition', 'images', 'task_present', 'input_tokens', 'model_calls', 'scope'])
        for r in rows:
            w.writerow([r['name'], r['image_count'], r['contains_original_task'],
                        r['input_shapes']['input_ids'][1], 0, r['scope']])
    summary = {
        'status': 'Ordinary task-context control implemented and real processor inputs verified',
        'new_evidence': 'Pinned reward-call task omission; legal-subset-safe task passthrough',
        'captures': len(rows), 'native_with_task': sum(r['contains_original_task'] for r in rows[:3]),
        'control_with_task': sum(r['contains_original_task'] for r in rows[3:]),
        'extra_input_tokens': [rows[i+3]['input_shapes']['input_ids'][1] -
                               rows[i]['input_shapes']['input_ids'][1] for i in range(3)],
        'successful_capture_seconds': capture['seconds'],
        'initial_capture_failure': 'v1 terminal assertion failure: raw newline versus exact JSON-string '
                                   'task detection. Prompts and failure record retained; v2 corrects validator.',
        'model_calls': 0, 'training_steps': 0, 'new_generated_scenes': 0,
        'network_attempts_during_capture': 0,
        'sources_unchanged': True, 'critic_rankings_created': 0,
        'human_preference_labels_created': 0, 'independent_method_benefit': 'unproven',
        'training_ready': False, 'goal_achieved': False,
        'disk_free_bytes_directory_measurement': shutil.disk_usage(ROOT).free,
        'reserve_bytes': 5 * 1024**3,
        'limits': capture['limits'],
        'next_scientific_test': 'Use the ordinary task-context control for every representation arm; '
                                'separate common-pool ranking from each policy complete-generation outcomes. '
                                'Do not count same-source auxiliary supervision, prompt repair or source '
                                'transport as a novel learned decision mechanism.'}
    dump(ROOT / 'summary.json', summary)
    (ROOT / 'COMMANDS.txt').write_text(
        'From repository root (existing local processor and two archived views required):\n'
        'PYTHONDONTWRITEBYTECODE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 '
        '/root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python '
        'scripts/autodl/capture_critic_task_inputs.py --output '
        'runs/autodl/20261110_critic_task_context/input_capture_NEW\n'
        '# Output path must be new. Does not load model weights or generate rankings.\n'
        'PYTHONDONTWRITEBYTECODE=1 python '
        'runs/autodl/20261110_critic_task_context/verify_stage.py\n')
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in
        [r['name'], r['image_count'], r['contains_original_task'],
         r['input_shapes']['input_ids'][1], 0]) + '</tr>' for r in rows)
    reading = ''.join('<li><a href="' + html.escape(r['url']) + '">' + html.escape(r['title'])
        + '</a>: ' + html.escape(r['read_now']) + '</li>' for r in ledger)
    report = '''<!doctype html><meta charset="utf-8"><title>原任务与 Critic 输入</title>
<style>body{max-width:1050px;margin:36px auto;font:16px/1.7 system-ui;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;text-align:left}pre{white-space:pre-wrap;background:#f4f5f6;padding:20px}small{color:#555}</style>
<h1>原始要求已经进入普通 Critic 对照；这不是新算法</h1>
<p>实际执行固定奖励源码调用，到本地 Qwen 处理器图文编码为止。原生三条路径均未带原任务，透传三条均保留原任务；合法性筛选后的图片标签正确。六份输入不是六个房间，没有调用模型、训练或产生偏好结论。</p>
<table><tr><th>输入条件</th><th>图片数</th><th>原任务</th><th>实际编码 token</th><th>模型调用</th></tr>''' + table + '''</table>
<p>成功编码用时 ''' + f'{capture["seconds"]:.3f}' + ''' 秒，不含阅读、实现及第一次失败。v1 因检查器将含换行原文与 JSON 字符串作字面比较而终止，保留原记录；v2 修正检查器。三候选只复用两幅旧真实场景图，合法性布尔值为接口夹具；未重跑几何检查。</p>
<p>论文将 Critic 写成接收指令 U；本次验证固定公开代码的实际输入，不能据此断言论文训练缺少任务。旧八次任务提示诊断未重复。现在的科学比较仍是同原始标签的直接动作、辅助决策和自生成决策后动作，全部共用相同任务信息、工具与成本口径。</p>
<h2>算法、推导及停止误判的条件</h2><pre>''' + html.escape((ROOT / 'algorithm_and_theory.txt').read_text()) + '''</pre>
<h2>本轮实际阅读</h2><ul>''' + reading + '''</ul>
<p><a href="input_summary.csv">完整输入表</a> · <a href="method_differences.csv">方法差异</a> · <a href="input_capture_v2/capture.json">实际编码证据</a> · <a href="source_flow.json">源码传递链</a> · <a href="COMMANDS.txt">运行命令</a> · <a href="summary.json">边界与状态</a></p>
<p>完整生成成功和独立训练收益仍未建立；未请求扩资源、未下载新权重，目标保持继续。</p>'''
    (ROOT / 'stage_report.html').write_text(report)
    artifact_files = [p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
                      and p.name not in {'artifact_manifest.json', 'evidence.zip', 'delivery_audit.json'}]
    artifact_files += [REPO / 'scripts/autodl/critic_task_context.py',
                       REPO / 'scripts/autodl/capture_critic_task_inputs.py']
    manifest = [{'path': str(p.relative_to(REPO)), 'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(artifact_files)]
    dump(ROOT / 'artifact_manifest.json', manifest)
    with zipfile.ZipFile(ROOT / 'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in artifact_files + [ROOT / 'artifact_manifest.json']:
            z.write(p, str(p.relative_to(REPO)))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
