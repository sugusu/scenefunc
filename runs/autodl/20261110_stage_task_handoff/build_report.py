"""Publish exact task handoff evidence without a scene-quality claim."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
WORK = Path('/root/autodl-tmp/agent_method_preparation_20261027/work')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def main():
    result = json.loads((ROOT / 'actual_v2/result.json').read_text())
    native = REPO / 'runs/autodl/20261110_live_manager_request/source'
    driver = WORK / 'native_execution_inventory_20261104/source/src/agent_rollout_decoupled.py'
    replay = WORK / 'native_execution_inventory_20261104/source/Designer/replay_chained.py'
    sources = [
        (replay, 'derive_room_scene', 'RoomSpec.prompt when nonempty, else house_prompt -> RoomScene.text_description'),
        (driver, 'RolloutSession.make_roomscene_env', 'furniture/wall/ceiling ToolEnvironment receives session.room_prompt'),
        (native / 'scenesmith/agent_utils/scene_analyzer.py', 'analyze_furniture_for_manipulands',
         'scene.text_description -> scene_description in parent-selection prompt'),
        (native / 'scenesmith/manipuland_agents/stateful_manipuland_agent.py', 'add_manipulands',
         'parent selection precedes context selection and per-parent workflows'),
        (native / 'scenesmith/manipuland_agents/base_manipuland_agent.py', '__init__',
         'normal constructor creates SceneAnalyzer'),
        (REPO / 'scripts/autodl/local_manipuland_workflow.py', '__call__',
         'old original_room_request field actually reads scene.text_description'),
        (REPO / 'scripts/autodl/complete_native_pipeline.py', 'common_manipuland_driver',
         'old callback does not explicitly pass args.prompt to LocalManipulandWorkflow')]
    source_rows = [{'path': str(p), 'sha256': sha(p), 'read_scope': scope, 'observed_flow': flow}
                   for p, scope, flow in sources]
    dump(ROOT / 'source_reading.json', source_rows)
    ledger = [
        {'title': 'SceneSmith v2', 'url': 'https://arxiv.org/html/2602.09153v2',
         'read_now': 'Targeted §3.1/3.3 and appendix G opening/G.1, B.1 and B.5; not a new full-paper reading',
         'finding': 'Parent selection and context furniture are already explicit prior capabilities. '
                    'Original-task transport to a common adaptation is a baseline issue, not a new method.',
         'limits': 'No official whole-system reproduction or new learned result'},
        {'title': 'Fixed native source and current common callback', 'url': 'local:source_reading.json',
         'read_now': 'Functions and flows listed with file SHA; native base pin '
                     'e888297a8b0a2f3acbb8634edf9def59cceab836',
         'finding': 'RoomScene text may be a room-specific prompt, not a verbatim original request. '
                    'The old common workflow alias does not prove original-task provenance.',
         'limits': 'This is a conditional information-flow finding, not a measured frequency of lost requirements'}]
    dump(ROOT / 'reading_ledger.json', ledger)
    summary = {
        'status': 'original_task_passthrough_before_parent_selection_and_during_workflow_implemented',
        'native_room_objects_preserved': result['native_room_objects'],
        'native_state_unchanged': result['native_state_before_sha256'] == result['native_state_after_sha256'],
        'selection_inputs_captured': len(result['selection_rows']),
        'local_workflow_inputs_captured': len(result['workflow_rows']),
        'native_parent_candidates_in_input': len(result['selection_rows'][0]['rendering_params']['include_objects']),
        'model_calls': 0, 'training_steps': 0, 'new_complete_scenes': 0,
        'paid_API_calls': 0, 'preference_labels_created': 0,
        'successful_probe_seconds': result['seconds'],
        'parent_peak_rss_mib': result['parent_peak_rss_mib'],
        'whole_runtime_ready': False, 'learned_advantage_verified': False,
        'goal_achieved': False, 'environment': result['environment'],
        'initial_failure': 'v1 used the exported JSON directory as the resource root. '
                           'Native restore rejected missing room.sdf. v2 explicitly uses '
                           'the existing verified original room root; source files unchanged.',
        'limits': result['limits'],
        'selection_instance_initialization': 'SceneAnalyzer.__new__; service constructor not run. '
                                             'The actual native selection method executes until captured analyze_scene call.',
        'data_disk_free_bytes_directory_measurement': shutil.disk_usage(ROOT).free}
    dump(ROOT / 'summary.json', summary)
    with (ROOT / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['path', 'condition', 'original_task_explicit', 'scene_unchanged', 'model_calls'])
        for kind, rows in [('native_selection', result['selection_rows']), ('common_workflow', result['workflow_rows'])]:
            for row in rows:
                w.writerow([kind, row['condition'], row['condition'] != 'native', True, 0])
    theory = '''原始任务传递与完整生成机制：本轮结论和下一步

目标仍为从文字到完整可编辑室内场景。现在比较的候选是同原始监督的直接动作、
辅助决策直出动作、自生成决策后动作；三者须共享原始要求、后端和预算。
本轮修正的是共同输入，不能记为某个学习方法的收益。

Source flow:
  original U -> floor planner -> room-specific U_room -> RoomScene.text_description
  old common small-object workflow called this last field original_room_request.
room-specific 字段可以等于 U，但不能仅凭字段名认定它保留了 U 全部条件。
本次研究房间的 description 是先前研究者重建描述，不是本轮模型改写的证据。

若两个原始任务经分解得到同一 U_room，而它们对后续对象有不同要求，则只接收
U_room 的后续策略无法从这个输入区分两者。这是信息不足的普通必要条件，不是
模型必然出错的实测，也不是新的理论。保留 U 与 U_room 只消除了这项输入歧义，
仍不能保证模型遵守、正确分配对象或满足空间功能。

修正发生在两个位置：
1. native SceneAnalyzer 的 furniture_analysis prompt_kwargs.scene_description
   携带明确区分来源的 U/U_room，不改变 Scene 文本、对象、网格或家具列表。
2. common local workflow 的 public_context 用 task_provenance 区分原文和派生
   描述。原生选择建议仍是 model proposal，不提升为用户硬要求。
只有后一点不足以影响先前漏选的支撑家具；因此原文必须在 parent selection 前可用。

已有 SceneSmith 原方法包括父家具分析、附近上下文家具和每父对象工作流。这些
都属于前作，不能把接入其本来能力写成新型 Agent。本轮没有测试新的偏好机制。

实现保留旧脚本及结果：新模块对旧工作流 AST 只替换一个上下文字段构造，反向
还原后方法 AST 与原件一致；其他循环、预算、反馈、动作验收未改。完整入口的
新 callback 是显式替代参数，原生产保护未放宽。

实际验证使用原有 15 对象原生 RoomScene；SceneAnalyzer 通过 __new__ 建立，
没有启动其服务构造。选择方法真实过滤得到 6 个候选对象，
在 rendering/model 前截获 kwargs。另执行本地工作流的请求构造，观察与状态为
明确的只读包装，不是 native SDK 工具运行。两个中文任务是输入运输夹具，不是
真实用户偏好或新场景，不从它们构造训练真值。没有产生 completion/ranking。
raw workflow journal 的 started 是人为截获点，进程已经正常终止；不是后台任务。

整场还未就绪，后续工作必须针对实际连接：
- 最近正常资产请求是 native Python3.11 + model Python3.12 RPC；完整桥接仍
  使用进程内 model/processor。现有3.11为CPU Torch、没有Transformers。不能
  把两个独立资格拼成整场可运行，更不能靠多开GPU解决解释器依赖差异。
- 同一模型宿主需同时支持政策和共同工具请求，保持工具侧冻结，不重复加载
  两份8B。当前一请求RPC资格不自动授权新的生成调用或完整政策服务。
- 真实五阶段、材料检索、阶段资产、输出完整性与任务检查仍需整场验证。新callback
  接口正确不等于这些分支已运行。

后续科学比较应先取得共同后端的真实完整生成，再比较同源同预算的三政策；
所有方法共同使用此任务透传和上一轮 Critic 任务透传。若任务透传自身解决错误，
收益归普通基线；只有候选政策在共同基线上再有独立整场收益才讨论研究贡献。
本轮不请求资源，不训练，不新增API，不声称CVPR贡献已成立。
'''
    (ROOT / 'algorithm_readiness_delta.txt').write_text(theory)
    (ROOT / 'INTEGRATION.txt').write_text('''Existing guarded full-generation call may explicitly use:
  from task_grounded_completion import task_grounded_manipuland_driver
  completion_driver = task_grounded_manipuland_driver(tool_vlm_binding)
  # Supply it as run_native_trial(... completion_driver=completion_driver).
Do not bypass service preflight or claim the full runtime is qualified.
This callback passes args.prompt before native parent selection and to the
ordinary per-parent workflow. The caller must preserve args.prompt provenance.
Native construction/exports and full model calls were NOT run in this stage.
''')
    (ROOT / 'COMMANDS.txt').write_text('''From repository root; use the isolated native Python3.11 dependency path
recorded in ../20261110_live_manager_request/COMMANDS.txt. No weight load:
  python -B scripts/autodl/audit_stage_task_handoff.py \\
    --out runs/autodl/20261110_stage_task_handoff/actual_NEW \\
    --state runs/autodl/20261110_native_observation_export/export_v2/actual_room_state.json \\
    --scene-root runs/autodl/20261110_native_state_runtime/checkpoint_restore_v1/imported/room_paper_shared_living_reconstruction \\
    --image runs/autodl/20261110_native_observation_export/checkpoint_views_v1/step_006.topdown.png
Output must be new. Keep --scene-root separate from JSON export directory.
  python -B runs/autodl/20261110_stage_task_handoff/verify_stage.py
''')
    report = '''<!doctype html><meta charset="utf-8"><title>完整任务在阶段间的传递</title>
<style>body{max-width:1000px;margin:35px auto;padding:0 20px;font:16px/1.7 system-ui}pre{white-space:pre-wrap;background:#f3f5f6;padding:20px}</style>
<h1>把原始要求送到父家具选择之前，而不是只补后续摆放提示</h1>
<p>原生 RoomScene.text_description 可以来自房间子任务。旧共同工作流把它命名为原始要求，不能证明来源。现已新增显式的原始/派生任务传递，用于共同强基线。</p>
<p>在旧 15 对象场景上执行原生选择器输入构造和工作流请求构造。原生过滤得到 6 个候选对象；两个任务为运输夹具。成功运行 ''' + f'{result["seconds"]:.3f}' + ''' 秒，没有模型调用、场景生成、训练、偏好标签或几何质量改进结论。</p>
<p>第一次因把 JSON 所在目录误当资源根目录，原生恢复正确拒绝缺失 SDF。已保留失败，并显式指定原有资源根目录；未修改状态文件。</p>
<pre>''' + html.escape(theory) + '''</pre>
<p><a href="actual_v2/result.json">实际输入与原生状态证据</a> · <a href="summary.csv">完整统计表</a> · <a href="source_reading.json">源码读取范围</a> · <a href="reading_ledger.json">论文阅读记录</a> · <a href="INTEGRATION.txt">接入方式</a> · <a href="COMMANDS.txt">运行命令</a></p>
<p>这是普通输入修正；完整生成和独立学习收益仍待验证。没有要求用户开启新资源。</p>'''
    (ROOT / 'stage_report.html').write_text(report)
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in
             {'artifact_manifest.json', 'evidence.zip', 'delivery_audit.json'}]
    files += [REPO / 'scripts/autodl/task_grounded_completion.py',
              REPO / 'scripts/autodl/audit_stage_task_handoff.py']
    manifest = [{'path': str(p.relative_to(REPO)), 'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in sorted(files)]
    dump(ROOT / 'artifact_manifest.json', manifest)
    with zipfile.ZipFile(ROOT / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / 'artifact_manifest.json']:
            z.write(p, str(p.relative_to(REPO)))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
