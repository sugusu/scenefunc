"""Verify the recovered files and summarize the pinned Git research evidence.

Read-only checks on old studies and bundles. Does not invoke an experimental
model, optimizer, renderer or native scene session. Writes new report files.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys


PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
ROOT = Path('runs/autodl/20261110_native_resource_closure')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_git(path):
    raw = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', PIN+':'+path])
    return raw, hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    def source_json(path):
        raw, sha = read_git(path); sources[path] = sha
        return json.loads(raw)
    src = Path('runs/autodl/20261110_corpus_state_recovery/source/native_project.py')
    before = digest(src)
    assert before == '1c418410b1d5db8606305f200154418dd59844ce1c22fee3afb903286417ad19'
    spec = importlib.util.spec_from_file_location('original_resource_validator', src)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native; spec.loader.exec_module(native)
    verified = []
    for run, meta in [('actual_v2', 'input_metadata'), ('checkpoint_v1', 'checkpoint_metadata')]:
        bundle = ROOT/run/'bundle'
        manifest, state, runtime, task = native.validate_bundle(bundle)
        path = f'runs/920/P920-PAPER-v2_20261105T182424Z/{meta}/manifest.json'
        raw, sha = read_git(path); sources[path] = sha
        assert (bundle/'manifest.json').read_bytes() == raw
        room = state['rooms'][manifest['selected_room_id']]
        verified.append({'run': run, 'files': len(manifest['files']), 'manifest_sha256': sha,
            'bytes_including_manifest': sum(r['bytes'] for r in manifest['files'].values())+len(raw),
            'object_records': len(room['objects']),
            'furniture_objects': sum(o['object_type']=='furniture' for o in room['objects'].values()),
            'original_validator_passed': True, 'original_manifest_unchanged': True})
    assert digest(src) == before
    training_path = 'runs/920/P920-STOPDATA-v1_20261107T102628Z/TRAINING_RESULT.json'
    training = source_json(training_path)
    train_summary = {name: {k: a['training'][k] for k in
        ('status', 'optimizer_updates', 'changed_adapter_tensors', 'peak_allocated_bytes')}
        for name, a in training['arms'].items()}
    matched_path = 'runs/920/P920-MATCHEDROLL-v1_20261108T062940Z/SUMMARY.json'
    matched = source_json(matched_path)
    deficits = [r for r in matched['rows'] if r['transition']['initial_status']=='fail']
    eval_path = 'runs/920/P920-EVALFIX-v1_20261107T130916Z/LEARNING_RESULTS.csv'
    raw, sha = read_git(eval_path); sources[eval_path] = sha
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    eval_summary = {}
    for arm in sorted({r['arm'] for r in rows}):
        subset = [r for r in rows if r['arm']==arm]
        eval_summary[arm] = {'trajectories': len(subset),
            'activity_pass': sum(r['activity_status']=='pass' for r in subset),
            'improvement_without_new_loss': sum(r['scoped_improvement_without_new_loss']=='True' for r in subset)}
    learn_path = 'runs/920/P920-LEARN-v1_20261108T075646Z/STATUS.json'
    learn = source_json(learn_path)
    available = subprocess.check_output(['git','ls-tree','-r','--name-only',PIN,
        'runs/920/P920-LEARN-v1_20261108T075646Z'], text=True).splitlines()
    assert not any(p.endswith('TRAINING_RESULT.json') for p in available)
    restoration = {name:json.loads((ROOT/name/'summary.json').read_text()) for name in
        ['actual_v1','actual_v2','checkpoint_v1']}
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'source_pin': PIN,
        'working_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'recovery':verified, 'first_attempt_exact_files':restoration['actual_v1']['exact_files'],
        'source_validator_sha256_before':before, 'source_validator_sha256_after':digest(src),
        'new_recovery_run_seconds':{k:v['wall_seconds'] for k,v in restoration.items()},
        'historical_training':train_summary, 'historical_evalfix':eval_summary,
        'historical_matched':{'project_families':matched['independent_project_families'],
            'exposure':matched['exposure'],'arms':matched['arms'],
            'deficit_trajectories':len(deficits),
            'deficit_model_calls':sum(r['model_calls'] for r in deficits),
            'deficit_native_calls':sum(r['native_calls'] for r in deficits),
            'deficit_unclosed_reason_errors':sum('Unclosed reasoning block' in e for r in deficits
                for e in r['validation_rejections'])},
        'latest_learn_published_status':learn['status'],
        'latest_learn_status_date':learn['updated_at'],
        'latest_learn_final_training_receipt_published':False,
        'latest_learn_live_process_inspected':False,
        'complete_original_or_proven_equivalent_view_units':9,'source_units':24,
        'recovered_missing_state_units':15,'missing_decision_targets':14,
        'recommendation':'Continue one bounded equal-supervision full-scene test after input/runtime closure; no scale-up or new objective claim.',
        'new_model_calls':0,'new_training_updates':0,'new_scene_generations':0,
        'native_session_resumed':False,'native_runtime_instantiated':False,
        'new_renders':0,'method_gain_established':False,'whole_scene_success':None,
        'training_ready':False,'gpu_requested':False,'source_hashes':sources}
    write(args.output/'summary.json',result)
    with (args.output/'historical_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['study','arm','trajectories','activity_pass',
            'improvement_without_new_loss','satisfied_start_gate_pass','satisfied_starts',
            'deficit_resolved','deficit_starts'])
        writer.writeheader()
        for arm,row in eval_summary.items(): writer.writerow({'study':'EVALFIX','arm':arm,**row})
        for arm,row in matched['arms'].items(): writer.writerow({'study':'MATCHEDROLL','arm':arm,
            'trajectories':row['trajectories'],'satisfied_start_gate_pass':row['satisfied_start_gate_pass'],
            'satisfied_starts':row['satisfied_start_count'],'deficit_resolved':row['deficit_resolved'],
            'deficit_starts':row['deficit_start_count']})
    scripts = [Path(__file__),Path(__file__).with_name('recover_paper_native_resources.py'),
        Path(__file__).with_name('restore_frozen_paper_checkpoint.py')]
    for script in scripts:ast.parse(script.read_text())
    write(args.output/'reading_ledger.json',{
        'scope':'Targeted Git source and result audit, not a new full literature review.',
        'original_sources':sources,
        'current_source_reading':['phase_policy.py: actual generation adapter and two-stage cumulative budget',
            'verify_phase_material.py: original-view encoding and constructed-logit oracle; not a model training result',
            'resolve_training_views.py: bounded exact-image lookup; missing images remain missing',
            'native_project.py: resource validator, not native session startup',
            'recover_paper_native_resources.py and restore_frozen_paper_checkpoint.py: complete code'],
        'recovery_formula_reference':{'url':'https://github.com/RobotLocomotion/drake/blob/v1.49.0/math/roll_pitch_yaw.cc',
            'scope':'Source rotation formula consulted during resource restoration; no complete Drake runtime substitution.'},
        'not_done':['No new paper-wide novelty survey','No repeat of old model inference or GPU training',
            'No physics, renderer or native session invocation for the recovered Paper bundle',
            'No inspection of the other server live process']})
    write(args.output/'commands.json',{'cwd':str(Path.cwd()),'commands':[
        '/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python -B scripts/autodl/recover_paper_native_resources.py --out runs/autodl/20261110_native_resource_closure/actual_v2',
        'python -B scripts/autodl/restore_frozen_paper_checkpoint.py --base runs/autodl/20261110_native_resource_closure/actual_v2/bundle --out runs/autodl/20261110_native_resource_closure/checkpoint_v1',
        'python -B scripts/autodl/report_native_resource_closure.py --output runs/autodl/20261110_native_resource_closure/review_v1'],
        'note':'Output directories are immutable receipts. For reproduction choose fresh paths and update the dependent base. Recovery depends on prior exact asset_recovery_v4 material; binaries are ignored by Git, not bundled in this report.'})
    link=lambda path,label:f'<a href="https://github.com/sugusu/scenefunc/blob/{PIN}/{path}">{html.escape(label)}</a>'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Git进展与继续条件</title>
    <style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:17px/1.75 sans-serif;color:#202b35}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5df;padding:9px;text-align:left}}strong{{color:#174c71}}code{{overflow-wrap:anywhere}}</style>
    <h1>值得继续一轮有判别力的验证，尚不支持扩大训练</h1>
    <p>核对远端 runs/920 的 {PIN[:7]} 和 AutoDL 工作分支。Git 结果快照不是另一台服务器的实时进程状态。新增恢复是输入资源工作，不是模型效果。</p>
    <h2>已发布的模型结果</h2>
    <table><tr><th>证据</th><th>已完成</th><th>限制</th></tr>
    <tr><td>STOPDATA</td><td>两组适配器各8次参数更新，各144个张量变化</td><td>只是小型训练；训练损失下降不等于三维收益</td></tr>
    <tr><td>EVALFIX</td><td>12条轨迹：活动检查基座3/4，动作2/4，判断＋动作2/4</td><td>改善且不产生新损失均0/4；同一项目的两项任务×两种子，并非12个房间</td></tr>
    <tr><td>MATCHEDROLL</td><td>已满足起点：判断组停止且保持要求2/2，另两组1/2</td><td>两项缺陷修复均0/2；所有6条缺陷轨迹共18次调用、13次思考块未闭合错误、0次原生工具调用。不能据此否定三维推理能力</td></tr>
    <tr><td>LEARN</td><td>Git记录11月8日启动</td><td>没有已发布的最终训练结果；不能当成现在仍在训练或已成功</td></tr></table>
    <p>{link(training_path,'训练记录')} · {link(eval_path,'原始评测表')} · {link(matched_path,'逐轨迹结果')} · {link(learn_path,'最后发布状态')}</p>
    <h2>本地新增恢复已经做到什么</h2>
    <p>Paper 原始输入28/28文件、最终检查点34/34文件均逐字节匹配冻结清单，原资源校验器通过。两包共享大量文件，不能相加称为62份独立资产。最终检查点有15个对象记录，其中9件家具。未启动原生会话，未渲染，也没有新生成房间。</p>
    <p>首次输入重建仅25/28一致。三份SDF差异来自转写时漏掉原代码的零值加法，保留了负零；恢复原操作后匹配。原清单、尺寸、姿态和评测阈值没有改动。失败记录保留。</p>
    <p>这是研究者重建的开发房间，包含声明过的代理家具，不能用作独立设计偏好真值。上轮恢复的15份状态也不等于图像恢复：24单元中完整原图或已证明等价的图像仍只有9单元。14单元缺判断监督，不伪造目标来填补。</p>
    <h2>值得验证的机制及强对照</h2>
    <p>具体问题是：<strong>在相同场景、知识和监督下，让动作显式依据模型自己生成的空间判断，是否比同样学习判断但直接产生动作，更可靠地完成完整房间要求？</strong>当前代码已把该区别落在动作条件输入上；辅助监督、分阶段调用、合法JSON和普通回滚本身不作为创新。</p>
    <p>强对照保留相同知识的冻结基座、普通可计算规则/搜索、相同判断监督但直接输出动作的模型；额外生成后丢弃判断仅用于计算量诊断，不能代替强基线。各组共用资产、工具、观察、验收和累计预算。必须同时包括需要增加/修改和应当停止的状态。</p>
    <p>下一次只检验能否在新房间完成真实工具执行并提高原要求完成度，且不破坏已有正确关系，或同质量降低全部成本。正确停止的两个样例不能单独承担完整生成论文。已有材料全属开发；后续须按房间来源和模板隔离泛化检验。</p>
    <h2>当前不应追加什么</h2>
    <p>不继续扩展潜变量损失候选、给同一失败无限加搜索预算，或把资源恢复和数值梯度检查算成方法收益。先结束可运行输入与训练/部署观测的缺口，再做一轮固定范围的真实模型对照。如果普通方法解释全部收益，降低当前机制投入；不把整个三维智能体方向一并否定。</p>
    <p><strong>本次不请求开启GPU。</strong>本地尚未达到全数据训练及自由生成运行就绪；资源开机不能替代缺失的输入与运行证据。无新增付费API、模型实验、训练或渲染。</p>
    <p><a href="summary.json">机器可读结果</a> · <a href="historical_comparison.csv">历史比较表</a> · <a href="reading_ledger.json">阅读边界</a> · <a href="commands.json">恢复命令</a></p></html>'''
    (args.output/'stage_report.html').write_text(body)
    artifacts=[]
    for path in [*scripts,*sorted(ROOT.rglob('*'))]:
        if path.is_file() and not {'bundle','source_cache'} & set(path.parts) and path.name!='artifact_manifest.json':
            artifacts.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})
    write(args.output/'artifact_manifest.json',{'files':artifacts,'binary_resources_checked_separately':verified})
    print(json.dumps({'verified_bundles':verified,'deficit_diagnostic':result['historical_matched'],
        'report':str(args.output/'stage_report.html'),'artifact_files':len(artifacts)},ensure_ascii=False))


if __name__=='__main__':main()
