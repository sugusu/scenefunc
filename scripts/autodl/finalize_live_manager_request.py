"""Audit and package an ALREADY TERMINAL single native asset request.

This script never launches a model, service, download, or scene execution.
Geometry and image-bearing requests stay local under the existing exclusions.
"""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile


REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / 'runs/autodl/20261110_live_manager_request'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (STAGE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def publishable(path):
    rel = path.relative_to(STAGE)
    return (path.is_file() and not path.is_symlink()
            and not any(x in rel.parts for x in ('__pycache__', 'manager'))
            and 'request_local_only' not in path.name
            and path.name != 'request.json'
            and path.name not in ('artifact_manifest.json', 'evidence.zip'))


def main():
    result = load(STAGE / 'actual_v1/result.json')
    protocol = load(STAGE / 'protocol.json')
    assert result['status'] == 'normal_native_manager_completed_real_asset_request'
    assert result['actor_terminal'] and result['successful_assets'] == 1
    assert not result['failed_assets'] and result['fresh_CLIP_calls'] == 0
    assert result['external_network_attempts'] == []
    assert result['local_vlm_actor_state']['admitted_calls'] == 1
    receipts = list((STAGE / 'actual_v1/model_actor/model_calls').glob('*/receipt.json'))
    assert len(receipts) == 1
    model = load(receipts[0])
    assert model['status'] == 'completed' and model['fallbacks'] == 0
    assert model['ended_with_eos'] and model['paid_api_calls'] == 0
    prediction = load(receipts[0].with_name('text_output.txt'))
    processes = load(STAGE / 'actual_v1/service_processes.json')
    processes['model_actor_pid'] = load(STAGE / 'actual_v1/actor_process.json')['pid']
    process_checks = {key: not Path('/proc', str(pid)).exists()
                      for key, pid in processes.items() if key.endswith('pid')}
    assert all(process_checks.values()), process_checks
    source_rows = load(STAGE / 'source_manifest.json')['files']
    assert all(sha(STAGE / 'source' / row['path']) == row['actual_sha256']
               for row in source_rows)
    # Compare against a pre-existing receipt; this is a POST-run revalidation,
    # not a retrospective claim that these checks ran before the request.
    old = load(REPO / 'runs/autodl/20261110_native_retrieval_path/http_v2/summary.json')
    input_checks = []
    for name, expected in old['input_sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            path = REPO / path
        actual = sha(path)
        input_checks.append({'path': name, 'expected': expected,
                             'actual': actual, 'match': actual == expected})
    assert all(row['match'] for row in input_checks)
    raw = Path(protocol['data']) / 'objects/8/84815c44d0b7569cd2c023b75d6f11d21bdb7505.glb'
    assert sha(raw) == '9cad56876f26b5ef504e0d73038925e74461be20afe50dbbd85d09ab5cfae180'
    cleanup = load(STAGE / 'retired_environment_cleanup.json')
    assert cleanup['status'] == 'deleted_verified_retired_environment'
    assert not Path(cleanup['target']).exists()
    assert all(Path(p).exists() for p in cleanup['active_interpreter_targets'])
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                   '--format=csv,noheader'], text=True).strip()
    save('postrun_audit.json', {
        'scope': 'Post-run hashes/process checks only; no new physical replay or inference',
        'native_source_files': len(source_rows), 'source_hashes_match': True,
        'historical_retrieval_inputs': input_checks,
        'original_asset_path': str(raw), 'original_asset_sha256': sha(raw),
        'processes_terminal': process_checks, 'gpu_memory_observed': gpu,
        'retired_environment_absent': True, 'active_interpreters_exist': True,
        'data_disk_free_bytes': shutil.disk_usage(STAGE).free,
        'reserve_bytes': 5 * 1024 ** 3,
        'new_model_calls': 0, 'new_training_updates': 0, 'new_paid_api_calls': 0})
    save('asset_metadata.json', {
        'actual_hssd_id': result['source_hssd_id'],
        'requested_description': protocol['description'],
        'model_material_prediction': prediction['material'],
        'material_ground_truth': None, 'requested_material_satisfied': None,
        'mass_prediction_kg': prediction['mass_kg'], 'physical_mass_measured': False,
        'front_view_prediction': prediction['canonical_orientation'],
        'functional_front_verified': False,
        'different_from_previous_forced_asset': '1e1c18c687ffc14eb4267863fc389ef74c2e26e3',
        'geometry_distribution': 'Local only; no source/derived mesh or image redistribution'})
    summary = {
        'status': result['status'], 'real_local_VLM_calls': 1,
        'input_tokens': model['input_tokens'], 'output_tokens': model['output_tokens'],
        'model_body_seconds': model['elapsed_seconds'],
        'manager_request_seconds': result['manager_request_seconds'],
        'service_body_seconds': result['seconds'],
        'timing_exclusion': 'Initial native module import, preparation, review and report writing are outside service-body clock',
        'peak_cuda_allocated_bytes_after_load_reset': model['peak_cuda_allocated_bytes'],
        'parent_peak_rss_mib_not_process_tree': result['parent_peak_rss_mib'],
        'retrieval_mode': 'One exact historical embedding replay; available local IDs before top-K',
        'router_enabled': False, 'raw_source_AST_isolation': False,
        'drake_collision_geometries': result['drake_collision_geometries'],
        'drake_visual_geometries': result['drake_visual_geometries'],
        'fresh_text_retrieval_ready': False, 'full_runtime_ready': False,
        'new_complete_scenes': 0, 'whole_scene_success': None,
        'paid_api_calls': 0, 'training_updates': 0, 'goal_achieved': False,
        'retired_environment_free_byte_delta': cleanup['free_bytes_after'] - cleanup['free_bytes_before'],
        'cleanup_scope': 'Only broken, unused Python3.10 hsm_gpu environment; old scenes/results/assets and active weights excluded'}
    save('summary.json', summary)
    with (STAGE / 'summary.csv').open('w') as f:
        writer = csv.writer(f, lineterminator='\n'); writer.writerow(['metric', 'value'])
        writer.writerows(summary.items())
    save('reading_ledger.json', {
        'native_pin': 'e888297a8b0a2f3acbb8634edf9def59cceab836',
        'base_files': len(source_rows), 'existing_author_patches': 6,
        'read': {
            'scripts/autodl/run_native_manager_request.py': 'complete',
            'scripts/autodl/local_vlm_rpc.py': 'complete',
            'source/scenesmith/agent_utils/asset_manager.py': '148-195, 486-695, 1009-1042; targeted dispatch and success semantics',
            'source/scenesmith/furniture_agents/tools/furniture_tools.py': '60-118, 273-382, 480-694; move/create return semantics',
            'source/scenesmith/agent_utils/placement_noise.py': '1-93: pose-derived seed; not unseeded placement randomness'},
        'not_claimed': 'Not a complete new paper review, original-provider reproduction, material annotation, or full-room generation'})
    save('handoff.json', {
        'budget_closed': True, 'single_model_request_consumed': True,
        'no_automatic_retry': True, 'no_automatic_training': True,
        'actual_native_nonrouter_asset_path_qualified': True,
        'remaining': [
            'No current original CLIP weight file; fresh arbitrary text retrieval remains unavailable',
            '14,098 index rows versus 14,099 embedding rows: mapping provenance unresolved',
            'Local inventory433 and large-category143/5840; not complete HSSD',
            'Router-on semantic validation was not run; material/front not independently verified',
            'No fresh complete-room, real multimodal training or independent learned-policy gain'],
        'next_research_record': '../20261110_execution_supervision_review/algorithm_review.html'})
    page = '''<!doctype html><meta charset="utf-8"><title>原生资产请求实际运行</title>
<style>body{max-width:1000px;margin:36px auto;font:17px/1.7 sans-serif}pre{white-space:pre-wrap}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>
<h1>正常 AssetManager 公开入口完成一次真实资产请求</h1>
<p>实际运行的是原生完整模块和非 router 分支：真实 HTTP 检索、既有本地8B模型 RPC、Blender、V-HACD、SDF、注册、Drake 冷读。未用 AST 拼接管理器，未人工填写物理结果，没有质量重试。</p>
<p>固定描述是“a wooden nightstand”。检索只回放这一旧描述的真实向量，并先筛选本地可用资产；没有当前 CLIP 权重，没有新文本编码。原始全库索引映射仍未完成溯源，不能宣称任意新描述和完整生成已经准备好。</p>
<p>这次由检索选出84815c…，与之前人工固定的1e1c18…不同。模型预测材质plastic，原请求要求wooden；材质真值未核验，不能把名称绑定当作语义合格。SDF 可解析、惯性合法，也不证明真实质量、功能正面或完整场景成功。router已关闭，未测试原生router开启时的语义检查。</p>
<h2>实际记录与成本</h2><pre>SUMMARY</pre>
<p>59.84秒是服务执行体时间，含启动、真实请求与清理，未包含前面的正常模块导入，也不代表整段实例计费时长。17.98GB显存为模型载入后重置统计的生成峰值；1,191.6MiB是父进程RSS，不是整个进程树峰值。GPU现已空闲，三个子进程均结束。零付费API不等于零GPU成本。</p>
<h2>清理与只读范围</h2><p>按已有清理授权，仅删除已失效、解释器断链且无活进程引用的旧Python3.10 hsm_gpu环境。删除前记录4,852个文件的大小和哈希、包版本及链接。实测空闲字节变化见summary；旧研究结果、资产、当前CPU环境和模型权重保留。这个规模不足以恢复约4GB的CLIP权重。</p>
<h2>结论</h2><p>此前逐段拼接的资产链已接上正常公开管理器入口。这是共同后端的工程证据，不是算法创新，也不是新房间成绩。下一步训练目标分析见相邻execution_supervision_review目录；不能自动启动训练、模型重试或整场批次。</p>
<p><a href="postrun_audit.json">运行后核验</a> · <a href="asset_metadata.json">语义边界</a> · <a href="reading_ledger.json">实际源码阅读</a> · <a href="summary.csv">指标</a> · <a href="retired_environment_cleanup.json">清理凭证</a></p>'''
    (STAGE / 'stage_report.html').write_text(page.replace('SUMMARY', html.escape(json.dumps(summary, ensure_ascii=False, indent=2))))
    files = sorted(p for p in STAGE.rglob('*') if publishable(p))
    manifest = [{'path': str(p.relative_to(STAGE)), 'bytes': p.stat().st_size,
                 'sha256': sha(p)} for p in files]
    save('artifact_manifest.json', {'files': manifest, 'scope': 'Published code/source/metadata; no geometry or inline requests'})
    with zipfile.ZipFile(STAGE / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(STAGE))
        archive.write(STAGE / 'artifact_manifest.json', 'artifact_manifest.json')
    print(json.dumps({'summary': str(STAGE / 'summary.json'), 'files': len(files),
                      'evidence_bytes': (STAGE / 'evidence.zip').stat().st_size,
                      'free_bytes': shutil.disk_usage(STAGE).free}))


if __name__ == '__main__':
    main()
