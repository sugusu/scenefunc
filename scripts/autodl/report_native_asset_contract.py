"""Package actual asset preprocessing and source findings, not model results."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,data):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def functions(path,names):
    tree=ast.parse(path.read_text())
    found=[{'name':n.name,'start':n.lineno,'end':n.end_lineno}
           for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if set(names)!={r['name'] for r in found}:raise ValueError('Missing inspected function')
    return {'path':str(path),'sha256':sha(path),'functions':found}


def main(a):
    out=a.out
    cache=read(out/'cache_v2/summary.json');registry=read(out/'registry_v1/summary.json')
    surface=read(out/'desk_surface_v1/summary.json')
    assert cache['status']=='actual_cache_inspected'
    assert registry['status']=='completed'
    assert surface['status']=='native_extraction_completed'
    checked={}
    for hashes in [read(out/'cache_v2/source_hashes.json'),registry['source_hashes'],
                   read(out/'desk_surface_v1/source_hashes.json')]:
        for p,digest in hashes.items():
            if sha(p)!=digest:raise ValueError('Source changed: '+p)
            checked[p]=digest
    tool=out/'tool_source'
    for r in read(tool/'manifest.json')['files']:
        assert sha(tool/r['path'])==r['sha256']
    manager=a.base/'native_room_bridge_20261104/source/scenesmith/agent_utils/asset_manager.py'
    room=Path('runs/autodl/20261110_native_state_runtime/source/scenesmith/agent_utils/room.py')
    native_extractor=room.parent/'support_surface_extraction.py'
    audit={
        'pin':'e888297a8b0a2f3acbb8634edf9def59cceab836',
        'scope':'Selected native branches and three fully read source files; no transitive zero-network proof.',
        'source_files':[
            functions(manager,['_convert_mesh_to_simulation_asset','_convert_articulated_to_scene_object']),
            functions(room,['extract_and_propagate_support_surfaces','to_drake_directive']),
            functions(native_extractor,['_load_and_prepare_mesh','extract_support_surfaces_from_mesh',
                '_compute_clearance_via_raycasting','_compute_surface_bounds']),
            functions(tool/'vlm_service.py',['__init__','create_completion']),
            functions(tool/'mesh_physics_analyzer.py',['analyze_mesh_orientation_and_material']),
            functions(tool/'sdf_generator.py',['generate_drake_sdf','rescale_sdf'])],
        'findings':[
            'Native ordinary-object Drake directives omit objects without SDF; actual ten-template test exported zero add_model entries. This does not describe every GLB renderer/exporter.',
            'Native registry persistence omits surfaces/placement/immutable fields, but RoomScene can recompute surfaces. Do not infer irreversible upstream functional loss.',
            'Cached geometry is Z-up; native surface extractor unconditionally rotates Y-up input. Explicit inverse conversion recovered exact cached bounds in this desk run.',
            'HSSD ordinary mesh conversion calls VLM orientation/material/mass analysis. Turning off routing alone does not remove that call.',
            'VLMService constructs an OpenAI client and selects Responses or Chat calls. No effective endpoint/account credentials were inspected; source alone does not establish actual billing.',
            'SDF generation uses predicted mass and material friction, mesh inertia and supplied convex pieces. Mesh bounds/support surfaces do not supply these missing properties.',
            'SDF generation mutates collision-piece coordinates with Y-up to Z-up rotation. Any later reuse needs explicit frame ownership; no double-conversion experiment was performed.',
            'Articulated retrieval has a different branch using preprocessed SDF, but no suitable accessible asset pool was established in this turn.'
        ],
        'not_executed':['VLMService construction or requests','native mesh physics analysis',
            'SDF generation/collision decomposition','whole scene generation','new policy training'],
        'next':'Use attributable common preprocessing or a measured local model backend for tool-side analysis; never fill unknown physics with invented ground truth.'}
    write(out/'source_audit.json',audit)
    design={
        'status':'Experimental implications, not a new algorithm or new theorem',
        'candidate':'Compare self-generated D-conditioned furniture actions with same-supervision auxiliary-D/direct-A and ordinary search on final complete scenes.',
        'learning_objective':'Keep previously specified original-token-weighted conditional likelihood. This asset work changes neither labels nor objectives.',
        'cost_identity':'C_total = C_policy + C_tool_models + C_render_and_solver + C_asset_preprocessing; shared one-time preprocessing is reported separately with its amortization count.',
        'reason':'Asset choice, resizing or retries can change tool-side inference and preprocessing. A smaller policy token count is not necessarily a lower complete-scene cost.',
        'cache_controls':[
            'Share immutable source asset pool and preprocessing recipe across arms; preserve source IDs and raw geometry hashes.',
            'For reuse of a model-produced analysis, key on actual images/text, model/version, preprocessing and configuration; raw asset ID alone is insufficient.',
            'Use isolated writable per-arm scene state. Never import a prior arm\'s placement, selected action, success label or finalized scene as asset metadata.',
            'Choose cold versus warmed cache consistently. If preprocessing is shared, disclose its cost rather than charging the first arm only.',
            'Mass/front/material estimates remain estimates with provenance. Unsupported semantics stay unknown.',
            'All arms retain the same allowed modifications and common final-scene requirements; no deletion or size relaxation merely to pass.'
        ],
        'scope_limits':[
            'The ten meshes were selected in earlier work. They are not an unrestricted new asset benchmark or evidence of retrieval coverage.',
            'One successful desk extraction does not certify the whole library, functional access, stability or full pipeline.',
            'No new three-dimensional inductive bias or independent full-scene advantage is established.',
            'Do not train again just because local preprocessing works. First close the real shared execution path and preserve strong ordinary controls.'
        ]}
    write(out/'experimental_implications.json',design)
    reading={'date':datetime.now(timezone.utc).isoformat(),
        'new_full_source_reads':['tool_source/vlm_service.py','tool_source/mesh_physics_analyzer.py','tool_source/sdf_generator.py'],
        'targeted_source_reads':['AssetManager mesh conversion and articulated branch',
            'RoomScene surface extraction/propagation and directive export',
            'SupportSurfaceExtractionConfig, mesh loading/frame conversion, face clustering, clearance and primary extraction'],
        'actual_execution':['Original native directive/registry methods on retained records',
            'Original full-module mesh support extraction on one real desk'],
        'papers':[{'title':'ThinkBLOX','url':'https://arxiv.org/html/2607.13539v1',
            'scope':'Targeted recheck III-A/B and IV; no new complete-paper/figure reading or reproduction.',
            'finding':'Reason-then-place SFT and multimodal progressive layout are direct prior art; current preprocessing is a common tool, not a distinct learned method.'}],
        'new_human_annotations':0,'new_paid_calls':0}
    write(out/'reading_ledger.json',reading)
    summary={'previous_goal_turn':'progress: remote source/results independently checked and existing learning outcomes recalculated; full-scene readiness and strong-control priorities clarified.',
        'this_turn':'progress: ten real cached assets inspected; native serializer comparison implemented/executed; one real mesh processed by unmodified native surface extraction; full physics/service source branches read.',
        'assets_inspected':cache['asset_count'],'cached_sdf_assets':cache['asset_count']-cache['missing_sdf_count'],
        'cached_directive_models':cache['native_drake_add_model_count'],
        'desk_extracted_surfaces':surface['surface_count'],
        'desk_surface_area_m2':surface['surface_areas_m2'],
        'desk_wall_seconds':surface['wall_seconds'],'desk_peak_rss_kib':surface['peak_rss_kib'],
        'registry_complete_records_exact':all(r['complete_record_exact'] for r in registry['results']),
        'source_files_rehashed':len(checked),'old_sources_unchanged':True,
        'new_policy_calls':0,'paid_api_calls':0,'training_updates':0,'new_complete_scenes':0,
        'full_runtime_ready':False,'novel_algorithm_established':False,'gpu_requested':False,
        'remaining':['Shared simulation assets/metadata and all native resource closure',
            'Actual local tool-model/backend behavior and complete cost accounting',
            'Complete original/equivalent training observations and matched full-generation ordinary search',
            'Independent full-scene method outcomes']}
    write(out/'summary.json',summary)
    write(out/'source_verification.json',{'verified':True,'sources':checked,'scope':'Read-only input hashes; not replaying every recorded geometry test.'})
    env={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps'}
    python='/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python'
    checkpoint=str(a.base/'paris_local_run_results_20261104/B_mesh_feedback_sdpa_kv/tools/checkpoint.json')
    commands=[
        ['scripts/autodl/prepare_cached_asset_catalogue.py','--base',str(a.base),'--checkpoint',checkpoint,'--out','NEW_CACHE_OUTPUT'],
        ['scripts/autodl/check_native_registry_snapshot.py','--base',str(a.base),'--checkpoint',checkpoint,
         '--paper-state','runs/autodl/20261110_native_state_runtime/checkpoint_restore_v1/roundtrip_state.json','--out','NEW_REGISTRY_OUTPUT'],
        ['scripts/autodl/extract_cached_native_surfaces.py','--source','runs/autodl/20261110_native_state_runtime/source',
         '--catalogue',str(out/'cache_v2/catalogue.json'),'--asset-id',surface['asset_id'],'--out','NEW_SURFACE_OUTPUT']]
    write(out/'commands.json',{'cwd':str(Path.cwd()),'python':python,'environment':env,'argv':commands,
        'failed_attempt':'cache_v1 omitted required Drake dependency PYTHONPATH; cache_v2 used corrected environment, unchanged sample/code scope.',
        'parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    rows=''.join(f'<tr><td>{html.escape(r[0])}</td><td>{html.escape(r[1])}</td></tr>' for r in [
        ('真实缓存','10 个网格；0 个 SDF；原生物体 Drake 导出为 0 项'),
        ('状态恢复','旧反例中的 1 个支撑面经原注册表变为 0；完整快照保留为 1，另保留 immutable'),
        ('原生桌面提取',f"{surface['surface_count']} 个候选面，面积 {surface['surface_areas_m2'][0]:.6f} m²"),
        ('实际资源',f"{surface['wall_seconds']:.3f} 秒；峰值 {surface['peak_rss_kib']/1024:.1f} MiB"),
        ('研究效果','0 新模型调用，0 训练，0 新完整场景')])
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>共同资产路径：实际执行与方法边界</title>
<style>body{font:17px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:0 22px;color:#17212b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:9px}a{color:#175b9c}</style>
<h1>原生支撑面提取已实际跑通，完整资产流程仍有明确缺口</h1>
<p>这轮把十个现有真实 HSSD 网格与旧场景摆放分开，形成不含旧坐标的资产目录；实现完整原生对象状态保存，并实际运行原生几何支撑提取。它们属于共同执行基础，不是新的三维 Agent 方法。</p>
<table><tr><th>项目</th><th>实际结果</th></tr>'''+rows+'''</table>
<h2>不能误判的两处接口</h2>
<p>原生注册表没有保留全部场景字段，但原生 RoomScene 已有重新计算支撑面的流程。因此不能从字段缺失推导出整个上游系统不可恢复。本轮完整快照保留历史状态；另一个资产目录则不包含历史房间坐标，二者用途不同。</p>
<p>原生支撑提取接受 Y-up 网格，旧缓存为 Z-up。本轮显式反向转换后交给未修改的原生算法，恢复包围盒最大误差为 0。桌面候选面原点高于网格顶面 1 cm，来自原生默认偏移；它不是新阈值，也不是物理接触已经验证。</p>
<h2>为什么完整流程还没有就绪</h2>
<p>这十件缓存没有 SDF。普通物体的原生 Drake 指令导出会跳过缺 SDF 的项，实际得到 0 个 add_model；其他 GLB 导出路径不一定如此。不能用网格可见代替完整仿真资产可用。新提取的桌面也不提供质量、摩擦和凸分解。</p>
<p>完整读取 VLMService、mesh_physics_analyzer、sdf_generator 后确认：普通 HSSD 处理会渲染多视图、调用模型预测朝向/材质/质量，再建立仿真资产。没有实例化这些模型服务或读取账户配置。原生调用可在将来适配本地后端，但其正确性、资源和成本尚未实测。</p>
<h2>对算法与强对照的影响</h2>
<p>当前保留的机制问题仍是：相同监督下，自生成决定再行动，能否改善完整房间，并超过直接动作与等成本普通搜索？ThinkBLOX 已有三维渐进推理与动作监督；这一轮没有发现可单独承担创新的新机制。</p>
<p>比较成本必须包括主策略、工具侧模型、渲染求解和资产预处理。使用相同源资产、参数和缓存规则；可以共享同输入的资产分析，但不能共享另一组的已摆放场景。预测物理属性保留来源和不确定性，不称为真实标注。</p>
<p>下一步补齐共同资产/服务路径，才进入真实整场贯通。不要把这一个桌面的成功扩大成整个资产库准备完成，也不要因此扩大训练。</p>
<p>HSSD 原始网格与转换后的 GLB 保留在本地，不上传 Git；外部资源哈希见清单。复跑需要具备对应资产的访问许可与本地文件。</p>
<p><a href="summary.json">汇总</a> · <a href="desk_surface_v1/summary.json">桌面实际运行</a> · <a href="registry_v1/summary.json">状态恢复对照</a> · <a href="source_audit.json">源码依据</a> · <a href="experimental_implications.json">算法与成本边界</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="commands.json">复跑命令</a></p></html>'''
    (out/'stage_report.html').write_text(report)
    code=[Path('scripts/autodl')/n for n in ('prepare_cached_asset_catalogue.py','native_registry_snapshot.py',
        'check_native_registry_snapshot.py','extract_cached_native_surfaces.py','report_native_asset_contract.py')]
    files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.glb']+code
    external=[p for p in out.rglob('*.glb') if p.is_file()]
    record=lambda p: {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    write(out/'artifact_manifest.json',{'files':[record(p) for p in sorted(files)],
        'local_only_external_assets':[record(p) for p in external],
        'external_asset_scope':'HSSD-derived geometry is not redistributed in Git. Reproduce from authorized local source assets using the recorded command.'})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
