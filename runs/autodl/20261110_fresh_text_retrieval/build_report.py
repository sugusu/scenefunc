"""Package actual fresh CLIP/native retrieval, without redistributing assets."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    actual=HERE/'fresh_v1'
    s=json.loads((actual/'summary.json').read_text())
    rows=json.loads((actual/'requests.json').read_text())
    calls=json.loads((actual/'embedding_calls.json').read_text())
    parity=json.loads((actual/'historical_parity.json').read_text())
    first=json.loads((HERE/'restore_v1/transfer.json').read_text())
    stop=json.loads((HERE/'restore_v1/budget_termination.json').read_text())
    restored=json.loads((HERE/'restore_v2/transfer.json').read_text())
    protocol=json.loads((HERE/'protocol.json').read_text())
    cleanup=json.loads((HERE/'bytecode_cache_cleanup.json').read_text())
    if not (len(rows)==3 and len(calls)==6 and s['fresh_mesh_responses']==3):
        raise ValueError('Actual bounded execution incomplete; do not package success')
    reading=[
        {'source':'Pinned SceneSmith clip_embeddings.py',
         'url':'https://github.com/nepfaff/scenesmith/blob/e888297a8b0a2f3acbb8634edf9def59cceab836/scenesmith/agent_utils/clip_embeddings.py',
         'read':'Whole module: cache selection, native text/image encoding and cosine similarities',
         'actual_execution':'Unchanged native get_text_embedding and cache-hit loader; local-file OpenCLIP constructor explicitly populated cache',
         'not_claimed':'Native automatic remote weight download, image forward qualification or a modified/new encoder'},
        {'source':'Pinned SceneSmith HSSD data_loader.py',
         'url':'https://github.com/nepfaff/scenesmith/blob/e888297a8b0a2f3acbb8634edf9def59cceab836/scenesmith/agent_utils/hssd_retrieval/data_loader.py',
         'read':'Metadata classes, index loading and lookup, lines1–185',
         'actual_execution':'Native loader, scoring/filtering path and HTTP/export modules fully imported under package namespace',
         'not_claimed':'Every index-image identity independently verified'},
        {'source':'Pinned HSSD HTTP client',
         'url':'https://github.com/nepfaff/scenesmith/blob/e888297a8b0a2f3acbb8634edf9def59cceab836/scenesmith/agent_utils/hssd_retrieval_server/client.py',
         'read':'Retry bound and request path including for attempt in range(max_retries)',
         'actual_execution':'One attempt per fixed request via max_retries1, real loopback server',
         'not_claimed':'General failure-recovery evaluation'},
        {'source':'Original HSM/DFN download records and official model config',
         'revision':'01b771ed0d1395ca5ffdd279897d665ebe00dfd2',
         'model':'apple/DFN5B-CLIP-ViT-H-14-378',
         'url':'https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/tree/01b771ed0d1395ca5ffdd279897d665ebe00dfd2',
         'read':'Original approved request, LFS SHA, successful old download receipt and735byteconfig; installed OpenCLIP2.32 signature/model config',
         'actual_execution':'Recovered same3,947,081,637bytefile inRAM and matched original officialc07a17…SHA; CPU local loading and six text forwards',
         'not_claimed':'New model selection, dataset expansion, training or paid model API'},
        {'source':'Existing five-stage controller and external-service contract',
         'local':['scripts/autodl/complete_native_pipeline.py','scripts/autodl/native_service_contract.py',
                  'scripts/autodl/check_retrieval_constructor.py'],
         'read':'Full common pipeline and service guard; constructor qualification opening through cleanup checks',
         'finding':'Fresh retrieval closes one dependency only. Five-stage policy/model/asset composition and final task verification remain separate; old doubles are not new full-generation evidence.'},
    ]
    dump(HERE/'reading_ledger.json',reading)
    boundary={
        'stage_result':'Actual previously-unseen text descriptions now reach native retrieval/export/cold-load through original CLIP parameters',
        'new_learned_method':False,
        'research_invariants':[
            'All future compared policies must use exactly the same declared local asset pool and availability/mapping exclusions.',
            'Do not silently shift/truncate the old embedding-ID map or treat metadata names as complete semantic truth.',
            'Full original task inputs and required-instance denominators are retained even if assets are unavailable; unavailable/ambiguous requirements remain visible.',
            'Query encoding, size reranking, geometry export and semantic suitability are separate claims. Three returned assets are not three generated rooms.',
            'The explicit native model-cache binding is a local-file integration. The original get_text_embedding function performs the forward; historical vectors only validate three old queries afterward.',
            'The earlier single8Basset-analysis budget remains closed. This stage has no new8Bcalls, no physics responses invented, and no training.',
        ],
        'reusable_entry':'scripts/autodl/native_clip_runtime.py: bind_original_clip',
        'local_binding_steps':[
            'Verify same pinned native source and exact original DFN checkpoint/config hashes.',
            'Call bind_original_clip(native_clip_module,weight_path,original_config,device="cpu") once before native HSSD service starts.',
            'Use unchanged native get_text_embedding for every actual request; do not install the former archive-only provider.',
            'Apply the identical explicit available/prefix-supported candidate filter to every study arm.',
        ],
        'current_capabilities':{
            'fresh_native_text_encoding_executed':True,
            'fresh_native_HTTP_mesh_retrieval_executed':True,
            'new_manager_material_or_front_validation_executed':False,
            'router_on_validation_executed':False,
            'fresh_complete_room_generation_executed':False,
            'independent_policy_or_training_advantage_established':False,
            'full_runtime_ready':False,
        },
        'asset_scope':{'available_local_assets':s['available_inventory_assets'],
            'native_large_category_rows':s['candidate_pool_log'][0]['native_category_candidates'],
            'actual_common_retained_large_rows':s['candidate_pool_log'][0]['common_retained'],
            'excluded_suffix_start':12619,
            'independent_whole_index_identity_proven':False},
        'next_identified_work':[
            'Use the new explicit local-file binding in the common generation entry rather than silently treating the old historical-vector provider as a full retriever.',
            'Continue the already specified same-source direct/D-auxiliary/D-conditioned comparison; do not replace it with a retrieval-improvement claim.',
            'Before an authorized complete-room execution, specify the original full description, all five stage budgets, shared asset operations and final semantic/geometry assessment; include model-side tool calls and failures in total cost.',
            'Do not declare the remaining five training views recovered or add new training labels because these three queries succeeded.',
        ],
        'goal_achieved':False,
    }
    dump(HERE/'research_and_runtime_boundary.json',boundary)
    weight=Path(restored['weight_path'])
    dump(HERE/'ram_weight_lifecycle.json',{
        'weight_path':str(weight),'verified_original_sha256':restored['actual_sha256'],
        'exists_at_packaging':weight.is_file(),'bytes':weight.stat().st_size if weight.is_file() else 0,
        'ephemeral_RAM_not_data_disk':True,
        'retained_for_following_same_model_preparation':True,
        'no_background_model_or_server_retained':s['threads_stopped'],
        'lost_on_instance_restart_or_RAM_cleanup':True,
        'do_not_call_persistent_download_or_ready_full_pipeline':True})
    dump(HERE/'summary.json',{
        **{k:v for k,v in s.items() if k not in ('source_module_hashes','input_hashes','network_blocks')},
        'initial_transfer_budget_seconds':600,'initial_transfer_actual_process_seconds':stop['actual_process_elapsed_seconds'],
        'initial_transfer_terminated_at_budget':True,'initial_completed_ranges':len(first['completed']),
        'single_range_recovery_seconds':restored['seconds'],
        'restoration_body_byte_upper_bound':restored['conservative_total_weight_transfer_upper_bytes'],
        'maximum_historical_vector_error':max(p['max_absolute_error'] for p in parity),
        'data_disk_free_bytes_at_packaging':shutil.disk_usage(HERE).free,
        'deleted_regenerable_bytecode_files':cleanup['planned_files'],
        'cache_cleanup_measured_free_byte_delta':cleanup['measured_free_byte_delta'],
        'original_python_sources_preserved_by_cleanup':cleanup['all_original_sources_unchanged'],
        'semantic_independent_judgment':False,'method_effect_established':False,'goal_achieved':False})
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['query','status','source_asset_id','original_wordnet_key','source_row','seconds','independent_semantic_truth'])
        for r in rows:w.writerow([r['text'],r['status'],r['hssd_id'],r['original_metadata']['wordnet_key'],r['embedding_row'],r['seconds'],'unknown'])
    (HERE/'COMMANDS.txt').write_text('''Actual executed dependency restoration and fresh qualification:
python scripts/autodl/restore_original_clip_in_ram.py --out runs/autodl/20261110_fresh_text_retrieval/restore_v1
The matching live process was terminated at605.28seconds after its600second budget;235/236ranges were complete. Receipt retained, no whole-file restart.
python scripts/autodl/finish_single_clip_ram_range.py --stage runs/autodl/20261110_fresh_text_retrieval
NATIVEPY=/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python
Use the existing native dependency PYTHONPATH recorded in execution_environment.json.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "$NATIVEPY" -u scripts/autodl/run_fresh_native_clip_retrieval.py --stage runs/autodl/20261110_fresh_text_retrieval

Do not overwrite existing execution directories or automatically rerun inference.
A new manual reproduction must use a new stage directory with the same frozen three-query protocol and an explicit valid restoration receipt. The RAM checkpoint is temporary; never bypass its full SHA/config checks.

Full generation integration uses native_clip_runtime.bind_original_clip before native service startup. No old query-vector replay wrapper belongs on that fresh path.
Future reproductions should also set PYTHONDONTWRITEBYTECODE=1 to avoid refilling source-backed bytecode caches; this was not set for the archived qualification.
''')
    contents=['<h1>从新描述到真实网格：原生检索已实际跑通</h1>',
      '<p>本轮恢复此前已授权的同一官方CLIP权重，执行六次真实文本编码，并将三个新描述送入原生HSSD HTTP检索、网格导出和冷加载。三项均完成。这是完整场景生成的共同后端准备，不是新的Agent算法或三间完整场景。</p>',
      '<h2>实际结果</h2><table><tr><th>新描述</th><th>源元数据</th><th>原生行号</th><th>请求秒数</th><th>网格核对</th></tr>']
    for r in rows:
        vals=[r['text'],r['original_metadata']['name']+' / '+r['original_metadata']['wordnet_key'],r['embedding_row'],f"{r['seconds']:.3f}",'导出、冷加载及尺寸一致']
        contents.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>')
    contents+=['</table><p>元数据来源与广义类别相符，但没有新增独立视觉／用途评审。“low”的尺度适合性、正面、材质与质量仍未知。三件源资产的up/front元数据均为空；本轮没有用手写属性补齐，也没有调用8B模型推断属性。</p>',
      '<h2>真实新编码，不是旧向量回放</h2>',
      f'<p>实际文本编码{s["actual_text_encoder_forwards"]}次：3次旧查询一致性核对、3次新请求。最大旧向量差为{max(p["max_absolute_error"] for p in parity):.9g}，三项余弦相似度均为1。旧向量只在新编码完成后作比较；新描述不在旧查询档案内。原生get_text_embedding函数未重写，封装只计时、保存输出并记录调用；模块缓存由固定本地文件显式加载。</p>',
      '<p>模型为原ViT-H-14-378-quickgelu／DFN5B固定版本。权重完整SHA与原官方记录c07a17…一致，配置SHA与原文件一致；加载时还核对安装的OpenCLIP2.32架构，并显式传入官方预处理配置。本轮只验证文本路径，不能据此宣布图像前向或自动远端下载路径已验证。</p>',
      '<h2>候选池与索引限制</h2>',
      '<p>本地库存433个资产。原生large_objects筛选有5840行，本次统一保留可用且位于已知注释偏移边界之前的141行。旧索引14098项、嵌入14099行的整体映射仍未修复；没有删除、移动或截断原矩阵。排除已知偏移后缀也不等于独立验证了每一行。后续所有方法必须共享相同池与排除规则，无法覆盖的原要求仍计入任务记录，不得挑选成功样例隐藏资产缺失。</p>',
      '<h2>完整资源与失败记录</h2>',
      f'<p>官方权重3,947,081,637字节暂存内存文件系统，未写入数据盘。官方域名经现有代理失败，直连镜像提供了同版本字节范围；最终必须通过原官方全文件SHA。首轮235/236分块完成后，一个连接停滞；在确认匹配进程运行{stop["actual_process_elapsed_seconds"]:.2f}秒并超过600秒预算后将其终止。随后只恢复16MiB缺块，耗时{restored["seconds"]:.2f}秒，并校验整文件。未重下全文件或无限重试，完整记录保留。</p>',
      f'<p>模型加载与核验{s["model_load_seconds"]:.2f}秒，编码／请求执行主体{s["execution_seconds"]:.2f}秒，进程峰值RSS{s["peak_rss_kib"]/2**20:.2f}GiB。后者不包含全部内存文件系统与外部进程，不能当作整套实例最低内存。实际使用已有25GiB限额实例、CPU4线程，未使用GPU。数据盘仍保留5GiB余量，没有删除旧结果或活动模型。</p>',
      '<p>首次下载中断、单块恢复、模型加载和请求均属于准备成本，不能只报告某个请求的几秒钟来宣传算法效率。无新增付费API、8B语言模型调用、训练更新、物理SceneObject创建或完整房间生成。CPU运行不等于整套实例不计费，本报告没有供应商扣费账单。服务器与请求线程已停止；约3.68GiB权重仍暂存在RAM，实例重启或清理RAM后会失效。它不等于永久权重部署。</p>',
      f'<p>清理前检查的三个旧压缩包均有当前目录未保留的内容，因此未删除。另按既有授权清理{cleanup["planned_files"]}个均有对应.py源码的可再生字节码缓存；逐项源码前后哈希一致，实际空闲空间增量约{cleanup["measured_free_byte_delta"]/2**20:.2f}MiB。未删除科学结果、活动权重、原代码或正在使用的Codex程序。</p>',
      '<h2>对研究实验的具体影响</h2>',
      '<p>新描述检索不再需要用历史查询向量代答；可复用入口为scripts/autodl/native_clip_runtime.py。它应在共同生成服务启动前绑定到同一原生CLIP模块缓存。此前已封存的单资产8B分析预算没有扩展，本轮也没有自动启动新的整场或策略对照。</p>',
      '<p>仍需验证资产分析／语义适配、五阶段真实策略组合和固定原描述的最终要求。旧接口检查、旧单资产请求与本次新检索不能拼接为一次已经完成的原生整场运行。当前24个监督单元及其缺失视图、目标、权重均保持原样。训练问题仍是同源直接动作、决策辅助和自生成决策条件动作的公平比较；检索恢复属于共同工程，不承担创新主张。</p>',
      '<h2>证据与代码</h2><ul>']
    for name in ['fresh_v1/summary.json','fresh_v1/requests.json','fresh_v1/embedding_calls.json','fresh_v1/historical_parity.json','fresh_v1/native_model_binding.json','restore_v1/budget_termination.json','restore_v2/transfer.json','research_and_runtime_boundary.json','ram_weight_lifecycle.json','bytecode_cache_cleanup.json','archive_cleanup_audit.json','reading_ledger.json','COMMANDS.txt']:
        contents.append(f'<li><a href="{name}">{name}</a></li>')
    contents.append('</ul><p>源GLB和导出几何不随证据包再分发。冷加载导出在核对后删除，记录保留原输入／导出哈希、顶点面数和文件坐标系尺寸；不把这些回执冒充独立物理评测。</p>')
    (HERE/'stage_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新描述原生检索资格</title><style>body{max-width:1150px;margin:40px auto;padding:0 20px;font:17px/1.75 system-ui;color:#172330}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4dd;padding:9px;vertical-align:top}th{background:#edf3f8}a{color:#165ba8}</style><body>'+''.join(contents)+'</body></html>')
    (HERE/'scripts').mkdir(exist_ok=True)
    for name in ['restore_original_clip_in_ram.py','finish_single_clip_ram_range.py',
                 'native_clip_runtime.py','run_fresh_native_clip_retrieval.py']:
        shutil.copy2(ROOT/'scripts/autodl'/name,HERE/'scripts'/name)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact_manifest.json','evidence.zip']]
    dump(HERE/'artifact_manifest.json',{'files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],
        'model_weights_redistributed':False,'source_meshes_or_images_redistributed':False})
    with zipfile.ZipFile(HERE/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[HERE/'artifact_manifest.json']:z.write(p,p.relative_to(HERE))
    print(json.dumps({'report':str(HERE/'stage_report.html'),'files':len(files),'zip_bytes':(HERE/'evidence.zip').stat().st_size}))


if __name__=='__main__':main()
