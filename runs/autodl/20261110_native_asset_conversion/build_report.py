"""Assemble the actual execution/source report without rerunning services."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')


def main():
    actual = ROOT/'actual_v2'
    run = read(actual/'result.json')
    encoded = read(ROOT/'physics_input_v1/result.json')
    protocol = read(ROOT/'protocol.json')
    sources = Path(protocol['native_source'])
    original = read(actual/'input_hashes.json')
    assert all(sha(p) == h for p, h in original.items())
    assert run['server_terminal'] and run['source_and_asset_unchanged']
    assert run['status'] == 'native_asset_converted_and_physics_views_rendered'
    assert run['geometry_comparison']['max_bidirectional_vertex_distance_m'] == 0
    assert run['geometry_comparison']['original_faces'] == run['geometry_comparison']['converted_faces']
    assert encoded['model_loaded'] is False and encoded['cuda_initialized'] is False
    assert encoded['image_count'] == 4
    events = read(actual/'http_events.json')
    posts = [r for r in events if r['method'].lower() == 'post']
    assert [r['path'] for r in posts] == ['/convert_glb_to_gltf', '/render_multiview']
    assert all(r['status'] == 200 for r in posts)
    assert sha(ROOT/'physics_input_v1/native_request_local_only.json') == encoded['request_sha256']
    for row in read(ROOT/'prompt_source_receipt.json'):
        assert sha(ROOT/'source'/row['path']) == row['sha256']
    files = []
    for directory in [actual/'asset', actual/'physics_views']:
        for p in sorted(directory.rglob('*')):
            if p.is_file():
                files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    p = ROOT/'physics_input_v1/native_request_local_only.json'
    files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    dump(ROOT/'local_only_resources.json', {'files':files,
        'meaning':'Licensed derived geometry/images and inline-image request stay local; hashes and metrics may be published.'})
    # Preserve all exact native source identities, including the already recorded
    # author integration patch rather than pretending this is pristine upstream.
    m = read(sources/'manifest.json')
    dump(ROOT/'native_source_receipt.json', {
        'source_manifest':str(sources/'manifest.json'), 'source_manifest_sha256':sha(sources/'manifest.json'),
        'source_pin':protocol['scene_source_pin'], 'author_integration_patches':m['patches'],
        'verified_read_only_file_count':len(original),
        'original_hashes':'actual_v2/input_hashes.json',
        'reading_scope': {
            'blender/mesh_conversion.py':'Complete convert_glb_to_gltf_impl; full body executed by real HTTP route.',
            'blender/server_manager.py':'1-150, 403-548, 581-645: startup, readiness, conversion and render request; native full module imported, no claim all retry branches were exercised.',
            'blender/renderer.py':'317-465: setup and view construction, plus actual full native render function executed.',
            'mesh_physics_analyzer.py':'1-380: request/parse path read; only request-building AST statements executed after real rendering. No VLM/parser execution or returned estimates.',
            'prompts/data/mesh_physics/hssd.yaml':'Complete; source-verified original prompt, no template substitutions.',
            'prompts/manager.py':'Complete; actual normal PromptManager import/get_prompt executed.',
            'prompts/registry.py':'PromptEnum/Meta and MeshPhysicsPrompts read; complete module normally imported.',
            'utils/openai.py':'Complete; actual JPEG encoding helper executed.'}})
    rows = [
        ['native_GLB_to_GLTF_RPC','executed',run['conversion_seconds'],'one fixed real asset; original GLB kept'],
        ['native_four_view_RPC','executed',run['view_render_seconds'],'four 512x512 side views, original elevation20/TAA8'],
        ['native_physics_request_CPU_encoding','executed',encoded['seconds'],'1566 input tokens; no model weights or completion'],
        ['VLM_material_mass_front','not_executed','','unknown; no invented response'],
        ['canonicalization_SDF_registry','not_executed','','requires actual admitted physics response'],
        ['complete_scene_generation','not_executed','','no new room or learned benefit'],
    ]
    with (ROOT/'summary.csv').open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n');w.writerow(['component','status','seconds','scope']);w.writerows(rows)
    summary = dict(status='native_conversion_views_and_request_executed',
        previous_goal_turn='no progress toward research: connection/Git/resource status only',
        this_goal_turn='progress: collision delivery published; actual native asset conversion and four-view RPC executed; exact physics request CPU-encoded and source boundaries checked',
        asset_id='1e1c18c687ffc14eb4267863fc389ef74c2e26e3',
        conversion_seconds=run['conversion_seconds'], render_seconds=run['view_render_seconds'],
        service_body_seconds=run['seconds'], processor_body_seconds=encoded['seconds'],
        executed_body_seconds_sum=run['seconds']+encoded['seconds'],
        timing_scope='Does not include source reading, import-only failure, fetching, report construction or billing. Bodies measured separately; not end-to-end new-room latency.',
        conversion_RPCs=1, render_RPCs=1, views=4, input_tokens=encoded['input_tokens'],
        original_faces=1934, converted_faces=1934,
        original_vertices=1977, converted_vertices=1989,
        max_bidirectional_vertex_distance_m=0.0, bounds_difference_m=0.0,
        original_files_verified_read_only=len(original), server_terminal=True,
        parent_peak_rss_mib_only=run['parent_peak_rss_mib_only'], child_peak_not_measured=True,
        native_render_GPU_memory_single_observation_mib=312, pure_CPU_rendering_claim=False,
        cpu_encoding_model_loaded=False, cpu_encoding_cuda_initialized=False,
        new_model_calls=0, new_training_updates=0, new_paid_API_calls=0,
        new_complete_scene_generations=0, whole_scene_success=None,
        full_runtime_ready=False, learned_benefit_established=False,
        free_data_directory_bytes=shutil.disk_usage(ROOT).free)
    dump(ROOT/'summary.json', summary)
    # Do not silently turn an isolated service return into a full five-stage gate.
    dump(ROOT/'full_runtime_admission.json', {
        'full_stage_assets':False, 'tool_side_no_paid_api':False,
        'common_evaluation':False, 'common_action_scope':False,
        'meaning':'These broad guards remain unqualified. This run itself used no external model provider, but does not prove all native branches have local replacements.',
        'completed':'Pinned collision route; real GLB conversion and physical-view request preparation.',
        'remaining':['actual native physics analysis and validation', 'canonicalization/SDF/registry lifecycle',
                     'floor materials and all required stage assets', 'full native generation with common observations/actions/evaluation']})
    dump(ROOT/'reading_ledger.json', {
        'scope':'Targeted primary-source recheck plus actual tool source; no exhaustive novelty clearance.',
        'entries':[
            {'title':'Scene-Designer', 'url':'https://arxiv.org/html/2610.23121v1',
             'this_turn_read':'Sections3.1 and D.2-D.3 targeted re-read, especially action-conditioned reasoning, replay and perturbation construction; not all experiments or figures.',
             'finding':'Observation-thought-action supervision already exists; generated training reasons can condition on target actions. This is legal supervision, not evidence of self-generated deployment value.'},
            {'title':'SceneSmith', 'url':'https://arxiv.org/html/2602.09153v2',
             'this_turn_read':'A.1-A.4 and opening B.1 rechecked; exact native asset conversion/physics source ranges in native_source_receipt.json.',
             'finding':'Five stages, physical asset acquisition, preserved-context rollback and structured/visual observations are existing components. Reusing them qualifies the common backend, not novelty.'},
        ],
        'already_available_theory_and_controls':[
            '../20261110_stage_alignment_review/algorithm_and_controls.json',
            '../20261110_frontier_scene_evaluation/algorithm_and_theory.json',
            '../20261110_missing_decision_objective/reading_ledger.json'],
        'not_claimed':'New loss/theorem, first reasoning-action method, human preference evidence, learned complete-scene gain or CVPR-level contribution.'})
    dump(ROOT/'handoff.json', {
        'actual_completed':'Native conversion/rendering and original VLM request construction; no estimate response supplied.',
        'next_service_boundary':'Use exactly this prepared request for an explicitly bounded real local VLM qualification; validate native response, then canonicalization/SDF/registry on this same asset. No new model inference is silently included in this delivery.',
        'research_comparison':'Same-source auxiliary D supervision + direct A versus self-generated D-conditioned A, plus ordinary base/search at common cost. Reason text alone is not an independent new mechanism.',
        'full_generation_endpoint_retained':True,
        'training_status':'24 frozen units,19 CPU-encoded,5 missing original observations; no new annotations or updates. This development material cannot demonstrate held-out whole-room benefit.',
        'resource':'All launched service processes terminal. No new resource opening needed for remaining source work. Do not keep a paid instance open merely for this report.'})
    table = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)
    report = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实资产转换、输入与研究边界</title>
<style>body{max-width:1060px;margin:38px auto;font:17px/1.65 system-ui;color:#1b2636}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:8px}code{font-size:.9em}small{color:#526376}</style>
<h1>资产链路实际推进；仍未证明新的三维决策机制</h1>
<p>一次真实 Blender GLB→GLTF 转换和一次四视角渲染已完成；原生物理分析消息已使用真实输出完成本地编码。这里没有新生成房间，没有模型回答、训练或付费 API。旧连接检查轮仅为状态复述，本轮才新增执行证据。前一项碰撞服务证据已发布为 Git 8364d64。</p>
<h2>本轮实际执行</h2><table><tr><th>分支</th><th>状态</th><th>秒</th><th>范围</th></tr>'''+table+'''</table>
<p>服务运行61.280秒，编码10.574秒，分别计时之和71.854秒。首次导入缺少 colorama，在创建 Blender 进程之前失败；完整失败日志保留。仅补上已有依赖目录后进行一次有界恢复，没有下载安装、换资产、改参数或质量挑选。原生服务器正常清理结束（SIGTERM，-15）。</p>
<p>原网格1977个顶点，转换后1989个顶点；面数均为1934。世界坐标中的双向顶点最近距离最大值和边界差均为0。此证据支持该转换的几何位置完整性，不代表材质外观完全一致或正面方向已确定。外置二进制和两张纹理均存在，原始 GLB 未删除。</p>
<p>原配置为四个512×512侧视图、20度仰角、TAA8。<code>CUDA_VISIBLE_DEVICES=''</code>及<code>gpu_id=None</code>不能证明渲染不用显卡；Eevee 期间实际观测到312 MiB显存，结束后释放。父进程峰值约552 MiB，未测子进程峰值，不能宣称全流程只需该内存。</p>
<h2>消息真正到了哪一步</h2>
<p>原生 PromptManager、HSSD提示、图像编码函数和请求构造语句实际执行。四图消息经已有本地处理器编码为1566个输入token，没有载入权重或初始化CUDA。原生编码器先把PNG转成JPEG字节，而请求仍标注PNG类型；记录保留这一不一致，不把它悄悄修成“原样复现”。本地PIL可解码这些字节，不代表远端接口一定接受。处理器将每图预算设为65536像素，不能称为官方API的相同视觉预处理。</p>
<p>原生要求JSON响应不等于本地解码器已实施JSON约束；质量、材质、正面、标准化、SDF及注册仍未执行。不得把提示中的示例质量填入结果或把模型估计当测量真值。</p>
<h2>当前算法能成立的主张及不能成立的主张</h2>
<p><a href="https://arxiv.org/html/2610.23121v1#S3.SS1">Scene-Designer §3.1</a>已经采用观察—思考—动作训练，并使用目标动作构造训练推理。因此“先输出决策文字，再输出工具动作”不是这里新发明的算法。<a href="https://arxiv.org/html/2602.09153v2#A1.SS4">SceneSmith A.4</a>已有保留上下文的回滚；本轮复用其资产链路属于共同后端工程。</p>
<p>目前应检验的是同一监督下的部署差异：辅助分支学 D(X) 与 A(X)，部署直接产生动作；条件分支学 D(X) 与 A(X,D)，部署必须先从实际 X 自行生成 D，再生成 A。两组保留原有token权重、相同资产和动作权限；缺少D的14条动作样本不能删除或伪造D。普通搜索还应能使用节省的推理预算。相同目标token不意味着相同训练计算量，已写好的局部损失函数也不代表完整Trainer已可靠运行。</p>
<p>理论能支持的仅是边界：给定模型参数，若D只由X和独立随机数产生，则它不能带来X之外的目标信息；它仍可能改变有限计算下的策略。使用参考D的动作准确率，不能推导自产D时的整场成功率。现有数据不足以估计这种偏差，更不能给出性能保证。多种布局都可能正确，应检验实际场景要求而非唯一工具字符串。</p>
<p>最新真实基础模型诊断出现了一次新增相交及两次原地重试；普通几何检查已有直接解释。它不是加入决策文字、记忆或新训练的正面证据。现有24条开发单元只有19条完成输入编码，并非独立完整场景训练/测试集。当前不宣称已找到CVPR主创新，也不把一次基础模型失败推广为整个路线无效。</p>
<h2>下一项执行边界</h2>
<p>固定本资产和消息，接下来只需一次明确限额的真实本地物理分析资格验证，再依据实际答案运行原生标准化、SDF和注册；不能用假答案跨过依赖。随后才有条件测试共同完整生成。完整资产、所有阶段、共同评价和动作范围四项总准入均保持未通过；不会为跑出房间而取消表面物体、物理元数据或悄悄切换付费后端。</p>
<p>本轮启动的服务均已结束。后续源码审查无需继续占用已开启的整套计费实例。</p>
<h2>可核查证据</h2><ul>
<li><a href="summary.json">实际汇总</a>、<a href="summary.csv">逐分支表</a></li>
<li><a href="actual_v2/result.json">服务结果</a>、<a href="actual_v2/http_events.json">RPC记录</a>、<a href="actual_v2/input_hashes.json">旧文件哈希</a></li>
<li><a href="physics_input_v1/result.json">真实请求处理器记录</a>、<a href="physics_input_v1/native_image_encoding.json">JPEG与PNG类型证据</a></li>
<li><a href="native_source_receipt.json">原生源码及作者补丁</a>、<a href="reading_ledger.json">阅读范围</a>、<a href="full_runtime_admission.json">未完成准入</a></li>
<li><a href="local_only_resources.json">本地几何/图像清单</a>、<a href="evidence.zip">不含资产的证据包</a></li></ul>
<small>范围：工具链执行、输入完整性和研究识别条件；不是完整生成成绩、受训策略成绩或专业偏好研究。</small></html>'''
    (ROOT/'stage_report.html').write_text(report)
    # Exclude self-referential inventory and licensed payloads, include all code.
    skip = {'artifact_manifest.json','evidence.zip','delivery_check.json'}
    allowed = []
    local = {str(ROOT/r['path']) for r in files}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in skip and str(p) not in local and '__pycache__' not in p.parts:
            allowed.append(p)
    allowed.append(REPO/'scripts/autodl/execute_native_asset_conversion.py')
    inventory = [{'path':str(p.relative_to(REPO)), 'bytes':p.stat().st_size, 'sha256':sha(p)} for p in allowed]
    dump(ROOT/'artifact_manifest.json', {'files':inventory, 'geometry_and_image_payloads':'local_only_resources.json'})
    with zipfile.ZipFile(ROOT/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in allowed+[ROOT/'artifact_manifest.json']:
            z.write(p, str(p.relative_to(REPO)))
    print(json.dumps({'status':summary['status'], 'metadata_files':len(inventory),
                      'archive_bytes':(ROOT/'evidence.zip').stat().st_size}))


if __name__ == '__main__':
    main()
