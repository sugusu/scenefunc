"""Build the source/multiview qualification report from actual saved evidence."""
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


stage = Path(__file__).resolve().parent


def load(name):
    return json.loads((stage / name).read_text())


def save(name, obj):
    (stage / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


actual = load('actual_v1/summary.json')
views = load('actual_v1/view_provenance.json')
camera = load('camera_candidates_v2.json')
constructor = load('nala_constructor_receipt.json')
nala = load('nala_source_manifest.json')
assert (stage / 'source_nala/LICENSE').read_text().startswith('MIT License')
for row in nala['files']:
    if 'error' not in row:
        assert hashlib.sha256((stage / 'source_nala' / row['path']).read_bytes()).hexdigest() == row['sha256']
summary = {
    **actual,
    'available_image_references': len(views),
    'distinct_available_image_hashes': len({v['sha256'] for v in views}),
    'count_correction': '95 is image references across19 units, not95 independent images;80 distinct hashes.339 is object records across15 states, not339 unique assets.',
    'source_camera_candidates': camera['image_references'],
    'distinct_camera_candidate_images': camera['distinct_image_hashes'],
    'maximum_camera_source_location_error_m': camera['maximum_source_location_error_m'],
    'independent_projection_validation': False,
    'exact_mesh_shape_supervision_extracted': False,
    'nala_constructor_status': constructor['status'],
    'nala_full_import_run': False, 'imagine3d_training_source_available_in_reviewed_commit': False,
    'new_source_models_downloaded': 0, 'goal_achieved': False,
}
save('summary.json', summary)
ledger = {
    'scope': 'Targeted representation/objective/source reading, not full-field novelty clearance',
    'entries': [
        {'title': 'Imagine3D-LLM', 'url': 'https://arxiv.org/html/2609.38177v1',
         'read': 'Sections3.1-3.3 and4.2; AppendixA, C.6, D andE. Selected setup/results text, not all tables/figures/appendices.',
         'code_status': 'cvlab-kaist/Imagine3D-LLM@5956eaef5c935e864f79fba848c878f711fea70b: fetched/read README; training and demo checklist unfilled. No implementation claimed read.',
         'finding': 'Geometry reconstruction and hidden-state scene summaries are established; comprehension results do not demonstrate complete-room generation.'},
        {'title': 'NaLA', 'url': 'https://arxiv.org/html/2606.29395v1',
         'read': 'Sections3-4.3,5.1-5.2; appendix0.D.2, selected metrics and0.E. Not every experiment/figure or appendix prompt.',
         'source_pin': nala['pin'],
         'code_read': {'src/nala/models/model7B.py': '1-203,251-358,471-585,620-645,933-978,1153-1280,1398-1562; remaining only relevant call-site searches',
                       'scripts/infer.py': 'Complete',
                       'src/nala/eval_defaults.py': 'Complete',
                       'src/nala/data/data_preprocess_new.py': '84-147,1098-1155; remaining only relevant field searches',
                       'docs/dataset.md': 'Complete', 'README.md': 'Complete',
                       'scripts/train.py': 'Only144byte entrypoint; no complete trainer read'},
         'finding': 'Point-cloud inputs and coarse/residual poses are direct scene-generation precedent; existing entrypoint and paper/source losses require version-specific qualification.'},
        {'title': 'Existing native observation source',
         'paths': ['../20261110_native_observation_export/checkpoint_renderer/tools/render_cardinal_views.py',
                   '../20261110_native_observation_export/checkpoint_renderer/tools/render_five_views.py',
                   '../20261110_native_observation_export/checkpoint_renderer/src_critic/blender_render.py'],
         'read': 'Complete cardinal/five-view workers; top-camera setup, metadata and ceiling hiding in blender_render.',
         'finding': 'Original source plus recorded sidecars can recover conditional camera matrices; cutaways and overlays need a separate image-loss contract.'},
        {'title': 'Related predictive geometry work search only',
         'urls': ['https://arxiv.org/abs/2610.28164', 'https://arxiv.org/abs/2609.13812', 'https://arxiv.org/abs/2606.04436'],
         'read': 'Search abstracts only; not method or code reading',
         'use': 'Candidate geometry/action claims still need overlap review; these results are not novelty clearance.'}],
    'not_done': ['No new model inference, rendering, training, independent projection check, NaLA benchmark or Gaussian teacher run',
                 'No demonstration of learned geometric representation, geometry certificate or independent full-room improvement']}
save('reading_ledger.json', ledger)
save('source_differences.json', {
    'nala_pin': nala['pin'],
    'constructor': constructor['actual_error'],
    'constructor_scope': 'Original ClassDef plus actual caller/default literal extraction executed; AST isolation, no full native imports or tokenizer/model load',
    'loss': {'paper': 'Position/scaling residual L1 plus periodic cosine orientation term',
             'code': 'PositionLoss slices5values:3position,1uniformscale,1rotation; rotation usesL1. forward combines4components with.25 coefficients.',
             'meaning': 'Do not label current published code as an identical loss/config reproduction or invent checkpoint compatibility.'},
    'lora': {'infer_literal_defaults': {'r': 64, 'alpha': 128},
             'add_lora_no_argument_defaults': {'r': 8, 'alpha': 32},
             'qualification': 'Source observations only; constructor blocks before load. No trained-checkpoint compatibility established.'},
    'postprocess': 'scripts/infer.py defaults --postprocess off; optional full cleanup must be reported separately',
    'source_patched': False,
    'imagine3d_pin': '5956eaef5c935e864f79fba848c878f711fea70b',
    'imagine3d_implementation_read': False})
with (stage / 'method_differences.csv').open('w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['method', 'learned_or_input_signal', 'endpoint', 'current_our_evidence', 'not_claimed'])
    writer.writerows([
        ['same-source D/A', 'decision and action text, shared actual scene', 'full generation intended', 'original objectives/source available', 'main novelty or learned whole-room gain'],
        ['Imagine3D-LLM', 'latent Gaussian reconstruction + language', '3D understanding', 'primary method read; implementation unavailable in reviewed commit', 'ported reconstruction or full generation'],
        ['NaLA', 'point-cloud geometry + coarse/residual poses', 'layout given room and assets', 'pinned source read; constructor mismatch reproduced in isolation', 'original benchmark/weights reproduced'],
        ['ordinary current-state geometry auxiliary', 'existing object IDs, positions, rotations, local bounds', 'control for action learning', '24rows retained;15states/339object records extracted;10state/view bindings', 'novel loss, expert labels, action consequences or training readiness']])
report = '''<!doctype html><meta charset="utf-8"><title>三维表示、监督材料与方法差异</title>
<style>body{max-width:1060px;margin:35px auto;padding:0 16px;font:17px/1.75 sans-serif}pre{white-space:pre-wrap;background:#f3f5f8;padding:15px}aside{background:#fff3d6;padding:15px}</style>
<h1>把“有三维表示”落实到可用监督与完整生成目标</h1>
<p>本轮补读直接近邻，并实际完成当前材料的几何监督提取与相机矩阵候选恢复。没有重新跑289/404坐标修复，没有生成模型调用、付费API或训练。工程与数据准备继续推进，但尚无独立完整场景学习收益。</p>
<h2>近邻改变了什么判断</h2>
<p><a href="https://arxiv.org/html/2609.38177v1">Imagine3D-LLM</a>已把几何重建监督接到MLLM隐藏状态；<a href="https://arxiv.org/html/2606.29395v1">NaLA</a>已直接接收三维资产并学习布局输出。因而，仅把决策文字换成三维token，也不足以构成主创新。前者主要评价理解，后者主要评价给定资产的布局；都不能不加限定地替代我们完整文字驱动场景生成的终点。</p>
<p>已核查官方源码状态：Imagine3D固定提交仍只有介绍和未完成的代码清单，不能说已复现实现。NaLA有公开源码；当前入口配置调用的lora_r参数与定义不匹配，本轮隔离执行该原配置类与调用字面量，确实得到TypeError。它不是整套上游运行，也不能推翻论文成绩。损失和默认参数的版本差异已逐项记录，未修改上游源文件。</p>
<h2>实际做完的数据工作</h2>
<pre>SUMMARY</pre>
<p>19个可用单元共有95条图像引用、80个不同图像哈希，并非95张独立样本。10条引用有标注改写的哈希证据，40条有明确切墙记录。原旁文件未直接保存完整K/R/t，本轮继续从源码和保存位置恢复50条矩阵候选，对应45张不同图像。</p>
<p>40条侧视记录与源码计算的位置最大误差约0.7微米。这验证了来源计算的一致性，未独立验证Blender投影、相机默认像素比例或像素中心约定。矩阵候选连同假设、切墙对象ID和原图哈希均已保存。第一次元数据解析因最早导出缺字段而停止；原错误和源脚本保留，第二版使用已发布Git对象补充该哈希证据，没有重跑物理或图像。</p>
<p>同时提取15个已存状态、339条对象记录：包括家具与建筑对象，以及重复状态中的对象。它们不是339个新资产，也不是339个专业设计标签。这里只提取姿态和包围盒，没有生成点云、精细表面或内部支撑真值。全部24个源单元保留，9个未在这份状态恢复账本中出现的单元明确记缺，另5个仍缺原观察；10个具有当前状态与图像的明确绑定。原动作、决策、权重及图像均未修改。</p>
<h2>为什么不能直接训练重建</h2>
<p>智能体观察图包含视角相关切墙和二维标注。若开展重建监督，应使用原始RGB，并显式处理每个视角的可见结构，必要时使用遮罩。原图缺少相机矩阵字段，不代表无法从源码恢复；本轮已完成候选恢复。二维标注没有被悄悄擦除，矩阵候选也未冒充完整标定，因此当前仍未准入光度训练。</p>
<pre>可用的图像目标应形如：
I_hat_v = Render( Visibility_v(S_hat), K_v, T_v )
L_photo = Σ_v || M_v ⊙ (I_hat_v − I_raw_v) ||
其中Visibility_v、M_v、原始RGB必须有来源；不能用带标注的图代替它们。</pre>
<p>这只是监督定义，不是新算法。当前339条记录可服务于普通的“已记录状态几何”辅助监督；不能叫未来动作效果，也不能证明正确设计。动作模型原本就能查询当前坐标和包围盒，所有对照必须获得同样的信息。</p>
<h2>几何监督如何公平接到当前训练问题</h2>
<pre>L_geo-aux = L_A(X,A) + λ_G Σ_i m_i L_geom(G_hat_i(X), G_recorded_i)
对照一：同样的几何监督，部署直接A(X)
对照二：同样的几何监督，部署自身G_hat，再A(X,G_hat)
原有辅助D/直接A与自身D后A的对照继续保留；不把新监督只给候选组。</pre>
<p>G按稳定对象ID绑定；位置以米计，旋转保留wxyz四元数。几何是当前状态，不把未来场景或正确坐标注入某一部署组。若把序列设为“图像—几何摘要—用户文字—动作”，因果注意力下摘要看不到后面的任务或未生成动作，而动作仍可直接看图像。因此它并不自动构成任务相关的未来模型或不可绕过的瓶颈。</p>
<p>若辅助几何监督、直接读状态或普通搜索已经解释收益，主张应收缩。只有在相同信息、标签、工具和总预算下，条件表示在未参与开发的房间持续改善完整要求完成，才值得扩大。几何token、语言理由、反馈与重建各自已有近邻，不能先给它们的组合命名再认定创新。</p>
<h2>理论边界：几何误差与真正约束</h2>
<p>对于同一刚性资产K，局部半径不超过r，平移为p、旋转为R、非负统一缩放为s；相对旋转角为δ。按同一本地点配对并用三角不等式，可得：</p>
<pre>d_H(p+sRK, p_hat+s_hat R_hat K)
 ≤ ||p−p_hat|| + r|s−s_hat| + 2r*s_hat*sin(δ/2).</pre>
<p>若两个物体的占据集合误差分别有可信上界ε_i、ε_j，预测间距大于ε_i+ε_j才可推出实际分离。这是普通几何界，不是新定理。我们尚未为模型取得这些上界；均值回归损失、Gaussian协方差、好看的重建图都不自动成为可信证书。更换资产还需要形状误差项，间距界也不证明支撑稳定或使用功能。</p>
<p>有限视图的光度误差亦不统一控制未观测几何。一个简单存在性例子是在无间接照明的遮挡渲染下修改所有相机都看不到的区域：图像不变，却可能改变该区域的碰撞或支撑。这不声称本轮Cycles会产生完全相同图像；它说明还需要几何假设与实际工具检查，不能从重建损失直接推出任务成功。</p>
<h2>资源与下一步</h2>
<aside>当前可执行的几何目标与相机候选已产出，仍不建议为移植大规模Gaussian教师开启更高资源或下载新模型。先把普通几何辅助监督作为控制条件，而非新主创新。给定固定资产集的布局试验可以独立准备，但必须明确是子任务，不能把缺少任意描述检索能力藏起来并称为完整生成。</aside>
<p>目前没有确认的CVPR主创新；这轮也没有把研究停掉。完整生成、独立收益、几何目标的有效性与资源峰值均仍需实测。准备工作不需要新增GPU运行。</p>
<p><a href="actual_v1/unit_admission.csv">逐单元关系与监督准入表</a> · <a href="actual_v1/current_state_geometry.jsonl">实际状态几何目标</a> · <a href="camera_candidates_v2.json">相机候选及来源</a> · <a href="theory_and_controls.json">算法与理论条件</a> · <a href="reading_ledger.json">阅读范围</a> · <a href="source_differences.json">源码差异</a></p>'''
(stage / 'stage_report.html').write_text(report.replace('SUMMARY', html.escape(json.dumps(summary, ensure_ascii=False, indent=2))))
save('handoff.json', {
    'status': 'Geometry baseline material prepared; full research goal remains active',
    'actual_cpu_artifacts': ['24retained-unit current-state target file', '95image-reference provenance table', '50source-derived camera candidates', 'pinned native3D model source/config qualification'],
    'do_not_launch': ['new model downloads', 'Gaussian teacher training', 'paid API', 'automatic full-scene batch'],
    'required_before_representation_claim': ['same information and geometry supervision in controls', 'own predicted rather than teacher geometry at deployment', 'independent full-room outcome benefit', 'explicit unknown and unsupported checks'],
    'not_a_block_for_all_work': 'Missing reconstruction calibration does not prohibit paper/source review, ordinary fixed-asset layout preparation, or native tool completion.',
    'training_admitted': False, 'goal_achieved': False})
files = sorted(p for p in stage.rglob('*') if p.is_file() and not p.is_symlink()
               and 'source_local_only' not in p.relative_to(stage).parts
               and '__pycache__' not in p.relative_to(stage).parts
               and p.name not in ('artifact_manifest.json', 'evidence.zip'))
artifacts = [{'path': str(p.relative_to(stage)), 'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
save('artifact_manifest.json', {'files': artifacts,
                              'source_nala_license': 'MIT preserved',
                              'not_distributed': 'Imagine3D local README; no assets/images/model weights'})
with zipfile.ZipFile(stage / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(stage))
    z.write(stage / 'artifact_manifest.json', 'artifact_manifest.json')
print(json.dumps({'report': str(stage / 'stage_report.html'),
                  'packaged_files': len(files), 'original_supervision_modified': False,
                  'new_model_calls': 0, 'training_updates': 0}))
