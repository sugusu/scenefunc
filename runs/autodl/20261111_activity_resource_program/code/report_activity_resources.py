"""Report measured activity programs, closed local-model trials and boundaries."""
import argparse
import csv
import hashlib
import html
import importlib.util
import itertools
import json
from pathlib import Path
import random
import zipfile

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    compiled=read(root/'compiled_programs.json');model=read(root/'model_probe.json');execution=read(root/'execution_result.json');protocol=read(root/'protocol.json')
    assert model['status']=='completed'
    rows=compiled['rows'];byid={x['id']:x for x in rows}
    # Independent brute-force reference validates the nontrivial four-extrema
    # reduction; no new scene or policy result is counted from these checks.
    from shapely.geometry import box,shape
    spec=importlib.util.spec_from_file_location('activity_compiler',Path(__file__).with_name('activity_resource_program.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=random.Random(20261111);checks=[]
    for seed in range(80):
        pools=[]
        for k in range(3):
            pool=[]
            for _ in range(rng.randint(0,8)):
                x,y=rng.uniform(-1,1),rng.uniform(-1,1);w,h=rng.uniform(.1,.8),rng.uniform(.1,.8)
                pool.append(dict(polygon=box(x,y,x+w,y+h)))
            pools.append(pool)
        brute=any(all(a['polygon'].intersection(b['polygon']).area==0 for a,b in itertools.combinations(t,2)) for t in itertools.product(*pools))
        fast=mod.three_rectangles(*pools)['status']=='witness';assert fast==brute
        checks.append(dict(index=seed,finite_pool_sizes=list(map(len,pools)),brute_force=brute,extrema=fast))
    dump(root/'finite_search_reference_check.json',dict(cases=checks,all_match=True,purpose='Algorithm correctness only; not80 scene-generation examples'))
    supervision=[]
    summary=[]
    for row in rows:
        minimal=0 if row['shared']['status']=='witness' else 2
        for plan in ['shared','independent']:
            feasible=row['shared']['status']=='witness' if plan=='shared' else row['independent_top_capacity_verified']
            count=0 if plan=='shared' else 2
            supervision.append(dict(task=row['id'],kind=row['kind'],plan=plan,
                geometry_feasibility_label=feasible,explicit_objective_minimality_label=feasible and count==minimal,
                subjective_human_preference_label=None,raw_purpose_semantic_correctness_label=None,
                novelty_label=None,new_objects=count,source_room_count=1,held_out=False,
                teacher='Ordinary area certificate or actual triangle-surface packing witness',
                certificate_path='compiled_programs.json',original_evaluator_score_not_replaced=True,
                training_used=False,eligible_as_claim_of_generalized_function_learning=False))
        summary.append(dict(task=row['id'],policy='ordinary_compiler',turn=0,plan='shared' if minimal==0 else 'independent',
            feasible=True,minimum_within_two_skills=True,new_objects=minimal,
            scope='Synthetic planar resource contract on one exposed complete room'))
    for r in model['attempts']:
        f=r['feedback'];summary.append(dict(task=r['task'],policy='local_SFT_custom_allocation',turn=r['turn'],plan=f.get('plan'),
            feasible=f['valid'],minimum_within_two_skills=f['minimal'],new_objects=f.get('new_furniture'),
            scope='Initial independent decision' if r['turn']==0 else 'Oracle-minimum-assisted feedback; not independent repair evidence'))
    dump(root/'supervision_semantics.json',dict(records=supervision,training_updates=0,
        loss_scope='Optional ordinary supervised component labels, not a new loss or validated learning result',
        distinctions=['Hard feasibility is not subjective preference','Both feasible plans may differ under an explicit cost objective','A witness is not an ergonomic or original-user acceptance label','Synthetic cases from one room must stay in one development group']))
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');writer.writeheader();writer.writerows(summary)
    protected=read(Path('runs/autodl/20261111_feedback_information_control/source_before_after.json'))['original_inputs']
    checks=[dict(path=x['path'],expected=x['before'],actual=sha(x['path'])) for x in protected]
    assert all(x['expected']==x['actual'] for x in checks)
    source=[dict(path=k,expected=v,actual=sha(k)) for k,v in protocol['source_inputs'].items()]
    assert all(x['expected']==x['actual'] for x in source)
    dump(root/'source_before_after.json',dict(protected_count=len(checks),all_protected_unchanged=True,
         protected_checks=checks,current_protocol_inputs=source,old_geometry_or_thresholds_changed=False))
    literature=[
        dict(title='Activity-centric Scene Synthesis for Functional3D Scene Modeling',year=2015,
             primary_url='https://graphics.stanford.edu/~mdfisher/actSynth.html',
             paper_url='https://graphics.stanford.edu/~mdfisher/papers/actSynth.pdf',
             code_revision='c1783688dee3db1b04b0d15401d64937d17d5db5',
             actual_reading='Project abstract; targeted paper/supplement method reading from preceding turn; this turn complete agent.h, scene.h, modelInstance.h, activityDatabase.cpp and synthesizerGeneral.cpp lines200-640. Not all888lines, dependencies, or code execution.',
             establishes='Activity/object bindings and learned placement interactions are prior. Source has an explicit score bonus for shared objects.',
             source_locations=['scene.h:36-53','agent.h:63-75','activityDatabase.cpp:47-62','synthesizerGeneral.cpp:329-336','synthesizerGeneral.cpp:450-466'],
             limitations='Shared-object score code alone does not establish that every candidate path realizes multi-agent sharing. No claim of prior temporal-persistence parity.'),
        dict(title='ScIntBench: Physical Interaction Plausibility in3D Scene Generation',version='2610.37326v1,2026-10-27',
             primary_url='https://arxiv.org/html/2610.37326v1',
             actual_reading='This turn methods3.1-3.4, AppendixB.2,B.3,C.1-C.7,D.1-D.3. No official system/simulator or original scenes run.',
             establishes='Prescribed native-capability manipulation evaluation with reference-controller calibration; missing capabilities remain missing.',
             difference='Current code measures planar resource reservation and reconstructs unchanged scene states. It does not validate articulation, dynamic physics or human use.'),
        dict(title='Scene-Designer native Actor/tool branch',primary_url='Local pinned upstream source',
             source='native_execution_inventory_20261104/source/Designer/agent_rollout.py',
             actual_reading='Parser, Task/Memory builder, prompt rendering; furniture tools and local catalog session read. AST copied builder/parser without importing gateway.',
             establishes='Native Actor has its own generation/manipulation tools. Our allocation tool is custom, not that API.',
             objective_boundary='Existing joint-SFT checkpoint only. No v10Critic adapter, Scene-GRPO training or native full Actor reproduction.'),
        dict(title='Previously reviewed Function2Scene, conditional-preference RPR, MUSE, SceneSmith',
             actual_reading='Prior stages only; not newly reread in full this turn',
             establishes='No claim that activity reasoning, conditional preferences, logs or feedback are first introduced here.',
             unknown='Original full APIs/data and independent preference labels remain outside current evidence')]
    dump(root/'reading_ledger.json',dict(entries=literature,full_field_novelty_established=False))
    theory='''Measured algorithm and theory,2026-11-11

Scope. One exposed, research-transferred Milan room, not a new user study or held-out house. Every original30 native records remains unchanged.30 includes13 furniture plus17 architectural records;32 includes two added furniture instances. Physical paper/plate meshes are not created. Reservations are declared planar proxies. Work has one person; meals have the resident and guest. Work materials occupy a region in BOTH modes. The person does not work and eat simultaneously. This distinction is essential: persistent resources do not imply an additional concurrent person or seat.

Program. Compile explicit activity modes M, object bindings b, and reservation activation A(r,m). A seat may be reused by the same resident across disjoint modes. For a surface t and each mode m, every active reservation P_r lies within the measured top polygon T_t, and any two reservations on the same surface have disjoint interiors. For shared work/meal use, all three material/meal rectangles coexist during the meal mode. For independent use, only the two meal rectangles share the original table; work lies on an added copy. The program is manually typed from explicit numeric task conditions; automatic raw-brief interpretation is not solved.

Ordinary search. Extract upward real mesh triangles in the frozen3cm top band, then form their planar union. Enumerate0/90degree rectangles on the existing1cm lattice with1micrometre inward safety margin. For a fixed work rectangle, first filter meal candidates disjoint from it. For each north rectangle [a,b,c,d], a south rectangle [e,f,g,h] is disjoint iff g<=a OR e>=c OR h<=b OR f>=d. Thus the minimum south right edge, maximum south left edge, minimum south top edge and maximum south bottom edge suffice to decide existence. Checking these four extrema for each north candidate is complete within the finite pool. Complexity is O(W(N+S)), compared with O(WNS) enumeration, ignoring fixed-shape polygon output validation. This is ordinary separating-axis geometry, not a new algorithm. Eighty independent small-pool comparisons to brute force validate this implementation; they are not scene results.

Certificate. If active planar regions are interior-disjoint subsets of T, sum area(P_r)<=area(T). The expansive contract requires0.7150000000000001m2, versus0.7036511243810306m2 measured top area. Hence sharing cannot satisfy that declared contract under any rectangle rotations. This does not prove that arbitrary real dining/work practices are impossible. A failed finite-grid search would mean not_found, not continuous infeasibility. Successful shared cases provide actual polygon witnesses, not conclusions based on AABB or area alone.

Execution. Bind witnesses to native instance identities and actual table transforms. Independently check transformed coverage against the native selected table mesh. The1e-8m numerical readback margin is only an independent transformation-validation tolerance, after the original1e-6m search inset; it does not relax any search acceptance or old score. Reconstruct both source RoomScene states through the actual native importer. Independent placement is the already completed ordinary solution, not a second new generation/search. Preserve original poses, dimensions and IDs. The old frozen evaluator remains unchanged and excludes shared work from its dedicated-workspace classification; these new resource labels do not overwrite it.

Learning target. Feasible(S,u) is a constraint label under a declared interpretation. Among feasible plans, minimal additions is a separate explicit benchmark objective J. Neither is an independently annotated human preference. A basic supervised baseline may factor p(binding,activation|brief,state) and train ordinary conditional cross-entropy; geometry labels may calibrate a feasibility head. Six closed two-skill records here are development diagnostics, not a training corpus, new loss or evidence for generalization. Do not train a universal Critic to call the feasible independent compact plan physically wrong simply because it costs two extra objects. Do not label expansive sharing as feasible because it uses fewer objects.

Empirical model boundary. The public joint-SFT model sees original five native scene views, complete state and exact explicit constraints through the upstream Task/Memory builder. Tool allocation schema is custom and limited to two preverified skills. This cannot support a claim about the author's complete Actor API, arbitrary furniture generation, policy replanning, independent human preference or a new Agent method. First-turn choices and oracle-minimum-assisted second-turn choices are separate. The second-turn memory explicitly reveals the optimal new-object count; it is NOT a fair measurement of unsupervised discovery from geometric feedback. All three actual forwards selected the independent skill. No retries or training were run.

Research decision. Keep the executable mode/resource contract as a strong ordinary baseline and source of checkable component labels. Do not elevate temporal labels, shared surfaces or a new prompt to novelty. A paper-worthy next method must automatically obtain a correct, nonoverconstrained activity program from realistic source briefs, select/construct geometry across unseen complete rooms, and beat the same program+ordinary joint optimization under equal inputs, actions and total cost. The current two-skill allocation probe cannot decide that. Raw-purpose grounding, broad construction coverage, independent evaluation and a cross-scene learned-method advantage remain unproved. Do not reopen the closed empty-house trials or repackage synthetic numeric variants as independent users/rooms.
'''
    (root/'algorithm_and_theory.txt').write_text(theory)
    with (root/'method_difference.csv').open('w') as f:
        wr=csv.writer(f,lineterminator='\n');wr.writerow(['component','implemented','ordinary_nearest_explanation','unproved'])
        wr.writerows([
            ['Activity/mode bindings','Explicit persistent material and sequential seat program','Ordinary resource scheduling; activity/object bindings already prior','Automatic interpretation of realistic briefs'],
            ['Geometry search','Actual triangle-surface finite packing plus necessary area certificate','Ordinary packing/separating-axis geometry','No new optimization algorithm claimed'],
            ['Complete scene connection','Native30/32record imports and reservation bindings','Reuse existing ordinary construction skills','No new raw-text scene generation'],
            ['Model decisions','3actual local-SFT calls; all independent bindings','Custom two-skill allocation component','Not native Actor parity or independent Agent advantage'],
            ['Supervision','6auditable feasibility/cost records, no training','Standard supervised baseline','Not human preference, new objective or generalization']])
    receipts=[read(Path(x['receipt_dir'])/'receipt.json') for x in model['attempts']]
    dump(root/'cost_ledger.json',dict(compiler_wall_seconds=compiled['wall_seconds'],compiler_peak_rss_kib=compiled['peak_rss_kib'],
        model_process_wall_seconds=model['elapsed_seconds'],model_peak_rss_kib=model['peak_rss_kib'],
        local_forwards=model['model_attempts'],input_tokens=sum(x['input_tokens'] for x in receipts),
        output_tokens=sum(x['output_tokens'] for x in receipts),max_cuda_allocated_bytes=max(x['peak_cuda_allocated_bytes'] for x in receipts),
        native_execution_wall_seconds=execution['wall_seconds'],native_execution_peak_rss_kib=execution['peak_rss_kib'],
        paid_API_calls=0,training_updates=0,new_model_downloads=0,
        total_instance_billing_unknown=True,warning='GPU/CPU/RAM bundled instance billing not measured. Existing model load retained libgomp OMP_NUM_THREADS warning; no rerun; no efficiency superiority claim.'))
    # Publication-style static measurement figure; diagnostic footprint proxies,
    # not generative imagery or visual proof of ergonomic validity.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4));surface=read(root/'measured_table_surface.json');top=shape(surface['polygon'])
    for ax,row in zip(axes,rows):
        ax.fill(*top.exterior.xy,color='#ededed',edgecolor='black',lw=.6)
        if row['shared']['status']=='witness':
            for (name,item),color in zip(row['shared']['witnesses'].items(),['#458bc1','#ea9f42','#63a56b']):
                q=shape(item['polygon']);ax.fill(*q.exterior.xy,color=color,alpha=.75,label=name)
            ax.set_title(row['id']+'\nconstructive shared witness',fontsize=9)
        else:
            ax.text(.5,.5,f"Required: {row['required_area_m2']:.3f} m²\nAvailable: {top.area:.6f} m²\nShared reservation infeasible\nunder declared area contract",transform=ax.transAxes,ha='center',va='center',fontsize=9)
            ax.set_title(row['id']+'\narea certificate',fontsize=9)
        ax.set_aspect('equal');ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
    fig.suptitle('Real tabletop geometry; declared planar occupancy proxies, not human-use validation',fontsize=10);fig.tight_layout();fig.savefig(root/'allocation_witnesses.png',dpi=150);plt.close(fig)
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in summary[0])+'</tr>' for r in summary)
    first=[x for x in model['attempts'] if x['turn']==0]
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整场景中的活动与资源分配：实际运行</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1100px;margin:40px auto;color:#182630}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #bbb;padding:7px}}aside{{background:#edf3f6;padding:15px}}img{{max-width:100%}}pre{{white-space:pre-wrap;font-size:14px}}</style>
<h1>活动、持续占用与完整场景：2026-11-11实际运行</h1>
<aside>成果是可执行的普通强对照与真实模型记录；尚未形成新颖Agent算法或CVPR效果证据。一个已暴露开发房间、三个数值条件，不是三个独立房间。</aside>
<p>真实桌面测量面积为0.7036511243810306平方米。小资料与两餐位可共存；大资料条件所需0.715平方米，超过面积上界。办公人与用餐人可以跨时段复用坐席，资料则在用餐时仍占表面。独立桌椅是充分解，不能当唯一正确答案。</p>
<p>CPU编译实际耗时{compiled['wall_seconds']:.3f}秒。本地8B模型完成{model['model_attempts']}次推理，进程耗时{model['elapsed_seconds']:.3f}秒；两项首轮决定都选择独立桌椅，小资料多添了两件。告知最少新增数0后仍选独立桌椅。这是自定义两工具组件，不能据此评价作者完整Actor或所有VLM。原生完整状态导入与区域绑定耗时{execution['wall_seconds']:.3f}秒，6项程序记录包含重复场景/策略，不是6次新生成。</p>
<img src="allocation_witnesses.png" alt="真实桌面及可核验平面区域"><table><tr>{''.join('<th>'+html.escape(k)+'</th>' for k in summary[0])}</tr>{tr}</table>
<h2>公平性和监督含义</h2><p>普通求解器和模型使用同一两工具施工库、占用条件与资产。第二次模型输入包含最优新增数量，属于oracle辅助反馈，不能与首次独立推断混合。输出模型是公开联合SFT检查点的FP16/V100适配；没有v10适配器、RL或原生完整Actor复现。原生消息构造和解析来自固定作者源码，allocate_resources是本地新接口。所有物体、姿态和尺寸保留，没生成纸张/餐盘几何，不声称物理稳定、人机工学或完整原任务成功。</p>
<p>6条监督记录分开标注几何可行性与显式少添家具目标，人的主观偏好全部留空。没有训练。旧冻结评分及阈值不变；原评分排除共享工作面，不用本轮占用结果替换旧分数。</p>
<h2>最近邻与阅读边界</h2><p><a href="https://graphics.stanford.edu/~mdfisher/actSynth.html">Fisher等2015</a>已有活动—物体绑定；本轮核对部分官方C++分支，不宣称跑过旧系统。<a href="https://arxiv.org/html/2610.37326v1">ScIntBench</a>采用主动交互评测；本轮仅定向读方法和附录，没有运行其模拟器。当前静态占用证明不能代替那些动态指标。</p>
<h2>算法、理论、源码和真实结果</h2><ul><li><a href="algorithm_and_theory.txt">完整算法、复杂度、面积证明与研究边界</a></li><li><a href="compiled_programs.json">实际占用证据</a> / <a href="model_probe.json">全部模型决定</a> / <a href="execution_result.json">原生执行</a></li><li><a href="summary.csv">逐项表</a> / <a href="supervision_semantics.json">训练目标标注含义</a> / <a href="reading_ledger.json">实际阅读记录</a></li><li><a href="source_before_after.json">{len(checks)}项受保护源文件及本轮输入哈希</a> / <a href="cost_ledger.json">耗时与成本</a> / <a href="evidence.zip">证据包</a></li></ul>
<p>后续必须验证现实描述到活动程序的解释、跨房间施工和强普通优化对照。若只是给出这个正确程序，普通工具已经会做，本轮不能据此宣称需要新Critic训练、偏好反转算法或长期记忆。</p></html>'''
    (root/'stage_report.html').write_text(content)
    # Retain exact code used for execution. No pretrained weights or licensed
    # large mesh files are copied into this small evidence bundle.
    code=root/'code';code.mkdir(exist_ok=True)
    for name in ['activity_resource_program.py','activity_resource_model.py','execute_activity_program.py','report_activity_resources.py']:
        (code/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    items=[p for p in root.rglob('*') if p.is_file() and p.name not in ['evidence.zip','delivery_manifest.json','exclusive_model.lock']]
    manifest=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(items)]
    dump(root/'delivery_manifest.json',dict(files=manifest,file_count=len(manifest),no_weights_or_mesh_copies=True,
         borrowed_scene_evidence='../20261111_milan_completion',full_scene_generation_success_not_claimed=True))
    with zipfile.ZipFile(root/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*sorted(items),root/'delivery_manifest.json']:z.write(p,p.relative_to(root))
    print(json.dumps(dict(report=str(root/'stage_report.html'),files=len(manifest),protected_sources=len(checks),
                         evidence_bytes=(root/'evidence.zip').stat().st_size)))

if __name__=='__main__':main()
