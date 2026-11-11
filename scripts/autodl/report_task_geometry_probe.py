"""Package same-task native-world probes, failed attempts and training limits."""
import ast
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import zipfile

os.environ.setdefault('MPLCONFIGDIR','/dev/shm/same_task_probe_matplotlib_20261111')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
from shapely.geometry import shape,Polygon
from shapely.affinity import translate

plt.rcParams['svg.hashsalt']='same-task-geometry-v1'

R=Path(__file__).resolve().parents[2]
W=R.parent.parent
D=R/'runs/autodl/20261111_same_task_geometry'
SCRATCH=Path('/dev/shm/same_task_geometry_20261111')


def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    final=read(D/'final_result.json');baseline=read(D/'ordinary_baselines.json');protocol=read(D/'protocol.json')
    assert final['status']=='same_task_world_probe_certified_after_explicit_numeric_diagnosis'
    assert [x['correct_shared_surface_feasibility'] for x in final['rows']]==[True,False]
    sys_paths=R/'runs/autodl/20261110_public_function_search/original_evaluator'
    import sys
    sys.path.insert(0,str(sys_paths))
    from native_project import validate_bundle
    worlds=[];resources=[]
    for wid in ['original_table','smaller_table']:
        path=SCRATCH/wid;m,h,rt,t=validate_bundle(path)
        assert hashlib.sha256(t['design_request'].encode()).hexdigest()==protocol['task_text_sha256']
        rid=m['selected_room_id'];new_names=[n for n in m['files'] if 'research_dining_table_xy065' in n]
        worlds.append(dict(id=wid,manifest=m,house_state=h,runtime=rt,catalog=read(path/'catalog.json'),task=t,
            new_resources=[dict(path=n,bytes=(path/n).stat().st_size,sha256=sha(path/n)) for n in new_names]))
        if wid=='smaller_table':resources=[(path/n,n) for n in new_names]
    dump(D/'complete_world_metadata.json',worlds)
    # Licensed derived mesh stays on the same data disk, outside published Git.
    target=W/'same_task_geometry_artifacts_20261111'
    needed=sum(p.stat().st_size for p,_ in resources)
    assert shutil.disk_usage(R).free>=5*1024**3+needed+2*1024**2
    for src,name in resources:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(src,dest)
        assert sha(dest)==sha(src)
    dump(D/'persisted_resource_manifest.json',dict(directory=str(target),
        total_bytes=needed,resources=[dict(relative=n,sha256=sha(p),bytes=p.stat().st_size) for p,n in resources],
        storage='Data disk only; no hardlinks and no full dataset/model download',
        redistribution='Geometry excluded from Git/evidence ZIP; authorized source HSSD assets are required to regenerate.'))
    old=read(R/'runs/autodl/20261111_feedback_information_control/source_before_after.json')
    protected=[]
    for x in old['original_inputs']:
        assert sha(x['path'])==x['before'];protected.append(dict(path=x['path'],sha256=x['before']))
    source_manifest,_,_,_=validate_bundle(Path('/dev/shm/milan_completion_20261111/bundle'))
    assert sha('/dev/shm/milan_completion_20261111/bundle/manifest.json')==protocol['source_manifest_sha256']
    dump(D/'protected_inputs_after.json',dict(all_unchanged=True,protected_count=len(protected),files=protected,
        original_bundle_allowlist_verified=len(source_manifest['files']),source_manifest_sha256=protocol['source_manifest_sha256']))
    shutil.copyfile('/dev/shm/same_task_geometry_20261111.log',D/'initial_attempt_stdout.txt')
    shutil.copyfile('/dev/shm/same_task_geometry_certificate_20261111.log',D/'certificate_attempt_stdout.txt')
    dump(D/'attempt_lineage.json',dict(
        attempt0=dict(process_terminal=True,exit_code=1,result_file_status_stale='running',
            reason='Individual-mode positive qualification failed after floating polygon union',
            failed_log_sha256=sha(D/'initial_attempt_stdout.txt'),raw_result_kept='world_results.json',
            total_wall_seconds=None),
        fixed_candidate_diagnostics=dict(meal_candidate_positions=4,task_or_geometry_changed=False,
            world_and_local_coordinates_examined=True,extra_clearance_loosened=False,
            numeric_hole_area_not_used_as_removal_threshold=True),
        correction=dict(source_triangle_coverage='Exact dyadic rational polygon subtraction',
            genuine_tiny_gap_test='Uncovered as required; duplicate triangles do not fill missing regions',
            certificate_elapsed_seconds=final['wall_seconds'],
            result='Same fixed worlds, parameters and1um inset; qualifying existing witnesses after covered-hole proof'),
        source_version=dict(probe_driver='Unchanged after original failed attempt',
            finalizer='Core executed, then fixed-candidate derivation/fallback added for standalone reproduction. No task, predicate, certificate rule or metric changed; earlier driver bytes not pre-hashed.',
            exact_predicate='Unchanged after certificate execution'),
        cost_scope='Initial attempt and manual diagnoses were not fully timed;12.365s is certificate phase only. No efficiency superiority claimed.'))

    # Standalone scientific figure: genuine surface/footprint evidence, not a room render.
    p0=shape(read(D/'original_table_surface.json')['polygon'])
    p0=translate(p0,xoff=.5,yoff=.6)
    g=shape(read(D/'local_frame_diagnostic.json')['canonical_local_surface'])
    proofs=final['rows'][1]['numerical_hole_proofs']
    assert len(proofs)==len(g.interiors) and all(x['removed_from_numeric_region'] for x in proofs)
    p1=Polygon(g.exterior.coords)
    pos=final['rows'][0]['certificate']['regions']
    modes=final['rows'][1]['independent_modes_evidence'];meal=modes['fixed_meal_candidates'][modes['first_accepted_candidate_index']]
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
    def draw(ax,p,title,patches):
        ax.add_patch(Patch(list(p.exterior.coords),facecolor='#eeeeee',edgecolor='#222222',linewidth=1.2))
        for poly,col,label in patches:
            ax.add_patch(Patch(list(poly.exterior.coords),facecolor=col,edgecolor=col,alpha=.5,label=label))
        ax.set_xlim(-.52,.52);ax.set_ylim(-.52,.52);ax.set_aspect('equal');ax.set_title(title,fontsize=11)
        ax.set_xlabel('Table-local x (m)');ax.set_ylabel('Table-local y (m)');ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='lower left')
    colors={'persistent_work':'#e69f00','meal_north':'#0072b2','meal_south':'#009e73'}
    display={'persistent_work':'work materials','meal_north':'meal region 1','meal_south':'meal region 2'}
    draw(axes[0],p0,'Original: joint reservation witness',[(shape(x['local_polygon']),colors[x['region']],display[x['region']]) for x in pos])
    draw(axes[1],p1,'Smaller: work mode alone',[(shape(modes['work_polygon_local']),'#e69f00','work')])
    draw(axes[2],p1,'Smaller: meal mode alone',[(shape(p),c,n) for p,c,n in zip(meal['polygons_local'],['#0072b2','#009e73'],['meal region 1','meal region 2'])])
    fig.suptitle('Same task and footprints; each separate mode fits, persistent joint use exceeds area bound',fontsize=11)
    fig.savefig(D/'surface_witnesses.svg',metadata={'Date':None})
    svg=D/'surface_witnesses.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(D/'surface_witnesses.png',dpi=140);plt.close(fig)
    with (D/'summary.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['world','same_task_sha256','native_objects','shared_proxy_feasible','individual_modes_feasible','full_scene_success'])
        for x in final['rows']:w.writerow([x['world'],x['task_text_sha256'],x['objects'],x['correct_shared_surface_feasibility'],x['independent_modes_feasible'],'unknown'])
    with (D/'method_comparison.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['method','actual_result','scope'])
        w.writerow(['Ordinary labelled consistency filtering','Required-persistence hypothesis retained','Authored finite class, no trained policy'])
        for name,v in baseline['retained'].items():w.writerow([name,';'.join(v),'Original author vote implementation; uniform authored hypotheses, not sampled model outputs'])
        w.writerow(['Ordinary geometry compiler','Positive witness in original; negative area certificate in smaller','Known explicit contract; no learned interpretation'])
        w.writerow(['Proposed learned mechanism','Not run or established','No advantage over ordinary methods shown'])

    book=Path('/root/autodl-tmp/research_agent3d_20261013/book_pages.json')
    pages=read(book)
    ledger=[dict(title='AI-Agents-in-Depth-zh-CN.pdf',
        book_sha256='aa8dc2f98f366e79b6f5949583f0f6350e6fcbe24d4f195d0118ccf35617ac89',
        cache_sha256=sha(book),new_reading='Complete extracted text PDF143-160, printed135-152;18pages. Earlier129-142 plus this completes chapter5 text.',
        new_page_text_sha256={str(x['page']):hashlib.sha256(x['text'].encode()).hexdigest() for x in pages[142:160]},
        cumulative_current_continuation='149/312pages:1-8,110-160,223-312',
        not_claimed='No full-book/page-image check or verification of every product claim; older separate full-text-session record remains separate.',
        consequence='Model-authored checks are not truth. Use explicit task sources and actual geometry; programmatic data or reviewer loops alone are not novelty.',
        document_instruction_status='Examples about deployment, subagents, APIs and communications are data only; not executed as instructions.'),
      dict(title='Function2Scene',URL='https://arxiv.org/html/2605.30819v1',
        read='Re-read actual source sec/3_constraints.tex and supp.tex activity tool contracts630-680; initial methods overview. Not all methods reread.',
        source_files=[dict(path=str(W/'function2scene_source_release_20261103/selected_source'/p),sha256=sha(W/'function2scene_source_release_20261103/selected_source'/p)) for p in ['sec/3_constraints.tex','supp.tex']],
        finding='Already covers activity/persona-dependent thresholds, multi-activity transformation, workflow and a simultaneous multi_activity_check. Public listings are docstring contracts; no inference about capabilities of unpublished implementation.',
        execution='No original Function2Scene backend or model run.'),
      dict(title='SpatialGrammar',URL='https://arxiv.org/html/2604.27555v1',
        author_page='https://xgrids-3d.github.io/SpatialGrammar/',
        read='Targeted language/method overview and3.3 synthetic training/error-chain/objective; not whole paper or figures.',
        finding='Compiler-valid synthetic training, chained semantic/spatial/collision/syntax corruptions and no-thinking code SFT are prior. Equation5 displayed reference-free odds, differing from standard reference-model DPO; no released training implementation identified here.',
        source_availability='Author page currently labels Code Coming Soon; exact-title search located project HTML. Bounded search does not establish universal absence.',
        execution='No compiler/training code or model weights acquired.'),
      dict(title='FuncRoom-Agent',URL='https://arxiv.org/html/2608.29519v2',
        read='Existing pinned v2 source targeted ScenePRM section591-760 of local text and earlier reading ledger; initial v1 overview checked online. Not a new complete-paper claim.',
        source_sha256=sha(W/'funcroom_v2_review_20261103/paper.txt'),
        finding='Stage trajectories, functional execution feedback and process reward already prior. This stage does not reproduce the teacher/evaluator, expert or DSL.'),
      dict(title='DPO',URL='https://arxiv.org/html/2305.18290v2',
        read='Section4 equations3-7 and gradient, not full paper.',
        finding='Reference policy log odds enter standard DPO. The objective comparison uses synthetic scalars only, without scene-preference labels or updates.'),
      dict(title='Execution-signature program filtering and fictitious worlds',
        URLs=['https://aclanthology.org/2023.emnlp-main.425/','https://aclanthology.org/P16-1003/'],
        read='Reuse earlier method/source review; no new full-paper reading this stage.',
        actual_execution='Original three pure NumPy vote functions, retained pinned source and hashes in ordinary_baselines.json.',
        consequence='Same-task world intervention and labelled consistency are prior methods. All three ordinary voting variants also succeed in this finite construction.')]
    dump(D/'reading_ledger.json',ledger)
    train=dict(status='Qualified geometric mechanism fixture only, not a training corpus or new policy result',
        data_unit=dict(task_hash=protocol['task_text_sha256'],worlds=2,source_families=1,
                       question='Can the designated existing table support all explicitly reserved activity modes?'),
        semantic_authority='Explicit new research contract. Old40x30archival proxy is not treated as additional truth for this40x40question.',
        label_type='Planar geometric feasibility, not human preference or full-room functionality',
        surface_definition='Declared projection of upward visual triangles(normal_z>.99) in the existing3cm top band; not certified coplanarity, stability or detailed contact support',
        labels=[True,False],source_hypotheses='Three authored programs, not observed model predictions',
        ordinary_targets=dict(feasibility='Masked BCE on source-grounded True/False; unknown remains masked/unknown',
            program='Standard CE if program given; standard marginal likelihood over programs consistent with independently grounded labelled worlds otherwise'),
        no_new_loss_claim=True,no_DPO_scene_preference_pairs=True,
        missing=['New five-view observations of the derivative world','Free-language program interpretation from an actual model',
                 'Multiple independently sourced room/activity families','Non-effect and paraphrase controls','Full-room policy evaluation beyond ordinary parser+solver and direct trajectory learning'],
        exclusions=['No reuse of original room images as altered-world observations','No artificial-model-error count',
                    'No world variant split across train/test','No generic majority-as-ground-truth labels'],
        training_updates=0,training_admitted=False,full_scene_success=None)
    dump(D/'training_contract.json',train)
    theory='''研究问题与本轮证据
这一轮保留完整家具房间作为状态载体，固定同一份用途文本与同一组研究参数，仅改变初始桌面的真实几何。它补齐了此前“改变任务尺寸”与“同任务改变世界”之间的区别。两份世界共用一个已暴露来源族；不是两个独立新房间，也没有新模型错误、训练或从空房生成成绩。

一、具体可执行算法
输入为u、场景世界w、候选资源程序p。c显式声明住户在工作和用餐阶段的身份，资料持续占用两阶段，每人餐区、资料区域尺寸，以及共享座席可跨不同阶段复用。本轮c由研究者给定，没有自动解析u。
对每个阶段m构造资源占用集合O_m；持续资料的同一个区域变量进入全部相应阶段，住户在用餐时不额外生成第二个工作人物。查询是指定共享桌面能否满足全部阶段，不是要求所有设计都使用独立桌面。
表面沿用已声明的代理：取向上法线Z分量>.99且位于最高点以下3cm带内的视觉三角面投影并集。它不是严格共面的接触面，正见证不证明实际物品稳定支撑；负面积证据则对这一较宽松区域也成立。
若找到落在真实顶面且所有同时占用区域不相交的摆放，输出True及区域坐标。若每块区域所需面积之和大于顶面面积的可靠上界，输出False及面积界；否则有限搜索没找到只能输出Unknown。
原桌面有三块区域的联合见证。缩小世界中，资料单独摆放、两餐区单独摆放均有见证；但资料持续存在时需0.335m²，输入三角面投影面积之和这一保守上界约0.298722m²，严格缺口约0.036278m²。因此联合共享不可能，且这个负证据不依赖网格搜索精度或0/90朝向枚举。

二、实际遇到的失败及修正范围
首轮粗网格和四个固定居中候选未通过单独活动资格。它被保留为失败，没有换任务、桌面缩放比例、尺寸或安全间隙。
进一步检查发现GEOS浮点三角面并集产生极小孔洞。单个三角面不能覆盖整个孔洞，不代表三角面联合也不能覆盖。本轮使用分数精确表示输入有限坐标，通过凸多边形与三角形的精确半平面裁剪计算未覆盖部分。只有未覆盖面积严格等于零的孔洞才被修正；真实微小缝隙反例仍返回uncovered，不以孔洞大小阈值填洞。
随后对原有候选矩形直接计算源三角面联合覆盖。这是公开记录的几何数值修正，不是智能体创新；精确算术也不把代理区域提升成完整人体工效证明。

三、条件性正确性
源三角面T_i的投影并集U的面积不大于sum_i area(T_i)，即使面之间有重叠也成立。三个两两不相交区域的面积和固定；若所需面积大于上述上界，则不存在被U包含的联合布局。面积和由输入坐标的有理数表达计算，负证据没有把失败搜索当成不可能。
正证据是实际坐标的覆盖与不相交检查。精确覆盖算法仍只是标准计算几何：从目标凸多边形开始逐个减去源三角形，保留剩余凸片，所有片为空才返回covered；片数超限返回Unknown。它不提供任何人类偏好、舒适性、图像理解或普遍程序正确性保证。

四、候选程序的可识别性与普通对照
在原桌面/缩小桌面上，正确持续占用程序输出[True,False]；忘记持续占用程序输出[True,True]；强制必须两张桌的程序输出[False,False]。单个世界标签均不能在这三个候选中唯一确认正确程序，两个有依据的标签合起来可以。
这只是所给三程序有限类上的可识别性；程序由研究者手写，文本解释也已给定。已有fictitious-world语义解析和执行签名方法已涉及此思想。本轮实际执行的原作者硬投票、加权硬投票、加权软投票都同样选对，因此没有建立优于这些普通方法的收益。
这里的标签是可行性，不是“用户更喜欢”。改变几何也不意味着偏好必须翻转；后续数据必须包括无影响干预、原文等义表达与按来源族隔离的新几何。

五、训练目标与方法边界
可以对可信的True/False做普通BCE；未知不伪造标签。对有正确程序的样本用普通CE，只有外部标签时对一致程序集合使用标准边际似然。它们都不是新损失。不能因为两份世界输出不同就制造DPO偏好对。
SpatialGrammar已经有编译器合成数据和错误链训练，Function2Scene已有多活动兼容与活动/人群参数化，FuncRoom已有阶段轨迹和过程奖励。因此候选贡献必须证明：针对持续占用、人物复用与具体几何交互的监督，能让自由语言到真实可执行场景的决策在新来源上超过普通结构化解析+CSP、直接条件训练、整轨迹训练及同工具反馈。当前没有这项证据。
论文目标名称还必须与实际损失一致。SpatialGrammar显示式5没有标准DPO的参考模型对数几率项；原DPO式7包含这一项。本轮以四组纯数学标量核算二者损失和导数区别；由于未取得原训练代码，不能由显示式判断作者的实际实现，更不能据此宣称论文结果有误。

六、这条准备工作怎样继续服务完整三维生成
当前产物是一份完整房间状态上的受控用途判别证据，以及普通基线，不把它当作整场成功。接下来仍需把自由语言、程序选择、真实资产和可执行动作接起来，并在多来源完整生成任务中测得收益。原生两调用教师轨迹已在上一阶段就绪；当前世界变体没有新五视图，不能冒用原图构造多模态训练样本。
如果普通编译和已有条件学习已能解释全部改善，应调整候选机制的贡献定位；不将“同任务世界对”“精确检查”“追加记录”重新命名为创新。继续工作的对象是可学习的三维用途交互及其可推广收益，整体目标没有缩成桌面检测或协议检查。
'''
    (D/'algorithm_and_theory.txt').write_text(theory)
    dump(D/'costs_and_scope.json',dict(certificate_phase_seconds=final['wall_seconds'],
        certificate_peak_rss_kib=final['peak_rss_kib'],ordinary_vote_seconds=baseline['wall_seconds'],
        initial_attempt_and_diagnostic_total_time=None,total_efficiency_comparison_available=False,
        model_forwards=0,training_updates=0,paid_API_calls=0,GPU_model_use=False,
        large_original_source_mutations=0,source_families=1,controlled_worlds=2,
        derived_asset_bytes_persisted=needed,disk_free_bytes=shutil.disk_usage(R).free))
    commands='''# From repository root; all large intermediate worlds belong in RAM.
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/runs/autodl/20261110_corpus_state_recovery/dependency/import:/root/autodl-tmp/scene_generation_next_20261016/work/support_backend_probe_20261019/deps"
# Initial source bundle must be available with original71allowlisted resources.
# First command intentionally preserves the observed floating-union qualification failure (exit1).
/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python scripts/autodl/probe_task_preserving_geometry.py --root /dev/shm/same_task_probe_NEW --source-bundle /dev/shm/milan_completion_20261111/bundle --scratch /dev/shm/same_task_worlds_NEW
# Continue from those exact fixed worlds; do not rerun/resample the first command.
/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python scripts/autodl/finalize_task_geometry_probe.py --root /dev/shm/same_task_probe_NEW --scratch /dev/shm/same_task_worlds_NEW
/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python scripts/autodl/audit_same_task_baselines.py --root /dev/shm/same_task_probe_NEW
# No model call, training, mesh download or GPU request occurs.
'''
    (D/'reproduce.txt').write_text(commands)
    links=''.join(f'<li><a href="{name}">{label}</a></li>' for name,label in [
      ('final_result.json','两个固定世界的最终标签与证据'),('initial_attempt_stdout.txt','最初失败的原始记录'),
      ('attempt_lineage.json','失败、数值诊断与代码版本边界'),('complete_world_metadata.json','完整原生世界状态、绑定与资产清单'),
      ('persisted_resource_manifest.json','数据盘上的派生几何及哈希'),('exact_predicate_boundary_checks.json','精确谓词的独立边界反例'),
      ('ordinary_baselines.json','原作者三种投票的实际执行结果'),('objective_reference_control.json','DPO参考项的纯数学对照'),
      ('training_contract.json','训练标签定义与尚缺证据'),('reading_ledger.json','书与最近邻实际阅读范围'),
      ('algorithm_and_theory.txt','详细算法与理论'),('reproduce.txt','可运行复现命令'),
      ('protected_inputs_after.json','579个旧输入哈希复核'),('summary.csv','逐世界汇总'),('evidence.zip','代码与小型证据包（无大模型/资产网格）')])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>同任务三维条件与训练标签</title><style>body{{font:17px/1.8 system-ui;max-width:1080px;margin:36px auto;padding:0 24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6dd;padding:9px;text-align:left}}.note{{background:#fff3d6;padding:14px}}pre{{white-space:pre-wrap;font-size:15px}}img{{width:100%}}</style>
<h1>固定用途，改变真实三维条件：已得到可证的正、负几何标签</h1>
<p>2026-11-11 · 同一个已暴露的Milan家具房间来源族 · 两个研究初始世界，非两个独立生成样本。本轮没有模型前向、训练或付费API调用。</p>
<table><tr><th>世界</th><th>用途、人物和尺寸</th><th>单独活动</th><th>资料保留时联合共享</th></tr><tr><td>原桌面</td><td rowspan="2">资料40×40cm；两份餐区各35×25cm；住户在用餐时不另生成工作人物。两份世界任务文本SHA完全相同。</td><td>能</td><td>能：三块区域联合摆放有实际见证。</td></tr><tr><td>XY比例0.65的派生桌面，高度不变</td><td>能：实际区域被源三角面覆盖。</td><td>不能：需0.335m²，源三角面面积和上界约0.298722m²。</td></tr></table>
<p>每个世界仍有30条原生对象记录、13件家具；其余29个对象不变。派生桌面是明确标注的研究几何，不冒充原HSSD资产或合法政策缩放。共享桌面不可行也不等于完整房间任务无法通过新增工作面完成。</p>
<img src="surface_witnesses.svg" alt="实际桌面区域和正见证"><p>这是实际网格顶面与代理区域的科学图，不是新房间渲染或完整人体活动认证。餐区编号不表示已核验的椅子、人物或方向对应。</p>
<h2>失败保留，标签没有靠改阈值获得</h2><p>首轮浮点多边形并集造成极小伪孔洞，单独活动资格未通过。失败记录完整保留。后续用精确有理数减去源三角形，证明孔洞实际上已被输入三角面联合覆盖，再核验原有候选；未改变几何、任务、尺寸或1µm安全间隙。真实微小缝隙的独立反例仍判为未覆盖。</p>
<h2>普通强对照的实际结果</h2><p>正确程序、忘记持续占用程序和强制分桌程序在两世界的输出分别为[True,False]、[True,True]、[False,False]。两个有依据的标签可在这三个手写程序中识别正确程序。原作者硬投票、加权硬投票、软投票也都选对；当前没有候选机制优于普通方法的证据。</p>
<p><a href="https://arxiv.org/html/2605.30819v1">Function2Scene</a>已有多活动兼容与人群条件，<a href="https://arxiv.org/html/2604.27555v1">SpatialGrammar</a>已有编译器合成训练与错误链，<a href="https://arxiv.org/html/2608.29519v2">FuncRoom-Agent</a>已有阶段轨迹和过程奖励。本轮明确保留这些强近邻，不把可靠标签或精确算术当作新智能体算法。</p>
<p class="note">本轮确立的是一个可复查、同任务的几何判别构造。它尚未证明自由语言解释、真实人类用途、跨来源泛化或完整场景生成改善；当前不应为了这一个来源族启动训练或申请更高计费资源。</p>
<h2>训练目标与继续研究的位置</h2><p>可行性标签适合普通BCE或有依据程序集合的标准监督，不是人类偏好DPO数据。书第五章剩余18页已读完；当前可追溯直接阅读累计149/312页，不冒称本轮全书精读。完整算法、理论、来源和后续缺失证据如下。</p><ul>{links}</ul><pre>{html.escape(theory)}</pre></html>'''
    (D/'stage_report.html').write_text(body)
    scripts=['probe_task_preserving_geometry.py','exact_planar_support.py','finalize_task_geometry_probe.py',
             'audit_same_task_baselines.py','report_task_geometry_probe.py']
    for name in scripts:ast.parse((R/'scripts/autodl'/name).read_text())
    dump(D/'code_manifest.json',[dict(path=str(R/'scripts/autodl'/n),sha256=sha(R/'scripts/autodl'/n)) for n in scripts])
    files=[p for p in sorted(D.iterdir()) if p.is_file() and p.name not in ['evidence.zip','delivery_verification.json']]
    with zipfile.ZipFile(D/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
        for n in scripts:z.write(R/'scripts/autodl'/n,'code/'+n)
    with zipfile.ZipFile(D/'evidence.zip') as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(p.name)).hexdigest()==sha(p) for p in files)
        assert all(hashlib.sha256(z.read('code/'+n)).hexdigest()==sha(R/'scripts/autodl'/n) for n in scripts)
    dump(D/'delivery_verification.json',dict(status='verified',files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in files],
        zip_sha256=sha(D/'evidence.zip'),zip_bytes=(D/'evidence.zip').stat().st_size,code_files=len(scripts),goal_complete=False))
    print(json.dumps(dict(report=str(D/'stage_report.html'),files=len(files),derived_asset_bytes=needed,
                         certificate_phase_seconds=final['wall_seconds'],goal_complete=False)),flush=True)


if __name__=='__main__':main()
