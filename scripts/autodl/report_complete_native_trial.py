"""Assemble measured scene evidence and source-grounded learning boundaries."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import shlex
import shutil
import struct


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',type=Path,required=True);a=ap.parse_args()
    s=a.stage.resolve();repo=Path(__file__).resolve().parents[2]
    valid=read(s/'trial_actual_v2/result.json');invalid=read(s/'trial_actual_v1/result.json')
    led=valid['model_ledger'];bad=invalid['model_ledger'];ret=read(s/'trial_actual_v2/hssd_service.json')
    artifact=read(s/'partial_artifact_v1/result.json');namespace=read(s/'namespace_diagnostic_v1/result.json')
    half=read(s/'floor_halfspace_certificate_v1/result.json');rollback=read(s/'mutable_asset_rollback_v2/result.json')
    assert artifact['cold_geometry_count']==17 and rollback['status']=='same_JSON_different_asset_state_reproduced'
    assert len(led['forwards'])==23 and led['output_tokens']==4553 and valid['status']=='failed_no_restart'
    assert (s/'TRIAL_CLOSED_v2.json').exists()
    # Revalidate original source inputs and the exact pre-inference code freeze.
    materialization=read(repo/'runs/autodl/20261110_remote_native_controller/materialization.json')
    source_rows=[dict(path=r['source'],expected_sha256=r['sha256'],actual_sha256=sha(r['source']))
                 for r in materialization['copied_sources']]
    for r in source_rows:assert r['expected_sha256']==r['actual_sha256']
    protocol=read(s/'trial_protocol_v2.json')
    frozen=[dict(path=p,expected_sha256=h,actual_sha256=sha(p)) for p,h in protocol['code_sha256'].items()]
    assert all(x['expected_sha256']==x['actual_sha256'] for x in frozen)
    weights=[]
    for r in protocol['weight_sha256_verified']:
        st=Path(r['path']).stat();same=(st.st_size,st.st_mtime_ns,st.st_ino)==(r['bytes'],r['mtime_ns'],r['inode'])
        weights.append(dict(path=r['path'],same_size_mtime_inode=same,previous_full_SHA256=r['sha256']))
        assert same
    dump(s/'source_before_after.json',dict(original_source_inputs=source_rows,pre_inference_code=frozen,
         checkpoint_stat_revalidation=weights,new_full_weight_payload_SHA256_pass=False,
         deleted_source_files=0,original_trial_inputs_unchanged=True))
    # Header-only resource scenarios, not an unauthorized training/fit test.
    headers=[];matrices=[]
    for p in sorted(Path(protocol['checkpoint']).glob('model-*.safetensors')):
        with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
        d=json.loads(raw);headers.append(dict(path=str(p),header_sha256=hashlib.sha256(raw).hexdigest()))
        for name,row in d.items():
            if 'language_model.layers.' in name and name.endswith('.weight') and len(row.get('shape',[]))==2:
                matrices.append(dict(name=name,shape=row['shape']))
    scenarios=[]
    for rank in (8,16):
        for label,items in [('decoder_qkvo',[x for x in matrices if '.self_attn.' in x['name']]),('decoder_attention_and_MLP',matrices)]:
            count=sum(rank*sum(x['shape']) for x in items)
            scenarios.append(dict(targets=label,rank=rank,target_matrices=len(items),adapter_parameters=count,
               FP32_adapter_params_grads_two_FP32_moments_GiB=count*16/2**30,
               base_FP16_weight_GiB=8767123696*2/2**30,
               activation_and_training_peak='not measured',V100_training_fit_proven=False))
    dump(s/'training_resource_scenarios.json',dict(header_only=True,headers=headers,scenarios=scenarios,
         model_weights_loaded=False,optimizer_updates=0,vision_embedding_lmhead_not_selected=True,
         inference_peak_is_not_training_peak=True,
         pinned_original_GRPO_two_independent_BF16_weight_copies_GiB=2*8767123696*2/2**30,
         two_copies_alone_exceed_V10032GiB=True,
         original_GRPO_source='train_grpo.py:711-727 creates distinct policy and frozen reference before optimizer',
         no_claim_that_Adam_memory_estimate_describes_the_source_Adafactor_optimizer=True))
    # Compact raw trace copies; full byte evidence remains in the companion bundles.
    import tarfile
    r=read(s/'evidence_solid_v2/receipt.json');archive=Path(r['path']);archive=archive if archive.is_absolute() else repo/archive
    assert sha(archive)==r['sha256'];review=s/'review_traces';review.mkdir(exist_ok=True)
    wanted={'scene/rollout_floor_plan/trace.json':'floor_trace.json',
            'scene/room_bedroom/rollout_furniture/trace.json':'furniture_trace.json',
            'scene/room_bedroom/rollout_wall/trace.json':'wall_trace.json',
            'actor/hard_deadline.json':'actor_deadline.json'}
    found={};manifest=None;by_hash={}
    with tarfile.open(archive,'r|xz') as tar:
        for item in tar:
            data=tar.extractfile(item).read()
            if item.name=='paths.json':
                manifest=json.loads(data)
                for row in manifest['files']:
                    if row['path'] in wanted:by_hash[row['sha256']]=(row['path'],wanted[row['path']])
            elif item.name.startswith('blobs/') and item.name.split('/',1)[1] in by_hash:
                h=item.name.split('/',1)[1];assert hashlib.sha256(data).hexdigest()==h
                old,name=by_hash[h];(review/name).write_bytes(data);found[old]=h
    assert set(found)==set(wanted)
    dump(review/'source_receipt.json',dict(archive_sha256=r['sha256'],exact_copies=found))
    furniture=read(review/'furniture_trace.json')
    summary=dict(status='bounded_native_generation_executed_partial_scene_cold_verified',
        previous_goal_turn='verified wait: actual PIDs and changing inference ledger were inspected',
        this_goal_turn='progress: real terminal execution, persisted complete evidence, native geometry/state diagnostics and source-level objective correction',
        task=protocol['task'],default_admission_task_not_heldout_benchmark=True,
        corrected_continuation=dict(forwards=len(led['forwards']),output_tokens=led['output_tokens'],
            policy_forwards=sum(x['role']=='native_joint_policy' for x in led['forwards']),
            tool_VLM_forwards=sum(x['role']=='native_tool' for x in led['forwards']),
            native_tool_attempts=valid['native_tool_attempts'],fresh_CLIP_queries=len(ret['encoder_requests']),
            seconds=valid['seconds'],forward_body_seconds=sum(x['seconds'] for x in led['forwards']),
            model_loads=len(led['model_loads']),same_frozen_model_for_policy_and_tools=True,
            peak_CUDA_allocated_GiB=max(x['peak_cuda_allocated_bytes'] for x in led['forwards'])/2**30,
            termination='owned actor hit the prospectively fixed900s deadline, exit124; not a 32-forward or output-token exhaustion',
            original_trial_outcome=valid['status']),
        setup_invalid_attempt=dict(forwards=len(bad['forwards']),output_tokens=bad['output_tokens'],
            cause='resume wrapper replaced HouseLayout while native tool closures retained the original instance',quality_evidence=False),
        prior_first_floor_step_cost=dict(forwards=1,output_tokens=141),
        new_stage_forwards=len(led['forwards'])+len(bad['forwards']),
        including_prior_first_step_forwards=len(led['forwards'])+len(bad['forwards'])+1,
        stages=valid['stages'],floor_geometry_actual=True,placed_furniture=4,
        required_object_names_bound=4,independent_semantic_completion_not_proven=True,
        actual_partial_GLB=dict(path=str(s/'partial_artifact_v1/partial_scene.glb'),sha256=artifact['GLB_sha256'],
                               bytes=artifact['GLB_bytes'],cold_geometry_count=artifact['cold_geometry_count'],
                               exporter_exit_code=artifact['exporter_exit_code']),
        namespace_diagnostic=namespace,half_space_certificate={k:v for k,v in half.items() if k!='rows'},
        mutable_asset_JSON_only_rollback_reproduced=True,
        all_original_native_sources_unchanged=len(source_rows),all_frozen_trial_code_unchanged=len(frozen),
        paid_API_calls=0,training_updates=0,whole_scene_success=None,learned_advantage_verified=False,
        independent_method_comparison_runs=0,goal_achieved=False)
    assert all(x['exit_code'] is not None for x in valid['owned_processes'])
    summary['all_owned_trial_processes_terminal']=True
    dump(s/'summary.json',summary)
    with (s/'summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['condition','role','forwards','output_tokens','wall_seconds','quality_interpretation'])
        writer.writerow(['setup_invalid_v1','all',len(bad['forwards']),bad['output_tokens'],invalid['seconds'],'invalid setup; not a quality sample'])
        for role in ['native_joint_policy','native_tool']:
            items=[x for x in led['forwards'] if x['role']==role]
            writer.writerow(['corrected_v2',role,len(items),sum(x['output_tokens'] for x in items),sum(x['seconds'] for x in items),'forward body time only; incomplete episode'])
        writer.writerow(['corrected_v2','entire_continuation',len(led['forwards']),led['output_tokens'],valid['seconds'],'deadline terminated; no full task success claim'])
    # Required-instance review is explicitly an assistant inspection, not a blinded independent score.
    state=read(s/'partial_replay_v2/actual_room_state.json')
    with (s/'required_instances_review.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['original_requirement','object_id','observed','independent_verified_success','limitation'])
        for oid in ['queen_bed_0','nightstand_0','nightstand_1','wardrobe_0']:
            ob=state['objects'][oid];size=[b-a for a,b in zip(ob['bbox_min'],ob['bbox_max'])]
            requirement={'queen_bed_0':'one bed','nightstand_0':'first of two required nightstands',
                         'nightstand_1':'second of two required nightstands','wardrobe_0':'one wardrobe'}[oid]
            w.writerow([requirement,oid,'mesh retained; size_m='+str(size),'unknown',
                        'name binding is not category/function proof; assistant viewed topdown and south only'])
        w.writerow(['wardrobe in room corner','wardrobe_0','near northeast in saved views; no arbitrary near threshold introduced','unknown','semantic category and complete functional use not independently verified'])
    sources=[]
    base=Path('/root/autodl-tmp/agent_method_preparation_20261027/work')
    source_files=[(base/'context_credit_source_review_20261103/sources/train_grpo.py','src/train_grpo.py',[['1','122'],['505','552'],['682','742'],['804','954']]),
      (base/'context_credit_source_review_20261103/sources/grpo_objective.py','src/grpo_objective.py','complete'),
      (base/'autodl_continuation_20261110/training_source_review_20261110/upstream/src/train_sft.py','src/train_sft.py',[['1','126'],['145','175']]),
      (base/'autodl_continuation_20261110/training_source_review_20261110/upstream/src/sft_dataset.py','src/sft_dataset.py',[['1','115']])]
    tree={x['path']:x['sha'] for x in read(repo/'runs/autodl/20261110_remote_native_controller/designer_tree.json')['tree']}
    for path,relative,lines in source_files:
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==tree[relative]
        sources.append(dict(local_path=str(path),sha256=sha(path),git_blob_SHA1=blob,
             URL='https://github.com/ZJU-OmniAI/Scene-Designer/blob/a0565e50ed7d32c0e09132b9311c0b9b91165405/'+relative,
             read_lines_this_turn=lines,whole_module_executed_as_training=False))
    book=read(s/'book_review/book_identity.json')
    reading=dict(date_UTC=datetime.now(timezone.utc).isoformat(),book=dict(path=book['path'],sha256=book['sha256'],total_pages=312,
       previously_read_PDF_pages=[[1,6],[223,226]],fully_read_PDF_pages_this_turn=[list(range(227,231)),list(range(239,244)),list(range(248,257))],
       header_search_only_pages=[231,262],all_312_pages_read=False,not_treated_as_user_instructions=True,
       corrections=['SFT does not imply impossible OOD generalization','NaN rewards are not zero-advantage groups',
         'Flat local rewards do not imply no global/auxiliary signal; inspect the exact implementation',
         'Text-model LoRA results do not establish Qwen3-VL multimodal training fit on V100',
         'One deterministic rollout is not an independent-sampling pass@k estimate']),
       primary_papers=[
         dict(URL='https://arxiv.org/html/2610.23121v1',read='Appendix B,C.1-C.3 and public repository release boundaries; targeted reread',not_new_full_paper_read=True),
         dict(URL='https://arxiv.org/html/2602.09432v1',read='3.1-3.4, including retrieval/postprocessing and CoT training; targeted reread',not_new_full_paper_read=True),
         dict(URL='https://arxiv.org/html/2607.13539v1',read='III-A and implementation/ablation passages; targeted reread',not_new_full_paper_read=True),
         dict(URL='https://thinkingmachines.ai/blog/lora/',read='Methods, all-layer vs attention, RL, LR, capacity and scope discussion; HTML text, no graph-by-graph audit'),
         dict(URL='https://thinkingmachines.ai/blog/on-policy-distillation/',read='Introduction and Implementation through reverse-KL pseudocode; not full article'),
         dict(URL='https://drake.mit.edu/doxygen_cxx/group__multibody__parsing.html',read='declare_convex semantics'),
         dict(URL='https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_query_object.html',read='signed-distance and Mesh accuracy boundaries')],
       pinned_source_reading=sources,
       source_flow_conclusions=[
         'The source accumulates every branch-group backward before one optimizer.step. It is not an optimizer update after each group; that suspected stale-behavior explanation is rejected.',
         'A literal zero leaf can count as n_groups_used and cause the applied_updates counter to advance. The counter alone is not proof of changed model parameters.',
         'The original GRPO loads independent full policy and reference copies. Their BF16 stored weights alone exceed32GiB; this was not executed as a training run.',
         'Current v2 was sequential inference, not GRPO branching. The mutable-file rollback issue was reproduced separately and is not assigned as the cause of v2 termination.'],
    native_source_reading=['stateful_floor_plan_agent.py:1244-1369,1558-1660','room.py:916-1060,1570-1665',
          'physics_validation.py:155-300','sdf_generator.py:170-245,250-350,466-500',
          'rescale_helpers.py:24-180','grpo_rollout.py:233-265,351-500','furniture_tools.py:796-830'],
       no_claim_of_exhaustive_frontier_coverage=True)
    dump(s/'reading_ledger.json',reading)
    algorithm=dict(status='falsifiable learning comparison, not an established novel method',
       target='Complete editable indoor3D generation from the unchanged original description, independently useful beyond ordinary search/feedback',
       state=dict(X='actual task, observations and history',D='self-generated decision proposal, never a measured fact',
           S='physical/scene state INCLUDING mutable asset files, registry and relevant runtime state',
           actual_new_counterexample='Same exact restored SceneJSON and next native rescale action give different SDF geometry after unrolled file mutations; JSON alone is not a sufficient branch checkpoint',
           qualified_fix='Byte restoration of the one mutated SDF was actually checked; entire registry/cache rollback remains unqualified'),
       objective='J(theta;psi0,B)=E[1{all original clauses independently verified true at termination} | shared fixed tool backend psi0 and shared total cost budget B]',
       termination='budget expiration is an incomplete/failed episode, never a successful STOP; preserve all unknown clauses and failed calls in denominator',
       conditional_factorization='p(D,A|X)=p(D|X)*p(A|X,D) is a chain-rule identity, not a new algorithm or performance theorem',
       supervised_arms=dict(A='action CE only',AUX='same original D and A targets, separately conditioned on X; A does not consume generated D',
                            J='original autoregressive joint think+action, already a strong native baseline',
                            P='generate own D then execute A conditioned on own D; extra forward/prefix/token cost charged'),
       loss_contract=dict(
          action='L_A = sum_i w_i^A CE(A_i* | X_i)',
          auxiliary='L_AUX = L_A + lambda sum_{i with original D} w_i^D CE(D_i* | X_i)',
          joint='L_J = sum_{i with original D} [w_i^A CE(A_i* | X_i,D_i*) + lambda w_i^D CE(D_i* | X_i)] + sum_{i without D} w_i^A CE(A_i* | original no-D protocol prefix); source masks and family masses preserved, missing D not fabricated',
          deployment='Conditional arms consume their own generated D. A teacher D supplied at evaluation is privileged-input scoring, not closed-loop policy performance.',
          separate_pass='P uses the existing phase-training component/mask contract; merely splitting a mathematically identical prefix is not a new loss. No claim that different actual prompts are identical.',
          reference='runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl and scripts/autodl/phase_trainer.py'),
       equal_data='Reuse original unit/family/component masses. Do not synthesize the14 missing D or5 missing original views from scoring answers. Current24 local units are not35K complete success trajectories.',
       controls=['ordinary native tools and bounded search','same-source auxiliary D with direct actions','original joint reasoning-action policy',
                 'same measured geometry available to all arms if a new geometric representation is introduced'],
       tools='Current trial has one frozen base model for both policy and tool VLM. A future policy adapter must be disabled for all tool calls, with role state verified, to identify a policy effect.',
       trajectory_gradient='If policy AND tool distributions are changed: grad J=E[R*sum_t(grad log p_theta(D_t,A_t|H_t)+grad log kappa_theta(omega_t|H_t,D_t,A_t))] when reward has no direct theta dependence. The joint score includes both generated D and executable A; only a properly marginalized action policy permits omitting an explicit D score. Omitting mutable tool terms is not this shared-model estimand.',
       runtime_budget='B is the shared budget vector (model forwards/output tokens, tool/candidate attempts, elapsed time), not an invented conversion of unlike costs. Treat varying tool/model duration and remaining budget as part of a budgeted/semi-Markov decision process. A fixed per-step discount and wall-clock success probability are not automatically equivalent.',
       branching_scope='Current v2 was sequential inference. The separately executed JSON-only branch counterexample qualifies a future common GRPO/search host, not a claimed cause of the current deadline.',
       reward_evidence=dict(original_metric_retained=True,floor_contact_reports_unreliable_in_this_case=True,
           native_flat_group='Pinned grpo_loss_for_group returns a literal detached-from-model zero leaf before calling its KL objective when both local and global advantages vanish; CPU control flow actually executed',
           not_generalized='Other objectives can retain KL/auxiliary gradients. NaNs require rejection, not zero-signal interpretation.',
           no_new_training_successes=True),
       why_not_immediate_RL=['No independently verified complete positive room from this trial',
           'Two native floor-body collision reports conflict with geometric separation evidence',
           'JSON-only branch reset omits mutable asset bytes',
           'Available asset subset and category verification constrain interpretation',
           '32GB inference success is not a real multimodal backward/optimizer memory test'],
       theory_limits=['Teacher-forced target loss is not closed-loop success on policy-induced scenes',
          'Ordinary potential shaping needs the correct terminal potential; it cannot repair an unsound verifier',
          'A generated fluent D does not prove action correctness or a causal mechanism',
          'OPD supplies teacher guidance on student prefixes but requires a reliable teacher and actual3D state transitions; teacher prose cannot stand in for executed geometry'],
       novelty='Reasoning, memory, rollback, same-source auxiliary learning, reverse-KL distillation and ordinary search all have direct prior art. No CVPR-level contribution or improvement is claimed yet.',
       actual_this_turn='Real native generation continuation, saved partial3D artifact and exact tool/state/GRPO diagnostics; no new training',
       next_work_boundary='Do not tune289/404, add samples, alter thresholds or label this partial room successful. Consolidate shared native execution and actual available expert supervision before any authorized learned comparison.',
       training_updates=0,goal_achieved=False)
    dump(s/'algorithm_and_theory.json',algorithm)
    source_links=''.join('<li><a href="'+html.escape(x['URL'])+'">'+html.escape(x['URL'])+'</a></li>' for x in reading['primary_papers'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整原生流程试跑与学习目标核查</title>
<style>body{{font:17px/1.7 system-ui;max-width:1080px;margin:38px auto;padding:0 24px;color:#17212b}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #ccd4db;padding:9px;text-align:left}}code,pre{{background:#eef2f5;padding:3px;white-space:pre-wrap}}.note{{border-left:5px solid #c47b18;padding:12px;background:#fff6e9}}img{{max-width:48%;vertical-align:top}}a{{color:#1258a0}}</style>
<h1>原生场景生成已实际运行；学习收益尚未建立</h1><p>记录日期：{reading['date_UTC']}。目录沿用11月10日命名；v2实际推理跨入11月11日。原始默认任务：<code>{html.escape(protocol['task'])}</code>。这是单一接入样例，不是新测试集或论文效果比较。</p>
<p class="note">已交付真实四件家具的部分卧室、五视角、可冷读取GLB和全部执行证据。原生整场流程在固定模型服务时限结束，墙饰未完成、天花和小物件未运行。不能写成完整生成成功，也不能把接通工具或修复回滚当作新算法。</p>
<p>本轮执行器是Scene-Designer调度＋SceneSmith原生工具，使用本地RPC适配、512像素俯视/512×384侧视和两种本地默认材质。并非原论文完整配置复现；本地权重同时承担策略与工具VLM，资产检索使用真正的官方CLIP。</p>
<h2>实际执行与成本</h2><table><tr><th>项目</th><th>结果</th></tr><tr><td>有效v2续跑</td><td>23次真实前向：19次策略＋4次工具VLM；4553输出token；35次原生工具尝试；8次新CLIP查询</td></tr>
<tr><td>时间</td><td>续跑主体{valid['seconds']:.3f}秒；模型前向体合计{summary['corrected_continuation']['forward_body_seconds']:.3f}秒，不含装载、预处理和服务工具。900秒服务时限导致退出124，尚未耗尽32次或16384输出token上限；未重启挑结果。</td></tr>
<tr><td>费用与训练</td><td>付费API0、训练更新0。本地推理消耗已开实例，不能说总运行零成本。</td></tr>
<tr><td>前置成本</td><td>初始已记录1次/141输出token另计；失效v1的15次/2379输出token保留为工程调试成本，不当质量样本。</td></tr>
<tr><td>配置与资源边界</td><td>同一个冻结8B本地模型；V10032GB、容器25GiB RAM/6CPU。v2实际峰值CUDA分配{summary['corrected_continuation']['peak_CUDA_allocated_GiB']:.3f}GiB；本轮没有测完整进程树RAM峰值。</td></tr></table>
<p>户型阶段返回，有一扇门、两扇窗；家具阶段走满10轮后由阶段上限推进，仍有原生检查失败。墙饰一次请求5个候选，只完成1个资产处理且未放置；另外4个因模型服务到时关闭而失败。候选资产是本地可用子集：大件141/5840、墙饰59/2357，不能据此评判完整HSSD或原论文配置。</p>
<h2>实际部分场景</h2><p><a href="partial_artifact_v1/partial_scene.glb">本地部分场景GLB</a>，{artifact['GLB_bytes']:,}字节，冷读取17个几何、31个节点。导出进程已写完GLB后以-11退出；进程异常与产物可读分别报告，不能把导出器记成正常通过。重读的GLB内没有外部图片/缓冲URI。</p>
<img src="partial_replay_v2/views/step_000.topdown.png" alt="真实部分卧室俯视"><img src="partial_replay_v2/views/step_000.south.png" alt="真实部分卧室南侧视角">
<p>四个必需实例名称都能绑定到网格。衣柜资产实际高约0.838米，类别与使用功能存在疑问；当前只有助手查看图像，没有独立语义验收，不能把4/4名称绑定当作任务成功。北墙与床的约4.6厘米相交仍保留。<a href="required_instances_review.csv">逐项审阅表</a>。</p>
<h2>先把反馈与状态定义查实</h2><p>原生报告3个碰撞物体对。53条原始负距离中有35条涉及地板；29条的几何全部位于地板上方或相切，其中10条却被报告穿透超过1毫米。所有证据使用原坐标、原尺度和实际OBJ顶点。支持分离证明不覆盖北墙等其他接触，旧分数未改，也没有生成新的成功标签。负距离的具体数值算法成因尚未确定。</p>
<p>一次同状态对照只改XML命名空间拼写：509个Mesh变为Convex，53条负距离、3个原生物体对及最大深度全部不变。因此“改前缀就解决穿透”已被本例否定，不能宣传为修复效果。详见<a href="namespace_diagnostic_v1/result.json">命名空间对照</a>、<a href="floor_halfspace_certificate_v1/result.json">分离证据</a>。</p>
<p>另一项实际源码试验证明：原生缩放会写SDF文件，单纯恢复SceneJSON不能撤销。相同JSON、相同下一次0.9缩放，在受前一个分支污染时文件尺度变为0.455625；从原文件恢复则是0.50625，逻辑对象却都报告0.50625。恢复这一SDF字节后结果一致并可完全撤销。试验调用原生helper的显式registry=None模式；没有验证整个SDK注册表、缓存或随机状态。<a href="mutable_asset_rollback_v2/result.json">完整实测</a>。这要求公共分支基线修正，属于普通工程正确性，不是新的Agent收益。</p>
<h2>算法、训练目标与强对照</h2><p>研究对象仍是生成决策能否改善完整场景。实际环境状态必须包括可变资产文件与注册表，不能只写场景JSON。决策D是模型提出的计划；真实执行结果是另一种证据。模型写了“已完成”或SDK返回success，都不替代原始要求验收。</p>
<pre>J(θ; ψ₀, B) = E[ 1{{原始要求在终态全部被独立验证为真}} | 固定共同工具ψ₀、相同总成本B ]
p(D,A | X) = p(D | X) · p(A | X,D)  （概率链式法则，不是新算法）</pre>
<table><tr><th>对照</th><th>保留什么</th><th>要辨别什么</th></tr><tr><td>A</td><td>同源动作监督，直接行动</td><td>直接训练是否已够用</td></tr><tr><td>AUX</td><td>同样D标签作为辅助监督，动作不读取生成D</td><td>收益是否仅来自更多监督</td></tr><tr><td>J</td><td>原生联合think＋action</td><td>不能把已有推理再包装成新机制</td></tr><tr><td>P</td><td>先生成自己的D，再据D行动</td><td>额外计算与依赖结构是否有独立收益</td></tr><tr><td>普通搜索</td><td>相同工具、资产、观测、上游权限和总预算</td><td>学习是否超过普通计算</td></tr></table>
<p>目标不是D写得更长、局部布尔项更多，而是同预算下整场要求、几何与保留效果。不同耗时工具使预算成为状态的一部分；额外前向、工具侧VLM、失败尝试、装载和渲染都要计入。未来只更新策略LoRA时，工具侧必须确实关闭该adapter并验证；否则同时改了环境，不能把差异全归给策略。</p>
<p>现有本地24个原始单步目标只有10条原始D、19条原始视图；这些覆盖数据沿用既有台账，不是本轮新增训练集，也不能冒充35K整场正例。当前试跑不自动加入成功示范。没有新训练授权，未运行8B反向或优化器。</p>
<h2>训练源码与书中论述的准确边界</h2><p>重新按Git blob核验训练源码。原生<code>grpo_loss_for_group</code>在局部、全局优势同时为零时，直接返回独立零张量，尚未调用含KL的目标。这一实现的两种常数组和非零全局分支已经CPU执行；只对独立零叶做backward，无模型参数。其他RL目标可能仍有KL或辅助梯度，不能混为一谈。<a href="grpo_flat_group_source_check_v3/result.json">源码执行记录</a>。</p>
<p>源码是在全部组累积梯度后统一step，不能说它逐组改了行为策略；但applied_updates计数器前进本身，也不能证明全零优势时参数有变化。原生GRPO加载两个独立完整模型，仅BF16权重约32.66GiB，已经超过V10032GiB，尚未包含梯度与激活；没有尝试在本机原样训练它。</p>
<p>本轮v2是顺序推理，没有运行GRPO树；资产文件回滚问题由独立探针再现，不能当作v2终止原因。</p>
<p>LoRA一手研究支持检查MLP覆盖和调参，不能据此保证本VLM训练适配。实读checkpoint头：decoder全部线性层rank8约21.82M参数，attention-only约7.67M；加上16.33GiB冻结权重仍未包含多模态激活。没有做训练峰值测试。<a href="training_resource_scenarios.json">计算假设与头哈希</a>。</p>
<p>本轮完整读PDF227–230、239–243、248–256页，并检索231–262页标题；前轮读过封面目录和223–226。没有读完312页。书的启发是先分清协议、策略、环境和可靠奖励；图示中“SFT不泛化”与“NaN导致零梯度”不能作为普遍结论。近邻已有CoT、诊断行动、Critic和回滚；在轨蒸馏也已有明确先例。当前没有新损失、新理论或CVPR创新成立的证据。<a href="reading_ledger.json">准确阅读范围</a>；<a href="algorithm_and_theory.json">完整方法与理论合同</a>。</p>
<h2>交付、只读与持久性</h2><p>两次试跑的1607条原始文件路径均保留字节、路径与哈希，分别固实压缩为193,633,176与73,221,696字节，已保存数据盘并冷恢复验证v2。原始ZIP压缩字节不保证可重建；所有成员字节均已校验。大伴随包和GLB保留本地，不塞入Git；公开审阅包含哈希、代码、精简原始轨迹和选定实际视图。</p>
<p>清理了5166个有源码的pyc及闲置CUDA11.8编译器，保留安装元数据和重装地址；另一个已证实冗余的旧297 ZIP此前在本阶段清理。没有删除模型、研究源码或原始实验结果，没有硬链接去重。原CUDA11.8构建若重跑需重装；当前CUDA12.8推理环境未改。</p>
<p>复核{len(source_rows)}个原始源码输入与{len(frozen)}个推理前冻结文件，均一致。原始试跑仍按失败保留；所有拥有的试跑子进程已退出。主研究目标保持未完成，不能用这些准入工作替代完整生成和独立方法收益。</p>
<p><a href="summary.csv">统计CSV</a> · <a href="source_before_after.json">源码前后核验</a> · <a href="COMMANDS.txt">已运行与冷复核命令</a> · <a href="evidence.zip">轻量证据包</a></p><h2>一手资料</h2><ul>{source_links}</ul></html>'''
    (s/'stage_report.html').write_text(text)
    environment=read(s/'ENVIRONMENT.json')
    env='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(':'.join(environment['native_dependency_paths']))+' '
    py='/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python -B '
    commands='Closed inference commands are historical evidence, NOT a command to reopen unused budgets.\n'
    commands+=env+py+'scripts/autodl/run_complete_native_trial.py --stage '+str(a.stage)+' --attempt v2\n\n'
    commands+='Zero-model diagnostics: use a NEW output directory; existing evidence is never overwritten.\n'
    for script,out in [('diagnose_native_sdf_namespace.py','namespace_recheck_NEW'),('probe_native_mutable_asset_rollback.py','mutable_state_recheck_NEW'),('recover_native_partial_glb.py','partial_artifact_recheck_NEW')]:
        commands+=env+py+'scripts/autodl/'+script+' --stage '+str(a.stage)+' --out '+str(a.stage/out)+'\n'
    commands+=py+'scripts/autodl/check_native_grpo_flat_group.py --out '+str(a.stage/'grpo_source_recheck_NEW')+'\n'
    commands+='python -B scripts/autodl/certify_native_floor_contacts.py --stage '+str(a.stage)+' --out '+str(a.stage/'halfspace_recheck_NEW')+'\n'
    (s/'COMMANDS.txt').write_text(commands)
    print(json.dumps(dict(status=summary['status'],report=str(s/'stage_report.html'),new_forward_count=summary['new_stage_forwards'],
                         goal_achieved=False,free_GiB=shutil.disk_usage(s).free/2**30)))


if __name__=='__main__':main()
