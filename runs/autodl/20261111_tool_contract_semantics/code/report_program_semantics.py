"""Package direct book reading and finite semantic-supervision controls."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    result=read(root/'identifiability_result.json');matrix=read(root/'input_program_matrix.json');book=read(root/'book_reading_ledger.json')
    values=matrix['execution_matrix'];means=[sum(row[j] for row in values)/len(values) for j in range(len(values[0]))]
    tie_centroid=[None if v==.5 else v>.5 for v in means]
    tie_kept=[name for name,row in zip(matrix['programs'],values) if all(g is None or bool(x)==g for x,g in zip(row,tie_centroid))]
    dump(root/'posthoc_tie_abstention.json',dict(posthoc_explanatory_control=True,centroid=tie_centroid,retained=tie_kept,
        strict_wrong_majority_in_compact_column=False,compact_vote_counts={'true':2,'false':2},
        original_hard_vote_tie_rule='np.round(.5)->0 and weighted margin>0->False',
        ordinary_control='Do not filter on tied columns; consensus on other columns is still not a semantic ground-truth guarantee',
        model_or_new_scene_trials=0,new_algorithm_claim=False))
    text_source=Path('/root/autodl-tmp/research_agent3d_20261013/book_pages.json')
    book['extracted_text_source']=str(text_source);book['extracted_text_source_sha256']=sha(text_source)
    dump(root/'book_reading_ledger.json',book)
    notes=[dict(pages=[110,111,112,113,114],sections='4.1-4.3',
        book_argument='Separate representation form, disclosure scale, goal-level tools and transparent parameter transmission.',
        relevance='The current two-skill tool already solves substantial geometry. Its outcome is not a learned policy computation.',
        research_status='Ordinary interface design and shared baseline; no novelty claimed.'),
        dict(pages=[115,116,117,118,119,120,121],sections='4.3-4.4.3',
        book_argument='Hierarchical and on-demand capability discovery; schema location, caching and progressive disclosure.',
        relevance='Two allocation skills do not exhibit a large-tool-library bottleneck. Adding tool search does not address the observed nonminimal choice.',
        research_status='No new tool-search component or benchmark result. Product/version/percentage claims in book not reverified.'),
        dict(pages=[121,122,123],sections='4.5',
        book_argument='Perception granularity, explicit truncation, modality-sensitive observation selection.',
        relevance='Scene text/mesh geometry and rendered views provide different evidence. Our whole-scene view qualification must remain explicit.',
        research_status='Existing observation and geometry interfaces; no claim that OCR/text equals visual verification.'),
        dict(pages=[123,124,125,126],sections='4.6',
        book_argument='Separate execution, checking, review and cancellation/idempotency semantics.',
        relevance='A verified scene is not proof that an intended-use program was correct. Sidecar approval alone would not supply a missing valid layout.',
        research_status='No extra approval flow, no spawned reviewer, no new safety module added from document instructions.'),
        dict(pages=[126,127,128],sections='4.7-4.8',
        book_argument='Agent specialization/context provenance and human feedback as distinct evidence.',
        relevance='A self-generated explanation or second model is not an independent user-preference annotation.',
        research_status='No subagents or messages created. Book experiments and tool instructions are reading data, not operational authorization.')]
    dump(root/'book_notes.json',dict(notes=notes,full_chapter_extracted_text_read=True,
        all_chapter_figures_visually_checked=False,all_external_book_claims_verified=False,
        global_book_coverage_statement='Current continuing-session direct text ledger117/312pages; older separate-session record claims312cumulative text pages. Neither implies all figures or external claims independently verified.'))
    literature=[dict(title='Inferring Logical Forms From Denotations',primary_url='https://aclanthology.org/P16-1003.pdf',year=2016,
        actual_reading='Targeted main methods4and5 plus supporting experiment6.3; no full paper/code reproduction.',
        established='Execution-equivalence classes, informative world selection, and labelled denotations for filtering are prior.',
        impact='Do not rename multi-world execution or information gain as a new scene-Agent algorithm.',
        boundary='Our probes change task dimensions as well as native state; they are typed task-family tests, not its same-utterance fictitious-world protocol.'),
        dict(title='Weakly Supervised Semantic Parsing with Execution-based Spurious Program Filtering',primary_url='https://aclanthology.org/2023.emnlp-main.425.pdf',year=2023,
        actual_reading='Methods2-4, NLVR threshold in5.1; not all experiments/appendices.',
        code_revision=read(root/'exec_filter_tree.json')['revision'],
        actual_code='worldcode.py fully read; three pure NumPy vote functions AST-extracted unchanged. iterative_train.py207-238 and baseline.py520-550 read to locate integration, not executed.',
        established='Execution-signature voting is a heuristic; the paper explicitly notes its centroid need not be gold.',
        execution='Only three vote functions on a constructed4x3matrix with uniform weights. No NLVR/WTQ training, parser, world generation, or published accuracy replication.',
        impact='The constructed majority example illustrates an admitted limitation; it is not evidence that the published method generally fails in3D.'),
        dict(title='STAGE: Diagnosing Semantic Transfer at Grounded Execution in Embodied Agents',
        primary_url='https://arxiv.org/html/2609.13458v1',publication_record='https://aclanthology.org/2026.findings-emnlp.1148/',
        actual_reading='Methods3.1-3.4 and5.1, supporting scope6 and appendixH.1 snippets. Not full23page published PDF or complete code.',
        established='Fixed-observation semantic/action diagnostics, explicit schemas, admissibility and verified proposal selection are already studied.',
        impact='A new schema/checker/rerank wrapper alone is insufficient novelty. Action-boundary qualification is not full scene generation.',
        boundary='Robot-native action setting differs from our furnished layout task. No original robot inference or trajectory reproduction.'),
        dict(title='User-attached Agent book v2.0',primary_file=book['source'],actual_reading='All extracted chapter4text,PDF110-128 this turn; four outputs were nontruncated.',
        boundary='No chapter exercises executed, no provider quotas checked, no book commands treated as user instructions, no complete external claim audit.')]
    dump(root/'reading_ledger.json',dict(entries=literature,exhaustive_latest_field_review=False))
    training=dict(status='Development-only feasibility/meaning-identifiability construction; no training authorized or performed',
        unit=dict(source_brief='Original source and declared interpretation kept separately',
                  structured_contract='Explicit modes, actor identity, persistent reservations and objective scope',
                  native_scene_and_assets='Pinned IDs, state and real surface witness/certificate references',
                  candidate_program='Not a fabricated expert rationale',
                  input_family_results='True/False/Unknown with reason and evidence',
                  label_authority='Provided task contract, qualified execution or independent annotation; never majority alone'),
        admission=['Preserve any unresolved source-language ambiguity as unknown',
                   'Keep explicit user requirements distinct from model assumptions',
                   'A successful final scene does not automatically validate an inferred program or reasoning text',
                   'Never label a candidate false solely because it did not win an arbitrary preference score',
                   'Task/room variants from one source stay in one development family',
                   'Source-grounded labels and data rights precede any new training'],
        ordinary_objectives=dict(full_program_supervision='Conditional CE only when the program is actually labelled',
            partial_denotation_supervision='L=-log(sum_{p in consistent_labelled_set}pi(p|x)); standard MML, not new',
            unknown='Unknown observations do not filter a candidate',
            empty_set='Flag inconsistency or unsupported candidate coverage; do not invent a target',
            subjective_preference='Not present in this construction'),
        strong_controls=['Ordinary program/contract compiler plus the same geometric solver',
                         'Existing execution-signature hard/soft vote and labelled consistency filtering',
                         'Same source/information, action space, total budget and final complete-scene evaluation',
                         'Unstructured planning/feedback with the same verification tools'],
        exclusions=['No new learned filtering objective claimed','No hallucinated human rationale','No held-out benchmark answers used as training data','No whole-room generation or generalization gain established'])
    dump(root/'training_contract.json',training)
    theory='''From book mechanisms to a testable program-learning question,2026-11-11

Goal remains complete indoor3D generation with defensible algorithmic evidence. This stage provides direct reading and an executed ordinary control. It neither establishes a CVPR-level contribution nor terminates the research.

1. Source observations.
The previous complete Milan scene has verified resource reservations. Its independent layout can support small work materials and two meal regions. A smaller resource inventory also supports the compact persistent-material task by sharing the real table. The expansive persistent-material task fails a sound area necessary condition on that one table. All are one exposed research-transfer family, not independent houses. Material regions are planar proxies; full ergonomics and subjective design acceptance remain unknown.

2. Candidate program construction, not model diagnosis.
We deliberately construct four template programs: correct persistent materials with sequential people; omitted material persistence; forced distinct work/meal surfaces; and an unrequested concurrent worker. These are NOT claims about what the previous model inferred. In fact the old prompt prescribed persistence and actor Boolean values explicitly, so echoing them is not independent evidence of raw-language temporal interpretation. Their actual limited-scope execution matrix is:
  correct:[1,1,0]; omit_persistence:[1,1,1]; force_distinct:[1,0,0]; add_worker:[1,0,0].
Columns are existing independent compact completion, existing initial compact resource task, and existing initial expansive resource task. The last two differ in declared task dimensions. Do not call them same-utterance world interventions or3newscenes. No geometry, native assets, thresholds or old results are changed.

3. Identifiability.
Given hypotheses H and observations D={(x_j,y_j)}, the version space is V(D)={h:Eval(h,x_j)=y_j for every known label}. Unknown observations impose no equation. On the one positive completed-scene seed all four programs agree; its restricted pass label cannot identify the correct interpretation. Two additional known task labels reduce4->2->1. Exhaustive subset search establishes that BOTH probe columns are necessary to identify the correct hypothesis in this four-program construction. This is a finite consistency result, not a general theorem that two queries suffice.

4. Why unlabelled disagreement is not knowledge of the user's meaning.
Execution signatures partition hypotheses but do not label the correct class. An unlabeled method observes the same signatures whether the hidden intended program is h1 or h2; without an additional grounding assumption or reference label it cannot guarantee correctness in both situations. Candidate multiplicity or learned weights are priors, not guaranteed semantic evidence. This standard identifiability issue is already visible in semantic-parsing literature; it is not introduced here.
We execute the unmodified author's three NumPy vote functions at source3d6130a9a42450b6d95455081fe4ecb34b20b342. With uniform weights and the paper NLVR threshold0.8, hard voting retains the two[1,0,0]programs and drops the reference. Soft voting retains the reference and those two programs. This intentionally constructed pool illustrates the limitation the paper itself states. It is not a3Dbenchmark comparison against the full paper method or evidence that voting fails in realistic data distributions.
The compact column is a2:2tie, not a strict wrong majority. The exact source resolves it as false via np.round(.5) or a strict positive-margin test. A posthoc ordinary tie-abstaining control retains the reference and two alternatives. This explains an avoidable false rejection; it does not identify the correct semantics or establish a new abstention method. No extra hypotheses were added to manufacture a stricter majority result.

5. Training objective and data authority.
If complete program labels exist, ordinary CE is appropriate. With partial known input/output labels, ordinary maximum marginal likelihood over V(D) preserves remaining ambiguity: L=-log sum_{h in V(D)}pi_theta(h|x). This is standard semantic parsing, not a new loss. No unknown is silently treated as failure, no empty V(D) is filled with an invented expert program, and no raw final-scene success flag is promoted to correct expert reasoning. Similarly, explicit added-object cost labels and user preferences remain different supervision types. The current one-room construction is development evidence, not an adequate training corpus.

6. Candidate research algorithm, with ordinary components named plainly.
Given source brief U, full native scene S and common asset/tool evidence K:
(a) A policy proposes activity/resource programs P: actor identities, activity modes, reservation activation, object/role bindings, explicit source requirements and clearly marked inferred assumptions. Do not prefill all meaningful fields from the scoring answer.
(b) A common execution backend compiles each supported program into resource/geometry constraints and construction actions, returning actual witnesses, failures or unknowns while preserving source identities. The solver, not the policy, gets credit for its numerical work.
(c) Programme candidates are checked against source-grounded task labels and qualified execution suites. Distinct consistent interpretations remain distinct; an unlabeled majority is not ground truth. Counterexample generation may help select informative tests but cannot supply their semantic labels by itself.
(d) A learned policy can be trained only from admitted labels, then freely generate and update complete native scenes under its own proposals. Ordinary compilation/search, ordinary plans/feedback and the original model retain the same evidence, operations and total budget.
This is a concrete candidate pipeline; every component listed so far has strong prior and standard baselines. The unresolved possible contribution must lie in learning source-grounded3D design decisions that improve genuinely new complete rooms beyond those controls. No such gain is established by the current matrix or earlier two-skill allocation probe.

7. Nearest-work constraints.
Pasupat/Liang2016 already use execution equivalence and labelled fictitious worlds, including informative query selection. Lee et al2023 already use execution signatures and voting. STAGE already has fixed-observation semantic/action diagnosis, schemas and verified action selection; its paper also notes a strong unstructured baseline. HDSL/Function2Scene/ESG previously reviewed already cover programs, functional constraints and temporal grounding. Thus tool search, an evidence graph, conditional ranking, a schema, a selector, a new name or an extraCoT field is not sufficient novelty.

8. Book scope and authority.
Read all extracted chapter4text,PDF110-128 in this turn. Tool form, visibility, argument fidelity, query granularity and execution/review are separate design decisions. The current library has only two construction skills, so large-library tool discovery is not an evidenced bottleneck. The book's external product claims, model names and caching percentages were not verified or adopted as results. Its examples of quotas, approvals, messaging and spawning agents are document data, not authority to perform those actions. Current-session direct text coverage is117/312; a separate older cumulative ledger claims312text pages and is not silently imported as fresh reading. No claim of complete figure or external-reference verification.

9. Actual limits.
No model/API calls, no training, no new scene creation, no GPU launch, no simulated people or new material meshes. The matrix reuses qualified prior surface witnesses/certificates and complete native-state identities. The restricted true/false result is not complete-scene generation success. A full paper still requires new-room data with valid task labels, an actually different learnable mechanism, end-to-end construction and meaningful advantage over equally informed ordinary methods.
'''
    (root/'algorithm_and_theory.txt').write_text(theory)
    old=read(Path('runs/autodl/20261111_feedback_information_control/source_before_after.json'))['original_inputs']
    checked=[dict(path=x['path'],expected=x['before'],actual=sha(x['path'])) for x in old]
    assert all(x['expected']==x['actual'] for x in checked)
    dump(root/'source_verification.json',dict(protected_count=len(checked),all_protected_unchanged=True,checks=checked,
        own_input_checks=read(root/'inputs_after.json'),book_pdf_sha256=sha(book['source'])))
    # Source acquisition is not all-source reading. Keep only the original pure
    # function module and its license; other fresh exploratory copies are obsolete.
    cleanup_path=root/'new_source_cleanup.json'
    removed=read(cleanup_path)['removed'] if cleanup_path.exists() else []
    for name in ['ASD.py','iterative_train.py','baseline.py']:
        p=root/'source'/name
        if p.exists():removed.append(dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size));p.unlink()
    dump(root/'new_source_cleanup.json',dict(removed=removed,old_research_files_removed=0,
        why='Unneeded newly fetched exploratory source copies after bounded reading; retained URLs/hashes and required original vote module/license'))
    retention=[]
    for item in read(root/'source_manifest.json'):
        path=Path(item['path']);kept=path.exists()
        if kept:assert sha(path)==item['sha256']
        retention.append(dict(source_path=item['source_path'],source_sha256=item['sha256'],
            local_retained=kept,recovery_url=item['url'],reason='Required original function source/license' if kept else 'New exploratory copy removed after bounded reading; acquisition record retained'))
    dump(root/'source_retention_status.json',dict(acquisition_manifest_not_all_retained_files=True,files=retention))
    with (root/'method_difference.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['candidate_component','strong_prior_or_control','actual_this_stage','unproved'])
        w.writerows([
            ['Tool visibility/parameter fidelity','Ordinary ACI/tool engineering','Complete tools chapter text reading','Not a new3Dmethod'],
            ['Multi-input execution signatures','2016labelled worlds and2023execution filtering','Original three vote functions on4x3constructed matrix','No observed model distribution/generalization'],
            ['Task grounding and verification','STAGE schemas/action checks; existing functional generators','Explicitly separated source label from majority','No independent intent recovery'],
            ['Program learning','ConditionalCE and partial-labelMML','Concrete supervision admission contract, no training','No new objective or learned gain'],
            ['Whole scene delivery','Ordinary compiler/geometry/backend shared','Reused existing30/32object native-state evidence','No new whole-scene generation']])
    (root/'RUN_COMMANDS.txt').write_text('''CPU-only closed construction, actual executed:
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python -B scripts/autodl/audit_program_identifiability.py --root runs/autodl/20261111_tool_contract_semantics
This output already exists; the runner rejects implicit overwrite. No generation or training occurs.

Rebuild report/evidence only:
/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python -B scripts/autodl/report_program_semantics.py --root runs/autodl/20261111_tool_contract_semantics

Original author source revision and exact source paths/URLs/hashes are in source_manifest.json. Three vote functions only were AST-extracted with NumPy scope; no AllenNLP/Pandas/Torch module import or author training launcher executed.
The4x3matrix is a deliberately constructed typed-task control using existing full-native-scene resource evidence. It is not12newscenes,4model mistakes, a same-utterance3world benchmark or independent human preference data.
''')
    code=root/'code';code.mkdir(exist_ok=True)
    for name in ['audit_program_identifiability.py','report_program_semantics.py']:(code/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    with (root/'summary.csv').open() as f:csvrows=list(csv.reader(f))
    trs=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in csvrows[1:])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>用途程序、来源标签与完整场景监督</title><style>body{{font:16px/1.75 sans-serif;max-width:1080px;margin:40px auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #bbb;padding:7px}}aside{{background:#edf4f7;padding:15px}}</style>
<h1>不能把“场景检查通过”直接变成“设计程序正确”</h1><aside>本轮完成工具章19页提取文字的完整阅读，定向核查3篇近邻方法，并实际执行原作者3个投票函数。没有新模型调用、训练、场景生成或整场成功主张。</aside>
<p>四个程序是假设性对照，不是模型实测错误：持续资料＋顺序使用、漏掉资料持续占用、强制独立桌面、额外加入办公人。它们都能解释已经完成的独立桌椅场景，因此仅凭这个正例不能分清用途含义。实际复用的原生状态有30或32个记录，包含建筑；不是30或32件家具。</p>
<table><tr>{''.join('<th>'+html.escape(x)+'</th>' for x in csvrows[0])}</tr>{trs}</table>
<p>在这个明确构造的候选池中，原作者硬投票保留两个错误程序；软投票还保留正确程序，但未排除另外两个。尤其需要说明，小资料列是2:2平局，原代码将平局落为false，并不是严格错误多数。事后补充的普通“平局不筛”也能保住正确程序与两个候选。作者论文已承认质心不保证是真值，故这不是对其原实验的否定。加入两个已声明任务标签后，普通一致性过滤保留唯一正确程序；标签来自研究者明确的数值合同，不来自多数模型，也不是使用者偏好。</p>
<p>两项探针改变了资料尺寸，属于参数化任务族，而非“同一句话、只改物理世界”。本轮没有新建三维世界、注入模型故障或重新跑物理引擎；仅核验并消费此前真实桌面区域证据。</p>
<h2>算法与训练应怎样推进</h2><p>候选管线是：依据原文提出活动/对象程序→用共同后端实际施工并返回证据→按有依据的任务标签保留解释→最后在自己生成的程序上完成整场。强对照包含普通编译与搜索、原作者过滤、普通反馈和相同工具的无结构计划。条件CE、MML、程序表示和筛选本身都是已有机制；当前尚无可宣称的新算法优势。</p>
<p>书中工具发现、参数保真、执行与审核的划分可用于界定模型职责，但两种预写施工工具尚无“大工具库”问题。正文已读不代表所有图表和引用都核实：当前直接文本台账为117/312页；另有历史会话累计312页文本记录，没有将其混写成本轮通读。</p>
<h2>交付与一手来源</h2><ul><li><a href="algorithm_and_theory.txt">具体算法、可识别性推导、近邻与研究边界</a></li><li><a href="training_contract.json">训练标签准入和标准目标</a> / <a href="identifiability_result.json">实际运行结果</a> / <a href="summary.csv">完整对照表</a></li><li><a href="book_reading_ledger.json">逐页阅读记录</a> / <a href="book_notes.json">工具章思考</a> / <a href="reading_ledger.json">论文源码覆盖</a></li><li><a href="source_verification.json">579项旧源文件与输入校验</a> / <a href="evidence.zip">证据包</a></li></ul>
<p><a href="https://aclanthology.org/P16-1003.pdf">Pasupat与Liang2016</a> · <a href="https://aclanthology.org/2023.emnlp-main.425.pdf">执行过滤2023</a> · <a href="https://arxiv.org/html/2609.13458v1">STAGE</a>。这是一轮有实际运行的算法与监督准备，尚未实现跨新房间的完整生成和学习收益，研究目标继续进行。</p></html>'''
    (root/'stage_report.html').write_text(page)
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ['evidence.zip','delivery_manifest.json']]
    dump(root/'delivery_manifest.json',dict(files=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)],
        no_original_book_or_model_weights_redistributed=True,old_experiments_readonly=True,new_generated_rooms=0))
    with zipfile.ZipFile(root/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*sorted(files),root/'delivery_manifest.json']:z.write(p,p.relative_to(root))
    print(json.dumps(dict(book_pages=19,protected_files=len(checked),author_vote_functions_executed=3,
        retained_after_known_labels=result['label_anchored_filter'][-1]['remaining'],report=str(root/'stage_report.html'))))

if __name__=='__main__':main()
