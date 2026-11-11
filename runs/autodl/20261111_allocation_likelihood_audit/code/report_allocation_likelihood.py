"""Package the closed conditional-action likelihood audit, without more inference."""
import argparse
import csv
import difflib
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import zipfile

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    protocol=read(root/'protocol.json');scores=read(root/'likelihood_scores.json');enc=read(root/'encoding_preflight.json')
    assert scores['status']=='completed' and scores['teacher_forwards']==6
    rows=[]
    for c in protocol['contexts']:
        x={z['candidate']:z for z in scores['rows'] if z['context']==c['id']};s,i=x['shared'],x['independent']
        check=read(root/(c['id']+'_causal_check.json'));assert check['within_1e_2']
        delta=s['sum_logp']-i['sum_logp'];token=s['first_different_token_logp']
        rows.append(dict(context=c['id'],shared_sum_logp=s['sum_logp'],independent_sum_logp=i['sum_logp'],
            shared_mean_logp=s['mean_logp'],independent_mean_logp=i['mean_logp'],
            shared_minus_independent_logp=delta,two_string_normalized_shared_score=1/(1+math.exp(-delta)),
            first_different_token_log_odds=token['shared']-token['independent'],
            remaining_target_log_odds_contribution=delta-(token['shared']-token['independent']),
            shared_target_tokens=s['target_tokens'],independent_target_tokens=i['target_tokens'],
            conditional_format='Upstream no-thinking SFT tool serialization, not exact prior generated whitespace',
            optimal_count_in_context=c['feedback_contains_optimal_count'],same_prefix_logit_error=check['same_prefix_token_logp_max_difference']))
    indexed={x['context']:x for x in rows};base=indexed['compact_materials_turn_0'];large=indexed['expansive_materials_turn_0'];feedback=indexed['compact_materials_turn_1']
    left=read(root/'compact_materials_turn_0_request.json');right=read(root/'expansive_materials_turn_0_request.json')
    l=left[1]['content'][-1]['text'];r=right[1]['content'][-1]['text']
    edits=[dict(operation=tag,left=l[i:j],right=r[k:v],left_offset=i,right_offset=k) for tag,i,j,k,v in difflib.SequenceMatcher(None,l,r,autojunk=False).get_opcodes() if tag!='equal']
    assert [(e['left'],e['right']) for e in edits]==[('3','8'),('2','5')]
    assert left[0]==right[0] and left[1]['content'][:-1]==right[1]['content'][:-1]
    change=dict(initial_context_character_changes=edits,all_images_and_other_message_fields_exact=True,
        size_change_log_odds_delta=large['shared_minus_independent_logp']-base['shared_minus_independent_logp'],
        size_change_first_token_log_odds_delta=large['first_different_token_log_odds']-base['first_different_token_log_odds'],
        feedback_log_odds_delta=feedback['shared_minus_independent_logp']-base['shared_minus_independent_logp'],
        feedback_first_token_log_odds_delta=feedback['first_different_token_log_odds']-base['first_different_token_log_odds'],
        evidence='Deterministic conditional score changes in this fixed model/serialization. Not calibrated semantic probability or held-out effect.')
    dump(root/'conditional_effect.json',change)
    with (root/'summary.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
    old=read(Path('runs/autodl/20261111_feedback_information_control/source_before_after.json'))['original_inputs']
    verified=[dict(path=x['path'],expected=x['before'],actual=sha(x['path'])) for x in old]
    assert all(x['expected']==x['actual'] for x in verified)
    sources=[dict(path=protocol[k],expected=protocol[k+'_sha256'],actual=sha(protocol[k])) for k in ['source_sft','source_model_forward']]
    assert all(x['expected']==x['actual'] for x in sources)
    request_checks=[dict(path=x['source_request'],expected=x['source_sha256'],actual=sha(x['source_request'])) for x in protocol['contexts']]
    assert all(x['expected']==x['actual'] for x in request_checks)
    encoded_by_id={x['context']:x for x in enc['rows']};input_checks=[]
    for c in protocol['contexts']:
        original=read(c['source_generation_receipt']);current=encoded_by_id[c['id']]['prompt_encoding']
        match={k:original[k]==current[k] for k in ['images','input_tokens','input_shapes','chat_template_sha256','max_pixels','max_input_tokens']}
        assert all(match.values()),'Do not infer identical model input from file paths alone'
        input_checks.append(dict(context=c['id'],matches_original_generation_receipt=match))
    dump(root/'source_verification.json',dict(protected_count=len(verified),all_protected_unchanged=True,
        protected=verified,source_code=sources,exact_saved_requests=request_checks,
        full_model_prompt_and_image_bytes=input_checks))
    ledger=[dict(title='Scene-Designer captured SFT dataset source',
        location=protocol['source_sft'],sha256=protocol['source_sft_sha256'],
        read='Pure _build_messages function206-294; StepwiseSFTDataset.__getitem__306-373 and collator375-420. No whole-module gateway import.',
        used='Actual AST-extracted tool-only no-thinking serializer; exact system/user prompts matched all three saved requests.',
        boundary='The custom allocate_resources schema is not the native scene Actor tool set. This does not establish training-distribution parity.'),
        dict(title='Pinned installed Qwen3-VL conditional generation source',location=protocol['source_model_forward'],sha256=protocol['source_model_forward_sha256'],
        read='Forward1315-1370, logits_to_keep slicing1361-1362, prefill/M-RoPE1176-1210.',
        used='Teacher-forced causal logits; no cache or gradient. Exactly target_length+1 output positions prevent full prompt-position vocabulary tensors.',
        boundary='FP16 V100 compatibility, not exact author BF16 inference or training reproduction.'),
        dict(title='HDSL published AACL-IJCNLP2026 version',primary_url='https://aclanthology.org/2026.aacl-long.247/',
        paper_url='https://aclanthology.org/2026.aacl-long.247.pdf',publication='Proceedings dates6-10Nov2026,pp5519-5538',
        actual_reading='This turn publication metadata and targeted PDF methods3.1-3.3, Fig2/Table1; previous arXiv reading retained separately.',
        finding='Addressable hierarchical programs, bounded generation and local repair are direct prior; program syntax alone is not a new contribution.',
        not_done='No fresh full20page reading, official code execution or benchmark reproduction.'),
        dict(title='ESG: Generating Physically Consistent Dynamic3D Scenes from Text Descriptions',primary_url='https://arxiv.org/html/2609.15392v1',
        version='v1,14Sep2026',actual_reading='Methods3.1-3.3; AppendixC.4 timeline; D.1 asset prompt and D.2 static layout prompt. No official code acquired/executed.',
        finding='Joint spatial/event representation and event-constrained optimization are already explicit. A timeline or schema cannot alone establish novelty here.',
        boundary='Its dynamic event task differs from the current static shared-table proxy; no claim that it solves or fails our task.'),
        dict(title='Function2Scene official project and previously acquired source',primary_url='https://function2scene.github.io/',
        actual_reading='Official project functional constraint/tool overview revisited; reused Nov03 archive inventory rather than claiming another backend execution.',
        finding='Persona/activity-derived constraints and tool-driven refinement are prior. The previously acquired29 listings are contracts, not a runnable released backend.',
        not_done='No author system inference, full public data reuse-license verification or independent preference study.')]
    dump(root/'reading_ledger.json',dict(entries=ledger,new_novelty_claim=False))
    theory='''Conditional action scores, training target and source qualification,2026-11-11

Research goal remains complete text-conditioned editable indoor scene generation. This audit does not complete that goal. It provides new actual checkpoint evidence that changes how the earlier identical choices should be interpreted.

1. Controlled inputs and scope.
Use precisely the three closed requests from the previous furnished Milan allocation stage: compact materials first turn, its oracle-minimum-assisted feedback turn, and expansive materials first turn. Initial compact versus expansive messages differ at only two character positions: work dimensions0.3x0.2 become0.8x0.5m. System, scene state, five original images and all other fields are identical. One exposed research-transfer room and artificial numeric variants are not multiple held-out rooms or independent people.

2. Native supervision serialization.
AST-extract the actual upstream _build_messages function. For each saved request, create exactly two explicit tool-call targets with no invented reasoning: shared and independent bindings. Rebuild the system and user prefix through that original function and require exact equality with the saved request. Apply the checkpoint's native chat template with no-thinking tool_calls assistant fields. This gives90target tokens for shared and87for independent. The target serialization is upstream SFT syntax, which differs in spaces/newlines from the earlier83token free-generated response. Therefore these are controlled training-representation scores, not a replay of the exact old greedy decoding history.

3. Exact causal score alignment.
Let P be prompt length, T target length, N=P+T. Query the conditional model on the complete image+text sequence with use_cache=False and no gradients. Request only logits_to_keep=T+1: positions P-1,...,N-1. Remove the final unused position, obtaining predictors P-1,...,N-2 for targets P,...,N-1. Sum token log probabilities to get ell(A|X), and also report ell/T. This is ordinary assistant-only conditional cross-entropy scoring, aligned to the original dataset's labels[:P]=-100 behavior for these individual samples. It is not a new objective and does not reproduce multi-sample training normalization or any optimizer update.

4. Local branch and numerical qualification.
The two target sequences share17assistant tokens before their first different token, 'meal' versus 'new'. Score both token alternatives after that IDENTICAL prefix. For each context, the two independent target forwards give exactly the same first-branch log probabilities, confirming the implemented causal-prefix check at this numerical precision. Future teacher-forced target tokens do not supply extra evidence to the earlier branch under the causal mask. This does not validate all attention implementations or the original paper system.

5. What identical choices can and cannot mean.
Define Delta(X)=ell(shared|X)-ell(independent|X). A fixed deterministic policy can choose the same action in two contexts even when their scores differ; identical output alone does not prove conditional independence. Moreover token-greedy output is not generally the maximum-probability complete program. Here the fixed-SFT-representation initial size change shifts Delta by about-0.108nats, and first-branch log odds by-0.109375, toward the independent resource. The small-task optimum-count feedback shifts Delta by about+0.043nats toward sharing, while independent remains preferred. These are observed conditional response changes, not proof of correct geometric reasoning, sufficient instruction following, or an independent Agent contribution.

6. Avoid invalid probability claims.
q=exp(ell_shared)/(exp(ell_shared)+exp(ell_independent)) is only a normalized score for these TWO EXACT token strings. It is not probability that sharing is feasible, all semantically equivalent programs, human preference or full-scene success. Unequal target lengths and serialization affect full program scores; per-token mean and same-prefix first-token odds are reported separately. The paired size-change comparison fixes both candidate strings and all other input fields, but it still concerns one checkpoint and one exposed scene. No statistical population conclusion follows.

7. Learning implication and strong control.
The existing ordinary geometry compiler supplies feasibility witnesses/certificates and an explicit added-furniture objective. Those are separate from subjective preference. Model score changes need not improve actual construction. A new Critic or extra CoT supervision is not justified simply by identical final choices. A valid next learned-method comparison must preserve source brief semantics, construction actions, asset access, numerical checks and total compute, and compare against ordinary activity-program compilation plus joint resource/geometry optimization. If that ordinary pipeline handles the target cases, training an extra selector has not established research value. Synthetic cost labels here do not become expert intent or human rationale by placing them in an SFT record.

8. Closest methods and remaining gap.
Published HDSL already makes local programs checkable/addressable; Function2Scene already derives functional constraints; ESG already represents/grounds spatial and temporal events. This stage neither excludes them by naming new fields nor claims they fail this shared-table task. The unresolved research question is whether learned, source-grounded interpretation of realistic use requirements can improve complete-room generation beyond an equally informed ordinary compiler/search at acceptable total cost. We still lack cross-room qualified supervision and a demonstrated new decision mechanism. Do not invent a paper novelty or begin unapproved training to make this audit look like one.
'''
    (root/'algorithm_and_theory.txt').write_text(theory)
    costs=dict(teacher_forwards=scores['teacher_forwards'],model_generate_calls=0,new_scenes=0,training_updates=0,paid_API_calls=0,
        wall_seconds=scores['wall_seconds'],peak_rss_kib=scores['peak_rss_kib'],
        peak_cuda_allocated_bytes=max(x['peak_cuda_allocated_bytes'] for x in scores['rows']),
        total_prompt_token_positions=sum(x['prompt_tokens'] for x in scores['rows']),
        total_teacher_target_tokens=sum(x['target_tokens'] for x in scores['rows']),network_attempts=scores['network_attempts'],
        total_instance_bill_unknown=True,no_new_weights_or_data_downloads=True,
        timer_scope='Weight load and six teacher-forced steps; excludes initial processor-only preflight and report preparation',
        summed_teacher_step_wall_seconds=sum(x['elapsed_seconds'] for x in scores['rows']))
    dump(root/'cost_ledger.json',costs)
    code=root/'code';code.mkdir(exist_ok=True)
    for name in ['audit_allocation_likelihood.py','report_allocation_likelihood.py']:(code/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    for name in ['likelihood_model.stdout','likelihood_model.stderr']:
        src=Path('/dev/shm/milan_completion_20261111')/name
        if src.exists():shutil.copyfile(src,root/name)
    (root/'RUN_COMMANDS.txt').write_text('''Closed actual likelihood audit; do not launch again into this root.
Prepare (CPU, completed before outcomes):
  <eval_python> -B scripts/autodl/audit_allocation_likelihood.py --prepare --root runs/autodl/20261111_allocation_likelihood_audit --previous runs/autodl/20261111_activity_resource_program
Processor-only preflight, no weights (completed):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <existing_model_python> -B scripts/autodl/audit_allocation_likelihood.py --encode-only --root runs/autodl/20261111_allocation_likelihood_audit
Actual bounded6teacher forwards (completed):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout --signal=TERM --kill-after=10s 360 <existing_model_python> -B scripts/autodl/audit_allocation_likelihood.py --root runs/autodl/20261111_allocation_likelihood_audit
Existing model Python: /root/autodl-tmp/agent_method_preparation_20261027/work/v100_runtime_20261104/runtime/bin/python
Existing eval Python: /root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python
Report only, no model execution:
  <eval_python> -B scripts/autodl/report_allocation_likelihood.py --root runs/autodl/20261111_allocation_likelihood_audit
Original image bytes remain in the pinned Git revision in the previous stage view_manifest.json; rehydrate them to the recorded RAM paths if the container has restarted, verifying hashes. No remote images or APIs are allowed by this audit.
''')
    trows=''.join('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ['context','shared_sum_logp','independent_sum_logp','two_string_normalized_shared_score','first_different_token_log_odds'])+'</tr>' for row in rows)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>条件动作概率：实际模型与训练源码核查</title><style>body{{font:16px/1.7 sans-serif;max-width:1080px;margin:40px auto}}td,th{{border:1px solid #bbb;padding:8px}}table{{border-collapse:collapse;font-size:14px}}aside{{background:#eef3f8;padding:15px}}pre{{white-space:pre-wrap}}</style>
<h1>相同动作不等于没有使用条件</h1><aside>本轮实际完成6次教师强制前向，耗时{scores['wall_seconds']:.3f}秒；没有生成新回答、训练、付费API或新场景。使用现有联合SFT检查点，不是作者完整Actor复现。</aside>
<p>保持原始5视图、完整场景状态和提示不变，仅资料尺寸的两位字符不同。对同一对原生SFT格式工具动作计算条件对数概率。共享/独立目标分别90/87token，所有提示前缀严格匹配，三个相同前缀数值检查误差均为0。</p>
<table><tr><th>保存上下文</th><th>共享logp</th><th>独立logp</th><th>两固定字符串中的共享归一化分数</th><th>首次差异token的共享相对log-odds</th></tr>{trows}</table>
<p>小资料→大资料时，固定动作分数向独立工作面变化；最优数量反馈使分数略向共享变化，但都未改变偏好的动作。上一轮三个输出相同，不能据此说模型完全忽略条件。此处归一化分数不是人类偏好、几何可行率或全部同义程序的概率。</p>
<h2>源码和理论</h2><p>直接抽取作者SFT消息构造函数，验证assistant监督位置；实际模型只返回T+1个logit位置，按自回归偏移评分，避免创建整个长提示的词表张量。工具结构是自定义两施工技能，目标空白格式与旧自由生成文本不同，因此不把该评分当旧生成历史逐token重放。</p>
<p><a href="https://aclanthology.org/2026.aacl-long.247/">HDSL正式版本</a>、<a href="https://function2scene.github.io/">Function2Scene</a>、<a href="https://arxiv.org/html/2609.15392v1">ESG</a>分别覆盖了结构程序、功能约束和时序事件求解。它们限定本项目不能仅凭新增中间结构或接口宣称新颖；本轮没有复现其完整系统，也没有宣称它们不能解决本例。</p>
<h2>可保留与尚未完成</h2><p>保留实际场景强基线、上下文消费审计和分开的可行性/成本监督。尚未得到跨房间训练数据、独立使用者偏好、新决策机制或完整生成优势。当前证据不足以启动新Critic训练或宣称CVPR主创新，研究目标继续进行。</p>
<ul><li><a href="algorithm_and_theory.txt">完整评分算法、训练mask、推导与边界</a></li><li><a href="summary.csv">实测数值</a> / <a href="likelihood_scores.json">逐token原始结果</a> / <a href="conditional_effect.json">两个字符的输入差异与分数变化</a></li><li><a href="reading_ledger.json">实际论文与源码阅读记录</a> / <a href="source_verification.json">579项保护源和原始请求哈希</a></li><li><a href="cost_ledger.json">资源与耗时</a> / <a href="evidence.zip">证据包与代码</a></li></ul></html>'''
    (root/'stage_report.html').write_text(page)
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ['evidence.zip','delivery_manifest.json','exclusive_model.lock']]
    dump(root/'delivery_manifest.json',dict(files=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)],
        native_model_source_unchanged=True,generated_answers=0,training_updates=0,new_scene_success_not_claimed=True))
    with zipfile.ZipFile(root/'evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*sorted(files),root/'delivery_manifest.json']:z.write(p,p.relative_to(root))
    print(json.dumps(dict(rows=len(rows),files=len(files),source_checks=len(verified),report=str(root/'stage_report.html'))))

if __name__=='__main__':main()
