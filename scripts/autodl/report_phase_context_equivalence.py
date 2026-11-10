"""Package the actual causal-condition audit and explicit reading boundaries."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--book-text-root", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "runs/autodl/20261111_phase_context_equivalence"
    result = json.loads((out / "actual_v4/result.json").read_text())
    if (result["units"] != 24 or result["split_components"] != 34 or
            result["comparisons"]["new_joint_prefix_split_versus_J"]["equal_context_tokens"] != 2943):
        raise ValueError("Report assumptions do not match actual results")

    download = json.loads((args.source_root / "download_manifest.json").read_text())
    for entry in download["files"]:
        if "sha256" in entry and sha(args.source_root / "plan_and_act" / entry["path"]) != entry["sha256"]:
            raise ValueError("Read source changed")
    write(out / "external_source_manifest.json", download)
    ranges = {
        "README.md": "whole",
        "LICENSE": "whole",
        "plan_and_act/cot/data_generation/act.py": [[45, 208], [212, 355]],
        "plan_and_act/cot/data_generation/plan.py": [[248, 335], [401, 454]],
        "plan_and_act/cot/inference/dynamic_plan.py": [[29, 450]],
        "plan_and_act/cot/inference/dynamic_act.py": [[350, 430]],
    }
    write(out / "reading_ledger.json", [
        dict(title="Plan-and-Act", url="https://arxiv.org/html/2503.09572v3",
             formal_venue="https://proceedings.mlr.press/v267/erdogan25a.html",
             read="Methods 3.1–3.4, 4.1–4.3; setup 5.1, results 5.2 and displayed 5.3.1; A.12; HTML text/tables, not all 44 PDF pages or figures",
             source_pin=download["pin"], source_read_ranges=ranges,
             main_overlap="Separate planning/execution, post-trajectory plan/reason labels, dynamic replanning and ordinary SFT sequences already exist",
             not_run=["model", "browser environment", "data generator", "training"],
             downloaded_training_data_or_weights=False),
        dict(title="Plan-and-Patch", url="https://arxiv.org/html/2610.10786v1",
             read="Main methods 3–4.3 and experiments 5.1–5.3, HTML text/tables; not full appendices or figure inspection",
             main_overlap="History-based repair-region selection, preserved prefix/suffix infilling, joint generation/repair training already exist",
             implementation_acquired_or_run=False,
             not_inferred="Not a completed 3D-scene method or evidence that our local generator benefits"),
    ])
    source = args.source_root / "plan_and_act/plan_and_act/cot/data_generation/act.py"
    tree = ast.parse(source.read_text())
    comparator_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "_same_action_str"]
    write(out / "source_observations.json", dict(
        pin=download["pin"], source_sha256=sha(source),
        teacher_receives_reference_action=True,
        observed_annotation_scope="Read first-action prompt and annotator code; not measured output corpus",
        same_action_comparator_calls_in_this_module=comparator_calls,
        caution="The comment about checking reference action is not an actual comparator invocation in this module; no empirical mismatch frequency is known. Do not turn this into a new repair project.",
        paper_declared_inference_excludes_reference_action=True,
        actual_model_or_source_code_execution=False))

    prior = json.loads((repo / "runs/autodl/20261111_critic_task_conditioning/book_reading_ledger.json").read_text())
    pages = list(range(149, 191))
    page_hashes = {str(n): sha(args.book_text_root / f"page_{n}.txt") for n in pages}
    covered = sorted(set(prior["covered_text_PDF_pages"]) | set(pages))
    if len(covered) != 212:
        raise ValueError("Cumulative reading count changed")
    write(out / "book_reading_ledger.json", dict(source=prior["source"],
        source_sha256=sha(Path(prior["source"])), PDF_pages=312,
        new_complete_text_PDF_pages=pages, extracted_text_sha256=page_hashes,
        covered_text_PDF_pages=covered, covered_count=len(covered),
        remaining_text_pages=sorted(set(range(1, 313)) - set(covered)),
        scope="Actual extracted text read; PDF149–159 finish chapter5, PDF160–190 chapter6. No page-image review or verification of every referenced product/paper claim.",
        document_instructions_executed=False,
        implications=["Task initiation, proposal and actual completion remain distinct",
                      "New observations and result ownership matter; ordinary execution facts do not establish new 3D learning"] ))
    write(out / "execution_attempts.json", [
        dict(attempt="v1", outcome="failed", error="Cannot uniquely locate the pinned image-budget source",
             retained_result_directory=False, model_forwards=0,
             fix="Explicit original helper path, verified against the prior pinned hash"),
        dict(attempt="v2", outcome="completed", result="actual_v2/result.json",
             note="32 components: two syntactic STOP records remained in A; positive objective conditions still equal"),
        dict(attempt="v3", outcome="completed", result="actual_v3/result.json",
             note="34 components: split by actual syntax while preserving immutable source-loss categories"),
        dict(attempt="v4", outcome="completed_final", result="actual_v4/result.json",
             note="Explicit decision stop-token metadata added; no extra independent samples or model runs"),
    ])
    write(out / "algorithm_contract.json", dict(
        goal="Complete editable indoor scene generation under original user requirements",
        candidate="Conditional application, adaptation or refusal of source design principles",
        currently_verified_new_learning_algorithm=False,
        same_prefix_split_is_control_not_candidate=True,
        student_objective="Existing immutable weighted causal CE; not a new loss",
        next_data_requirement="Independently supported applicability across task/geometry changes, with room/source-separated evidence and complete execution outcomes",
        controls=["A with identical knowledge", "AUX", "continuous J", "phase-prompt P", "same-prefix split identity C", "ordinary task-aware prompting, feedback, retrieval/search and prior conditional-preference learning"],
        cannot_infer=["P/J difference identifies phase factorization", "Reference D is deploy-time input", "Extra rationale tokens prove scene quality", "Five available view sets grant full-corpus training"],
        neural_training_admitted=False, new_GPU_trial_admitted=False,
        paid_API_calls=0, model_forwards=0, optimizer_updates=0, complete_generated_scenes=0,
        next_work="Read remaining book and obtain/check real conditional-design decision evidence; do not relaunch the closed trial quota or tune 289/404"))

    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comparison", "unit_id", "positive_tokens",
            "equal_context_tokens", "different_context_tokens", "by_stage"], lineterminator="\n")
        writer.writeheader()
        for name, comparison in result["comparisons"].items():
            for row in comparison["units"]:
                writer.writerow(dict(comparison=name, **{k: json.dumps(v) if k == "by_stage" else v for k, v in row.items()}))

    comparison_rows = "".join("<tr><td>" + html.escape(name) + "</td>" +
        "".join(f"<td>{comparison[k]}</td>" for k in ("positive_tokens", "equal_context_tokens", "different_context_tokens")) + "</tr>"
        for name, comparison in result["comparisons"].items())
    text = (out / "algorithm_revision_and_theory.txt").read_text()
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练条件实审：连续CoT与阶段拆分</title><style>body{max-width:1080px;margin:36px auto;padding:0 20px;background:#fafbfc;color:#202b36;font:17px/1.7 system-ui}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd4df;padding:9px;text-align:left}th{background:#e8eef6}pre{white-space:pre-wrap;font:15px/1.8 system-ui}.note{padding:16px;background:#edf3fa;border-left:4px solid #426e9d}a{color:#235c96}</style>
<h1>阶段拆分的收益，必须与提示和普通CoT分开</h1>
<p class="note">实际完成：24个既有局部动作的因果条件审计、34组件的可读取拆分对照、五组原视图的CPU处理器/位置编码验证。没有新生成房间、神经前向或训练；尚未建立新方法优势。</p>
<h2>不可变的2943个正监督token，对应条件是否相同</h2>
<table><tr><th>对照</th><th>正token</th><th>相同条件</th><th>不同条件</th></tr>''' + comparison_rows + '''</table>
<p>旧P增加了阶段system指令；移除后两个EOS仍受额外换行影响。新C保留J原始前缀，文本条件全部匹配。上述是token条件数，不是独立场景或成功数。</p>
<h2>当前能与不能证明的事</h2>
<p>相同目标、系数、完整条件和计算图下，损失项重分组不创造新学习目标。五个单元进一步通过实际图像张量与mRoPE比较；其余19个缺原视图，未获多模态准入。一次和两次generate的采样、缓存、停止及成本等价尚未验证。</p>
<p>继续检验的候选是对设计原则的条件应用与拒用，必须超过相同知识下的普通CoT、明确需求提示、反馈/搜索。当前数据不能识别它的跨场景收益，因此本轮没有申请GPU试跑或训练。</p>
<h2>新增阅读及边界</h2>
<p><a href="https://arxiv.org/html/2503.09572v3">Plan-and-Act</a>的已读方法/公开源码已有规划执行分工与轨迹后标注；<a href="https://arxiv.org/html/2610.10786v1">Plan-and-Patch</a>的已读方法已有局部计划重写。未运行两者。书本轮实际读PDF149–190，累计212/312页；尚未通读剩余100页。</p>
<h2>证据与可运行代码</h2>
<p><a href="actual_v4/result.json">最终逐项结果</a> · <a href="actual_v4/matched_joint_prefix_split.jsonl.gz">34条对照组件</a> · <a href="summary.csv">72条逐单元比较</a> · <a href="reading_ledger.json">论文/源码阅读范围</a> · <a href="book_reading_ledger.json">书本页码</a> · <a href="source_before_after.json">旧源文件哈希</a> · <a href="evidence.zip">证据包</a></p>
<details><summary>完整算法与条件性理论</summary><pre>''' + html.escape(text) + '''</pre></details></html>'''
    (out / "stage_report.html").write_text(page)

    old = json.loads((repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json").read_text())
    original = []
    for row in old["original_inputs"]:
        after = sha(Path(row["path"]))
        original.append(dict(path=row["path"], before=row["before"], after=after, same=after == row["before"]))
    if not all(r["same"] for r in original):
        raise ValueError("Protected old source changed")
    write(out / "source_before_after.json", dict(count=len(original), all_originals_unchanged=True, original_inputs=original))
    code = [repo / "scripts/autodl" / n for n in ("audit_phase_context_equivalence.py", "report_phase_context_equivalence.py")]
    write(out / "code_manifest.json", [{"path": str(p.relative_to(repo)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in code])
    files = [p for p in out.rglob("*") if p.is_file()]
    write(out / "manifest.json", [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(files)])
    with zipfile.ZipFile(out / "evidence.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files + [out / "manifest.json"]):
            archive.write(path, path.relative_to(out))
        for path in code:
            archive.write(path, "code/" + path.name)
    write(out / "delivery_verification.json", dict(protected_files=len(original), all_originals_unchanged=True,
        result="actual_v4/result.json", all_results_in_archive=True,
        evidence_bytes=(out / "evidence.zip").stat().st_size, evidence_sha256=sha(out / "evidence.zip"),
        actual_model_or_training_run=False, research_goal_complete=False))
    print(json.dumps(dict(out=str(out), protected=len(original), unchanged=True,
                         evidence_bytes=(out / "evidence.zip").stat().st_size)))


if __name__ == "__main__":
    main()
