"""Build a compact stage report from completed source/scorer evidence."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = repo / "runs/autodl/20261111_sceneexplorer_objectives"
    scorer = json.loads((out / "actual_v1/result.json").read_text())
    cache = json.loads((out / "cache_metric_audit_v1/result.json").read_text())
    book = json.loads((out / "book_reading_ledger.json").read_text())
    if not scorer["independent_reference_agrees"] or book["covered_count"] != 312 or not cache["unchanged"]:
        raise ValueError("Completed evidence does not support the report")
    protected = json.loads((repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json").read_text())
    entries = []
    for row in protected["original_inputs"]:
        path = Path(row["path"])
        now = digest(path) if path.is_file() else None
        entries.append(dict(path=str(path), before=row["before"], after=now, same=now == row["before"]))
    if len(entries) != 579 or not all(v["same"] for v in entries):
        raise RuntimeError("Protected inputs changed or audit scope changed")
    (out / "source_before_after.json").write_text(json.dumps(dict(
        manifest_source="runs/autodl/20261111_feedback_information_control/source_before_after.json",
        manifest_source_sha256=digest(repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json"),
        original_inputs=entries, unchanged=True, count=len(entries)), ensure_ascii=False, indent=2) + "\n")
    theory = (out / "algorithm_and_theory.txt").read_text()
    title = "全书文字阅读完成；功能目标、训练目标与缓存分支的实审"
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;color:#18242c;max-width:1060px;margin:32px auto;padding:0 22px}}
table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #bdc9d0;padding:10px;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui,sans-serif;background:#f4f7f8;padding:22px}}
a{{color:#075e9b}}.note{{border-left:4px solid #c98912;padding-left:16px}}</style>
<h1>{title}</h1><p>2026-11-11 · 完整室内三维生成研究持续进行。</p>
<p>本轮完成了剩余80页书本文字阅读、SceneExplorer作者稿完整文字与关键源码阅读，并实际运行作者样例的原始评分入口。另读可编辑/可组合KV论文的方法与限制，审阅其公开图像实验指标。算法说明已给出训练式与理论条件；尚未得到新策略的完整场景优势。</p>
<table><tr><th>实际完成</th><th>证据</th><th>不能推出</th></tr>
<tr><td>全书提取文字累计312/312页；本轮PDF29–108</td><td><a href="book_reading_ledger.json">逐页哈希与覆盖</a></td><td>所有图表、引用论文、产品行为或实验结果均已核实</td></tr>
<tr><td>SceneExplorer稿件16个PDF页文字与关键实现</td><td><a href="reading_ledger.json">实际阅读范围</a> · <a href="primary_sources_manifest.json">固定来源</a></td><td>官方完整系统、运动调度或功能评价已复现</td></tr>
<tr><td>原始sample3六模块评分={scorer['original_score']}；独立复算一致，{scorer['seconds']:.3f}秒</td><td><a href="actual_v1/result.json">原始运行</a> · <a href="actual_v1/inventory.csv">实例与状态</a></td><td>64种状态组合都生成了，或原样例完整三维质量成功</td></tr>
<tr><td>两份缓存图像脚本指标已实审，{cache['seconds']:.3f}秒</td><td><a href="cache_metric_audit_v1/result.json">AST定位</a></td><td>神经缓存实测，或完整动作/工具参数严格一致</td></tr>
<tr><td>579项旧输入重新计算哈希未变</td><td><a href="source_before_after.json">逐文件记录</a></td><td>所有旧实验或物理引擎均重新运行</td></tr></table>
<p class="note">新增模型前向、训练更新、付费API调用、原生整场生成均为0。原始PDF、HTML及第三方源码只留RAM，Git不分发这些全文；本轮没有删除旧实验或权重。</p>
<p>运行命令见 <a href="COMMANDS.txt">COMMANDS.txt</a>；固定版公开代码的局部差异见 <a href="actual_v1/source_findings.json">source_findings.json</a>。样例分数两项都为0，所以本例不能测出阻挡系数0.5与论文式(2)差别对结果的影响。</p>
<p>缓存论文：<a href="https://arxiv.org/html/2606.17107v1">Models Take Notes at Prefill</a>；场景前作：<a href="https://researchportal.bath.ac.uk/en/publications/sceneexplorer-an-interactive-system-for-expanding-scheduling-and-/">SceneExplorer正式记录</a>。</p>
<h2>算法与理论的具体说明</h2><pre>{html.escape(theory)}</pre>
<p><a href="evidence.zip">紧凑交付包</a> · <a href="manifest.json">文件哈希</a>。研究目标未完成，不能据此承诺CVPR成果。</p></html>'''
    (out / "stage_report.html").write_text(report)
    code_paths = [repo / "scripts/autodl" / name for name in (
        "qualify_sceneexplorer_objectives.py", "audit_cache_decision_semantics.py", "report_sceneexplorer_objectives.py")]
    code = [{"path": str(p.relative_to(repo)), "sha256": digest(p), "bytes": p.stat().st_size} for p in code_paths]
    (out / "code_manifest.json").write_text(json.dumps(code, indent=2) + "\n")
    package_files = sorted(p for p in out.rglob("*") if p.is_file()
                           and p.name not in ("manifest.json", "evidence.zip", "delivery_verification.json"))
    manifest = [{"path": str(p.relative_to(out)), "sha256": digest(p), "bytes": p.stat().st_size} for p in package_files]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in package_files + [out / "manifest.json"]:
            z.write(path, str(path.relative_to(out)))
        for path in code_paths:
            z.write(path, str(path.relative_to(repo)))
    print(json.dumps(dict(protected_sources_unchanged=len(entries), files=len(manifest),
                          report=str(out / "stage_report.html"), zip_bytes=(out / "evidence.zip").stat().st_size,
                          zip_sha256=digest(out / "evidence.zip"))))


if __name__ == "__main__":
    main()
