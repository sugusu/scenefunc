"""Assemble the privileged-context preparation evidence; never run a model."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    repo = a.repo.resolve()
    out = repo / "runs/autodl/20261111_privileged_training_objectives"
    result = json.loads((out / "actual_v1/result.json").read_text())
    batch = json.loads((out / "native_batch_contract.json").read_text())
    prior = repo / "runs/autodl/20261111_tool_decision_ownership/actual_audit_v1"
    calls = json.loads((prior / "tool_calls.json").read_text())
    decisions = json.loads((prior / "policy_decisions.json").read_text())
    assert sha(prior / "tool_calls.json") == batch["calls_sha256"]
    assert sha(prior / "policy_decisions.json") == batch["decisions_sha256"]
    assert len(calls) == 35 and len(decisions) == 19
    assert len(batch["commands"]) == len(calls)
    assert len([d for d in decisions if d["calls"] > 1]) == batch["multi_command_responses"] == 8
    assert sum(max(0, d["calls"] - 1) for d in decisions) == batch["commands_after_a_same_response_command"] == 17
    for derived, original in zip(batch["commands"], calls):
        assert (derived["packet"], derived["batch_index"], derived["tool_name"]) == (
            original["packet"], original["batch_index"], original["name"])
        assert derived["task_correctness_label"] is None
    protected_path = repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json"
    protected = json.loads(protected_path.read_text())
    rows = []
    for row in protected["original_inputs"]:
        path = Path(row["path"])
        digest = sha(path) if path.is_file() else None
        rows.append(dict(path=str(path), before=row["before"], after=digest, same=row["before"] == digest))
    if len(rows) != 579 or not all(v["same"] for v in rows):
        raise RuntimeError("A protected input changed")
    (out / "source_before_after.json").write_text(json.dumps(dict(
        protected_manifest_sha256=sha(protected_path), count=579, unchanged=True,
        original_inputs=rows, separate_cleanup_exception="source_download_cache_cleanup.json; target not in579protected inputs"),
        ensure_ascii=False, indent=2)+"\n")
    proof = (out / "algorithm_and_theory.txt").read_text()
    cleanup = json.loads((out / "source_download_cache_cleanup.json").read_text())
    title = "特权上下文、KL方向与原生命令批次：实际核查和训练合同"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font:17px/1.7 system-ui,sans-serif;max-width:1080px;margin:30px auto;padding:0 20px;color:#162631}}td,th{{padding:10px;border:1px solid #bdc9d0;text-align:left}}table{{border-collapse:collapse;width:100%;font-size:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui;background:#f3f6f8;padding:20px}}a{{color:#075d99}}.note{{border-left:4px solid #b87b16;padding-left:15px}}</style>
<h1>{title}</h1><p>2026-11-11 · 完整室内三维场景生成研究 · 本轮未进行模型推理或训练。</p>
<p>这轮新增三项直接相关的训练近邻审阅，实际执行微软原始KL核，并把新机制的前提与已有原生记录对齐。结论不是“蒸馏无效”，而是不能混用KL方向、词表代理、命令mask与信息条件，再将收益归给设计原则。</p>
<table><tr><th>实际取得</th><th>证据</th><th>范围</th></tr>
<tr><td>OPCD、Instruct Not Answer、ECOS方法与指定实验/附录</td><td><a href="reading_ledger.json">逐项阅读范围</a> · <a href="method_difference.csv">方法差异</a></td><td>并非三篇全部引用、图片与训练复现</td></tr>
<tr><td>6次原始数学核、2次有限向量取梯度，{result['seconds']:.3f}秒</td><td><a href="actual_v1/result.json">实际结果</a></td><td>概率向量为人造数学见证；0个模型实例、0次更新、0个场景</td></tr>
<tr><td>19次响应、35条旧工具记录；8个多命令响应、17条同批后续命令</td><td><a href="native_batch_contract.json">原生适配记录</a> · <a href="native_batch_contract.csv">完整表</a></td><td>一条旧部分轨迹；不是新生成或专家数据</td></tr>
<tr><td>579个受保护输入哈希未变</td><td><a href="source_before_after.json">核验</a></td><td>不等于重跑所有旧实验</td></tr>
<tr><td>清理约{cleanup['bytes']/1e6:.2f}MB旧公开PDF下载缓存；两版独有场景归档保留</td><td><a href="source_download_cache_cleanup.json">清理与恢复记录</a> · <a href="old_archives_retained.json">归档保留依据</a></td><td>旧阅读/数据/图页保留，旧下载清单需恢复缓存后复核</td></tr></table>
<p class="note">主要理论结果是标准KL目标的明确区分：完整KL、截取部分和与Top-k条件化目标不同；固定观察下，不同隐藏教师上下文的前向KL最优点为算术混合，反向KL最优点为几何混合。这些不是新定理、偏好标签或CVPR效果证据。</p>
<p>直接来源：<a href="https://arxiv.org/html/2602.12275v1">OPCD</a>、<a href="https://arxiv.org/html/2609.32201v1">Instruct Not Answer</a>、<a href="https://arxiv.org/html/2611.12232v1">ECOS</a>。来源字节哈希与官方代码提交见<a href="primary_sources_manifest.json">来源清单</a>。</p>
<p>可执行命令：<a href="COMMANDS.txt">COMMANDS.txt</a>。新源码与记录仅写本阶段；唯一旧路径删除是已授权、单独记账的PDF缓存清理。</p>
<h2>具体算法与理论</h2><pre>{html.escape(proof)}</pre>
<p><a href="evidence.zip">交付包</a> · <a href="manifest.json">文件哈希</a>。完整三维效果与可发表贡献仍需实际实验，研究目标保持进行中。</p></html>'''
    (out / "stage_report.html").write_text(page)
    code = [repo / "scripts/autodl" / name for name in (
        "qualify_privileged_training_objectives.py", "report_privileged_training_objectives.py")]
    (out / "code_manifest.json").write_text(json.dumps([dict(path=str(f.relative_to(repo)), sha256=sha(f), bytes=f.stat().st_size)
                                                       for f in code], indent=2)+"\n")
    files = sorted(f for f in out.rglob("*") if f.is_file()
                   and f.name not in ["manifest.json", "evidence.zip", "delivery_verification.json"])
    (out / "manifest.json").write_text(json.dumps([dict(path=str(f.relative_to(out)), sha256=sha(f), bytes=f.stat().st_size)
                                                  for f in files], indent=2)+"\n")
    with zipfile.ZipFile(out / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files+[out / "manifest.json"]:
            z.write(f, str(f.relative_to(out)))
        for f in code:
            z.write(f, str(f.relative_to(repo)))
    print(json.dumps(dict(protected_sources_unchanged=579, report=str(out / "stage_report.html"),
                          zip_bytes=(out / "evidence.zip").stat().st_size, zip_sha256=sha(out / "evidence.zip"))))


if __name__ == "__main__":
    main()
