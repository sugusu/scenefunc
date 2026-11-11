"""Assemble the verified-repair source study without neural or scene execution."""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

STAGE = "20261111_verified_repair_supervision"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--repo", type=Path, required=True)
    a.add_argument("--source-root", type=Path, required=True)
    a = a.parse_args()
    repo, root = a.repo.resolve(), a.source_root.resolve()
    out = repo / "runs/autodl" / STAGE
    data = json.loads((out / "actual_public_v1/result.json").read_text())
    math = json.loads((out / "actual_math_v1/result.json").read_text())
    assert data["original_kernel"]["all_coefficients_identical"]
    assert data["full_3D_generations"] == math["scene_executions"] == 0
    # Preserve a small MIT source snapshot, not the 117MB public data cache.
    selected = ["LICENSE", "README.md", "RESULTS.md", "psd/training/river_psd.py",
        "psd/targets/river_topk.py", "experiments/bfcl/agentic/build_river_native_rows.py",
        "experiments/bfcl/agentic/leak_audit.py", "experiments/bfcl/agentic/attempt_runner.py",
        "experiments/bfcl/repair/harvest.py", "experiments/bfcl/repair/slate.py"]
    for name in selected:
        p = out / "upstream_snapshot" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "repo" / name, p)
    provenance = json.loads((root / "download_manifest.json").read_text())
    provenance["original_data_in_RAM_only"] = True
    provenance["snapshot_license"] = "MIT code; CC-BY-4.0 public dataset, no raw conversations republished in this study"
    pages = root.parent / "primary_page_manifest.json"
    provenance["pages"] = json.loads(pages.read_text()) if pages.is_file() else []
    (out / "primary_sources_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")

    ledger = [
        dict(source="PSD author technical article", url="https://www.canvas.inc/research/privileged-self-distillation",
             read="Visible introduction, method, pseudocode, local loss, hint construction, results, limitations; not hidden site scripts/interactive artwork",
             status="Technical article/blog; peer-reviewed acceptance not established"),
        dict(source="PSD official code", pin=data["code_pin"],
             url="https://github.com/essamsleiman/psd/tree/"+data["code_pin"],
             read="README and RESULTS full; historical docs/implementation full as obsolete notes; river_psd lines51-148,285-320; river_topk117-269; harvest146-325; slate1-60,209-295,335-395; build_river_native_rows49-145; leak_audit1-115; attempt_runner full; turn_kl442-528,1177-1265; hints76-107",
             executed="Only three unmodified pure AST functions from river_psd on released cached targets; no clients/SDK/benchmark/model"),
        dict(source="PSD public artifacts", pin=data["data_pin"],
             url="https://huggingface.co/datasets/essamsleiman/psd-bfcl/tree/"+data["data_pin"],
             read="Data card, trace manifest, rollout summary; structural census all375 repairs, all264 traces/423 attempts, all1550 cached targets; selected first traces and first joint slate read qualitatively, not a human semantic audit of all conversations",
             limitation="Released River35B examples, not the published9B held-out experiment; verifier records not independently reexecuted; success-selected traces cannot give repair yield"),
        dict(source="PSP v2", url="https://arxiv.org/html/2609.29051v2",
             read="Sections4.1-4.3 (through transfer discussion), AppA and B complete extracted text; C.1 and C.2 through training-step listing plus optimizer paragraph; selectedD.2-D.5 results and controls; project page visible text",
             not_read="Remaining full paper, full tables/figures, all prompts and references, training implementation/reproduction",
             limitation="Project HuggingFace link points to paper index, not acquired code/weights. The theorem is a Taylor identity under stated update; empirical gradient alignment is not universal")]
    (out / "reading_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+"\n")
    protected_path = repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json"
    protected = json.loads(protected_path.read_text())["original_inputs"]
    checked = []
    for r in protected:
        p = Path(r["path"])
        after = sha(p) if p.is_file() else None
        checked.append(dict(path=str(p), before=r["before"], after=after, same=after == r["before"]))
    assert len(checked) == 579 and all(r["same"] for r in checked)
    (out / "source_before_after.json").write_text(json.dumps(dict(count=579, unchanged=True,
        protected_manifest_sha256=sha(protected_path), original_inputs=checked), indent=2)+"\n")

    d, k = data["public_repairs"], data["original_kernel"]
    rows = [
        ["普通重试/搜索+verified-retry SFT", "相同前缀与工具，不给提示", "通过续轨的实际token", "成功是保留条件", "必须计入全部失败重试和搜索成本；是强对照"],
        ["PSD", "教师有修复提示，学生无提示", "所选首个assistant响应的条件教师Top20分布；保持行为目标另计", "完整任务验证是接纳条件，不是逐token因果标签", "原方法已有，不作为三维新增算法；多提示通过不证明每条提示单独有效"],
        ["PSP v2", "被gate的组保留提示上下文重新采样并训练；测试无提示", "同上下文的普通GRPO", "组内任务回报", "实际依赖参考轨迹；无蒸馏项不等于梯度不依赖提示；迁移须测"],
        ["拟议三维适配", "同一原生助手响应边界、同一实际工具状态；证据访问受控", "先比较真实修复样本SFT和原PSD，未定义新损失", "完整场景要求+几何、未知项分开", "共享资产副作用和整场保留必须验证；尚无完整正例或额外收益"]]
    with (out / "method_difference.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "conditioning", "training_target", "verification", "research_boundary"]); w.writerows(rows)

    theory = (out / "algorithm_and_theory.txt").read_text()
    title = "失败修复怎样成为可用训练监督：原始样本、损失和迁移条件"
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{max-width:1100px;margin:30px auto;padding:0 20px;font:17px/1.75 system-ui;color:#162631}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #bdc9d0;padding:10px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui;background:#f3f6f8;padding:18px}}a{{color:#075d99}}.note{{border-left:4px solid #b77d1d;padding-left:14px}}</style>
<h1>{title}</h1><p>2026-11-11 · 目标仍为完整、可编辑室内三维生成 · 未启动新的模型推理、训练或场景执行。</p>
<p>这次获得了可检查的真实公开修复数据，实际执行作者原始目标打包函数，并补齐了跨上下文训练推理中缺少的数学条件。工作直接关系到如何取得可信训练样本，但不将数据结构、工程检查或标准损失冒充论文创新。</p>
<table><tr><th>实际完成</th><th>结果</th><th>解释范围</th></tr>
<tr><td>公开修复、轨迹、搜索逐条核对</td><td>{d['rows']}条修复、{d['unique_tasks']}个任务、{d['recorded_attempts']}次记录重试</td><td>来源为BFCL工具任务，非三维场景；核对保存验证结果，未重跑验证器</td></tr>
<tr><td>联合提示证据范围</td><td>{d['multi_hint_tasks']}任务、{d['rows_supported_by_joint_slate']}条修复来自多提示方案</td><td>整套方案通过，不等于单条提示的独立因果效果</td></tr>
<tr><td>原函数实际执行</td><td>{k['targets']}个真实缓存目标，{k['calls']}次纯函数调用</td><td>两种打包格式所有输入/目标/权重系数一致；未证明服务器归一化、随机性和训练步完全等价</td></tr>
<tr><td>两类目标的实际训练权重</td><td>repair/preserve目标数375/1175；各自行权重和约375</td><td>累计token权重为22389/25531.42；相等系数或行权重和不代表相同token权重总量</td></tr>
<tr><td>3个明确数学反例</td><td>同概率可反向梯度；轮内对齐可整轨反向；预条件可改变对齐</td><td>有限维人造数学见证，非模型训练结果；不否定论文经验结论</td></tr>
<tr><td>旧输入保留</td><td>579个受保护输入哈希一致</td><td>未改旧场景、评分阈值或模型权重</td></tr></table>
<p class="note">关键判断：修复后通过验证，可以产生训练候选；它既不自动证明修复提示是成功原因，也不保证训练去掉提示后仍会改善。PSD和PSP已经覆盖两种主要训练路线。当前应选择强基线和真实可重放的数据，而不是再给同一思想命名。</p>
<p>来源：<a href="https://www.canvas.inc/research/privileged-self-distillation">PSD作者方法</a>、<a href="https://github.com/essamsleiman/psd/tree/{data['code_pin']}">固定官方代码</a>、<a href="https://huggingface.co/datasets/essamsleiman/psd-bfcl/tree/{data['data_pin']}">固定公开数据</a>、<a href="https://arxiv.org/html/2609.29051v2">PSP v2方法与证明</a>。逐项阅读边界见<a href="reading_ledger.json">阅读记录</a>。</p>
<h2>算法、理论与强对照</h2><pre>{html.escape(theory)}</pre>
<h2>实际可复算证据</h2><p><a href="actual_public_v1/result.json">公开数据和原核结果</a> · <a href="actual_public_v1/public_trace_census.csv">264任务表</a> · <a href="actual_public_v1/public_attempt_census.csv">423次尝试</a> · <a href="actual_public_v1/public_repair_census.csv">375修复表</a> · <a href="actual_public_v1/public_target_census.csv">1550目标表</a> · <a href="actual_math_v1/result.json">数学见证</a> · <a href="method_difference.csv">方法差异</a> · <a href="COMMANDS.txt">命令</a> · <a href="primary_sources_manifest.json">来源与哈希</a> · <a href="execution_events.json">失败与恢复记录</a> · <a href="source_before_after.json">旧输入核验</a> · <a href="evidence.zip">交付包</a></p>
<p>环境中途的/dev/shm内容被清空，第一次数据核查在读入来源前报FileNotFoundError；没有产生样本统计。随后通过固定代码提交和固定数据提交重新下载，三份数据字节再次匹配发布清单；恢复不包含新权重或付费调用。数学校验结果始终保存在数据盘。</p>
<p>研究仍未完成：尚无这一路径产生的完整三维正例、模型更新、未见房间收益或CVPR级创新证据。下一步应继续解决具体的训练数据和方法差别，不能把本报告当作主目标完成。</p></html>'''
    (out / "stage_report.html").write_text(page)
    code = [repo / "scripts/autodl" / n for n in ["prepare_verified_repair_sources.py", "qualify_verified_repair_supervision.py", "qualify_cross_view_transfer.py", "report_verified_repair_supervision.py"]]
    (out / "code_manifest.json").write_text(json.dumps([dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size, sha256=sha(p)) for p in code], indent=2)+"\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ["manifest.json", "evidence.zip", "delivery_verification.json"])
    (out / "manifest.json").write_text(json.dumps([dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)) for p in files], indent=2)+"\n")
    with zipfile.ZipFile(out / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[out / "manifest.json"]: z.write(p, str(p.relative_to(out)))
        for p in code: z.write(p, str(p.relative_to(repo)))
    print(json.dumps(dict(report=str(out / "stage_report.html"), zip_bytes=(out / "evidence.zip").stat().st_size,
        zip_sha256=sha(out / "evidence.zip"), protected_inputs_unchanged=len(checked))))


if __name__ == "__main__":
    main()
