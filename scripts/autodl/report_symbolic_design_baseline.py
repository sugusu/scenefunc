"""Package original analysis and derived public-ontology results, not source/data."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

STAGE="20261111_symbolic_design_baseline"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",required=True,type=Path);ap.add_argument("--source-root",required=True,type=Path)
    a=ap.parse_args();repo,root=a.repo.resolve(),a.source_root.resolve();out=repo/"runs/autodl"/STAGE
    result=json.loads((out/"actual_inventory_v1/result.json").read_text())
    assert result["ontology_categories"]==79 and result["public_task_inventories"]==21
    source_meta={}
    for n in ["download_manifest.json","code_manifest.json","task_manifest.json","ontology_download_manifest.json","public_data_probe.json"]:
        source_meta[n]=json.loads((root/n).read_text())
    source_meta["upstream_license_status"]="No root license found in pinned GitHub tree or repository metadata; original new source and ontology remain RAM-only, not redistributed"
    (out/"primary_sources_manifest.json").write_text(json.dumps(source_meta,indent=2)+"\n")
    ledger=[dict(source="SceneCritic v1",url="https://arxiv.org/html/2604.13035v1",
                 read="Sections3-4 full method text; AppA and AppB including equations and tabulated thresholds; selected introduction/related work, setup and reliability study, test-bed7.1; no full fresh results/figure audit",
                 not_done="No human study replication, all13 model tests, raw source-dataset mining or full geometry evaluation"),
        dict(source="SceneCritic official source",pin=result["code_pin"],url="https://github.com/DIASENGUPTA/SceneCritic/tree/"+result["code_pin"],
             read="README full; SceneCritic_evaluator.py1-180 types and score,168-343 aliases/normalization,341-519 conversion/parsing,519-687 ontology/config/types,687-778 scale, selected970-1208 orientation, selected1282-1518 cooccurrence,1518-1785 completeness/orchestration; module import declarations and main guard",
             additional_source="ontology/generate_ontology.py1990-2109 config generator; function/class listings inspected. Other fetched files are acquisition only, not all read",
             executed="Imported unchanged pinned module (standard library+NumPy); original load_ontology/build_config_from_task/normalization/CompletenessVerifier only. No main, renderer, model or refinement driver"),
        dict(source="Public SceneOnto",url="https://drive.google.com/file/d/1Rj7WVNzUnkVp-GFBg4KILXGlTlJJxpkQ/view",
             bytes=1912316,sha256=result["ontology_sha256"],read="Full JSON parsed by original loader;79 categories counted; relevant field coverage for217 declared instances; not a manual audit of all underlying observations",
             version="Google Drive file ID is not immutable; downloaded bytes pinned by localSHA. Raw source datasets/meshes not fetched"),
        dict(source="21 official task inventories",read="All21 assets dictionaries parsed and checked against the same declared counts projected to the author's category interface",
             boundary="Input-annotation self-consistency diagnostic, not21 generated rooms or217 physically/semantically verified objects; ordinary pooled count is not a new method")]
    (out/"reading_ledger.json").write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+"\n")
    events=[dict(event="Initial source import admission",outcome="Rejected module because NumPy at line355 was absent from initial allowlist; no result written"),
            dict(event="Source dependency review",outcome="Read341-478; permitted inspected NumPy import, original author rule bodies and thresholds unchanged"),
            dict(event="Successful execution",outcome="79-category ontology load and21 declared-inventory checks",seconds=result["seconds"])]
    (out/"execution_events.json").write_text(json.dumps(events,indent=2)+"\n")
    pp=repo/"runs/autodl/20261111_feedback_information_control/source_before_after.json"
    before=json.loads(pp.read_text())["original_inputs"];checks=[]
    for v in before:
        p=Path(v["path"]);after=sha(p) if p.is_file() else None;checks.append(dict(path=str(p),before=v["before"],after=after,same=after==v["before"]))
    assert len(checks)==579 and all(v["same"] for v in checks)
    (out/"source_before_after.json").write_text(json.dumps(dict(count=579,unchanged=True,original_inputs=checks),indent=2)+"\n")
    title="真正运行普通知识规则对照：SceneOnto数据和原始计数路径"
    proof=(out/"algorithm_and_theory.txt").read_text()
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{max-width:1100px;margin:30px auto;padding:0 20px;font:17px/1.75 system-ui;color:#162631}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #bdc9d0;padding:10px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui;background:#f3f6f8;padding:18px}}a{{color:#075d99}}</style>
<h1>{title}</h1><p>2026-11-11 · 完整室内三维生成研究 · 本轮无新模型、训练或场景执行。</p>
<p>这轮取得并加载作者公开的1.9MB空间本体，使普通知识对照有实际数据支持。我们运行了原始计数路径，定位一处规范类别合并的重复计数，并实现普通容量计数作对照。没有把工程修正算作创新，也没有把声明清单当作生成结果。</p>
<table><tr><th>实际结果</th><th>数值</th><th>限制</th></tr>
<tr><td>官方本体</td><td>79类、1,912,316字节</td><td>原始观测未独立重新挖掘；数据文件以本轮SHA固定</td></tr>
<tr><td>公开任务清单</td><td>21份、217个声明实例</td><td>没有坐标，不是21场生成或质量测试</td></tr>
<tr><td>本体/完整尺寸字段覆盖</td><td>212/217</td><td>有统计不代表物体尺寸合格</td></tr>
<tr><td>至少一种朝向统计达到样本门槛</td><td>198/217</td><td>不是关系适用或朝向通过</td></tr>
<tr><td>原始自清单计数</td><td>20份为1.0；classroom_2为15/20=0.75</td><td>别名合并后分别比较原始期望项导致额外罚项；原源码未改</td></tr>
<tr><td>普通容量计数对照</td><td>21份声明数量一致</td><td>只验证数量；细分类、用途角色和几何未知</td></tr>
<tr><td>CPU实际运行</td><td>{result['seconds']:.3f}秒</td><td>本体/计数函数，非场景生成耗时</td></tr>
<tr><td>旧输入</td><td>579文件哈希未变</td><td>未修改原场景、阈值、模型和原始数据</td></tr></table>
<p>下一步的研究差别必须超过普通的任务解析、来源知识、实例匹配与规则搜索。高频空间先验不是用户硬要求；在输入中删除用途再比较条件Agent，会得到难以解释的弱基线优势。本轮尚未建立候选机制的独立收益。</p>
<p>来源：<a href="https://arxiv.org/html/2604.13035v1">SceneCritic原文</a>、<a href="https://github.com/DIASENGUPTA/SceneCritic/tree/{result['code_pin']}">固定官方源码</a>、<a href="https://drive.google.com/file/d/1Rj7WVNzUnkVp-GFBg4KILXGlTlJJxpkQ/view">作者公开本体</a>。<a href="reading_ledger.json">实际阅读记录</a> · <a href="primary_sources_manifest.json">来源及哈希</a>。</p>
<h2>算法定义、普通基线与理论</h2><pre>{html.escape(proof)}</pre>
<p><a href="actual_inventory_v1/result.json">实际原函数结果</a> · <a href="actual_inventory_v1/inventory_details.json">21清单逐项证据</a> · <a href="actual_inventory_v1/ontology_coverage.csv">本体覆盖表</a> · <a href="COMMANDS.txt">复算命令</a> · <a href="execution_events.json">运行与首次准入失败</a> · <a href="source_before_after.json">旧输入核验</a> · <a href="evidence.zip">交付包</a>。</p>
<p>没有新GPU请求、付费API或模型调用。本体数据与新原始源码在RAM，未重新分发。研究主目标保持进行中，未宣称完整生成成功或CVPR级成果。</p></html>'''
    (out/"stage_report.html").write_text(page)
    code=[repo/"scripts/autodl"/v for v in ["qualify_symbolic_design_baseline.py","report_symbolic_design_baseline.py"]]
    (out/"code_manifest.json").write_text(json.dumps([dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=sha(p)) for p in code],indent=2)+"\n")
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','evidence.zip','delivery_verification.json'])
    (out/"manifest.json").write_text(json.dumps([dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],indent=2)+"\n")
    with zipfile.ZipFile(out/"evidence.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/"manifest.json"]:z.write(p,str(p.relative_to(out)))
        for p in code:z.write(p,str(p.relative_to(repo)))
    print(json.dumps(dict(report=str(out/"stage_report.html"),evidence_zip_bytes=(out/"evidence.zip").stat().st_size,evidence_zip_sha256=sha(out/"evidence.zip"),old_inputs_unchanged=579)))


if __name__=="__main__":main()
