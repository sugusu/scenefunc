"""Persist only derived benchmark evidence, reading scope and original analysis."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

STAGE="20261111_critic_conditioning_evidence"


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--source-root",required=True,type=Path)
    a=p.parse_args();repo,root=a.repo.resolve(),a.source_root.resolve();out=repo/"runs/autodl"/STAGE
    result=json.loads((out/"actual_benchmark_v1/result.json").read_text())
    assert result["questions"]==149 and result["new_model_calls"]==0
    led=[
      dict(source="Scene-OTC-Bench",url="https://huggingface.co/datasets/XuanyuLiu/Scene-OTC-Bench/tree/"+result["pin"],
        scope="All149 structured rows, full data card, full checksum metadata; no actual149-question model evaluation",
        images="Only two images selected by first repeated hash in official question order downloaded and byte-hashed; no full image/manual semantic review",
        rights="No license field found in inspected dataset card; raw data/images not redistributed here; held-out set not trained on"),
      dict(source="Scene-Designer fixed source",pin="a0565e50ed7d32c0e09132b9311c0b9b91165405",
        url="https://github.com/ZJU-OmniAI/Scene-Designer/tree/a0565e50ed7d32c0e09132b9311c0b9b91165405",
        read="README; build_dataset.py388-459,679-727, ranking configuration/compatibility excerpts893-1017,1155-1315; perturb.py387-500 excerpts,1140-1217,1368-1425; scene_io.py visible module; offline_snap.py selected state/SDF sections; feedback_check.py selected dependency excerpts",
        not_read="Not all84KB driver or69KB perturb module; truncated output not counted as full module reading; no judge/render/network/model helpers executed",
        redistribution="New source remains RAM-only, no root code license established in inspected pin"),
      dict(source="SceneReVis",url="https://arxiv.org/html/2602.09432v2",
        pin="e2daa314bc9398e3a180bef241d45535812b7c97",
        read="This turn reread method3.3 and neighboring3.1-3.4 passages; newly read utils/scene_editor.py1-393 (excluding later example), not whole repository. utils/sample.py fetched but not substantively read",
        not_done="No reverse-generation dataset sample, no retrieval/render/source function execution, no claim full SDK replay"),
      dict(source="Scene-Designer paper",url="https://arxiv.org/html/2610.23121v1",
        read="Targeted D.2 causal consolidation and B conditional critic formulation, selected inference-critic passages; previous D.3 reading remains separately scoped",
        not_done="Not a fresh full-paper/figure audit or training reproduction"),
      dict(source="SceneCritic",url="https://arxiv.org/abs/2604.13035",
        repository="https://github.com/DIASENGUPTA/SceneCritic",
        read="Primary abstract and official README search result only",
        not_done="No full method/code/ontology read this turn; not evidence its rules cover every user purpose")]
    (out/"reading_ledger.json").write_text(json.dumps(led,ensure_ascii=False,indent=2)+"\n")
    sources=[]
    for name in ["source_manifest.json","revis_manifest.json","otc_source_manifest.json","otc_download_manifest.json"]:
        f=root/name
        if f.exists(): sources.append(dict(name=name,manifest=json.loads(f.read_text())))
    sources.append(dict(path="otc/README.md",bytes=(root/"otc/README.md").stat().st_size,
        sha256=sha(root/"otc/README.md"),note="Pinned readme first request TLS EOF, later successful retry at same pin; initial failure retained in fetch manifest"))
    (out/"primary_sources_manifest.json").write_text(json.dumps(sources,indent=2)+"\n")
    protected_path=repo/"runs/autodl/20261111_feedback_information_control/source_before_after.json"
    protected=json.loads(protected_path.read_text())["original_inputs"];checked=[]
    for row in protected:
        f=Path(row["path"]);after=sha(f) if f.is_file() else None
        checked.append(dict(path=str(f),before=row["before"],after=after,same=after==row["before"]))
    assert len(checked)==579 and all(v["same"] for v in checked)
    (out/"source_before_after.json").write_text(json.dumps(dict(count=579,unchanged=True,original_inputs=checked),indent=2)+"\n")
    title="把条件设计假设接回真实三维评价：149题与构造源码核查"
    proof=(out/"algorithm_and_theory.txt").read_text()
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{max-width:1100px;margin:30px auto;padding:0 20px;font:17px/1.75 system-ui;color:#162631}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #bdc9d0;padding:10px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.8 system-ui;background:#f3f6f8;padding:18px}}a{{color:#075d99}}</style>
<h1>{title}</h1><p>2026-11-11 · 完整室内三维生成研究 · 无新模型推理、训练或场景执行。</p>
<p>这轮取得并实际解析官方测试集的全部149条输入，结合新增构造源码阅读，明确了可以直接用作强对照的评价材料，以及它无法单独回答的条件设计问题。没有把排序处理或源码核查当作三维生成成功。</p>
<table><tr><th>实测/核查</th><th>结果</th><th>边界</th></tr>
<tr><td>官方结构化问题</td><td>94个二选一、55个四场景排序</td><td>0条新模型预测；不是149个完整生成任务</td></tr>
<tr><td>发布图像引用</td><td>408路径、260个清单内容哈希、11个重叠连通块</td><td>不等于11个房间；未下载核验全图</td></tr>
<tr><td>偏好一致性</td><td>424条展开偏好、414个不同内容对、重复对方向冲突0</td><td>检查标签一致性，未验证每个语义偏好是否正确</td></tr>
<tr><td>输入条件</td><td>149条都没有逐题用途、几何或工具历史字段</td><td>仅描述发布表；不否定条件Critic架构或作者其他训练信息</td></tr>
<tr><td>独立重复图像探针</td><td>两份各986466字节图像SHA一致</td><td>按第一重复哈希选择；不推广成已核验所有清单内容</td></tr>
<tr><td>观察/答案隔离与字母映射</td><td>1508次元数据置换检查通过</td><td>不是推理分数或模型鲁棒性结果</td></tr>
<tr><td>旧输入</td><td>579文件哈希未变</td><td>未改旧结果、阈值、模型或原图</td></tr></table>
<p>结论：一般设计质量已有可用、标签自洽的公开对照；用途变化下的取舍仍需单独实测，不能从该基准缺少相关字段推出模型不会。候选机制应与明确用途提示、普通条件偏好和同信息规则直接比较，不以新增模块名称代替贡献。</p>
<p>来源：<a href="https://huggingface.co/datasets/XuanyuLiu/Scene-OTC-Bench/tree/{result['pin']}">官方基准固定版</a>、<a href="https://arxiv.org/html/2610.23121v1">Scene-Designer</a>、<a href="https://arxiv.org/html/2602.09432v2">SceneReVis</a>。<a href="reading_ledger.json">实际阅读范围</a> · <a href="primary_sources_manifest.json">固定来源与哈希</a>。</p>
<h2>具体算法与理论边界</h2><pre>{html.escape(proof)}</pre>
<p><a href="actual_benchmark_v1/result.json">实际结果</a> · <a href="actual_benchmark_v1/question_review.csv">149条完整审阅表</a> · <a href="actual_benchmark_v1/content_blocks.json">内容重叠分组</a> · <a href="COMMANDS.txt">可复算命令</a> · <a href="source_before_after.json">旧输入核验</a> · <a href="evidence.zip">交付包</a>。</p>
<p>原始公开数据和两份图像仅在临时内存目录，未复制进Git，未用作训练。主研究目标仍未达成：本轮不声称新颖性、完整生成收益或CVPR成果。</p></html>'''
    (out/"stage_report.html").write_text(page)
    code=[repo/"scripts/autodl"/x for x in ["audit_scene_critic_benchmark.py","report_critic_conditioning_evidence.py"]]
    (out/"code_manifest.json").write_text(json.dumps([dict(path=str(f.relative_to(repo)),bytes=f.stat().st_size,sha256=sha(f)) for f in code],indent=2)+"\n")
    files=sorted(f for f in out.rglob("*") if f.is_file() and f.name not in ["manifest.json","evidence.zip","delivery_verification.json"])
    (out/"manifest.json").write_text(json.dumps([dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha(f)) for f in files],indent=2)+"\n")
    with zipfile.ZipFile(out/"evidence.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for f in files+[out/"manifest.json"]:z.write(f,str(f.relative_to(out)))
        for f in code:z.write(f,str(f.relative_to(repo)))
    print(json.dumps(dict(report=str(out/"stage_report.html"),zip_bytes=(out/"evidence.zip").stat().st_size,zip_sha256=sha(out/"evidence.zip"),unchanged_old_inputs=579)))


if __name__=="__main__":main()
