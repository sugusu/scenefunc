"""Package completed source reading and evidence qualification; no model path."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import zipfile


def sha(path):
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            state.update(block)
    return state.hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    repo, sources = args.repo.resolve(), args.source_root.resolve()
    out = repo / "runs/autodl/20261111_conditional_design_evidence"
    result = json.loads((out / "actual_v2/result.json").read_text())
    cards = json.loads((out / "actual_v2/qualified_source_cards.json").read_text())
    if result["cases_with_partial_fetched_HTML"] != ["tm8"] or len(cards) != 3:
        raise ValueError("Update the report if the source representation changes")
    primary = json.loads((sources / "primary_sources_manifest.json").read_text())
    for row in primary:
        if sha(sources / (row["id"] + ".html")) != row["sha256"]:
            raise ValueError("Primary source changed")
    write(out / "primary_sources_manifest.json", primary)
    paper = json.loads((sources / "convertible_manifest.json").read_text())
    if sha(sources / "convertible2020.pdf") != paper["sha256"] or paper["pages"] != 12:
        raise ValueError("Read paper differs")
    write(out / "reading_ledger.json", [dict(
        title="Motion Planning for Convertible Indoor Scene Layout Design",
        source=paper, read_PDF_text_pages=list(range(1, 13)),
        scope="Full 12-page author accepted-manuscript extracted text, including methods, Algorithm1, study tables and limitations; not page-image verification, cited works or implementation execution",
        final_metadata_source="https://researchportal.hkust.edu.hk/en/publications/motion-planning-for-convertible-indoor-scene-layout-design/",
        final_venue="IEEE TVCG 27(12), 4413–4424, December 2021; DOI 10.1109/TVCG.2020.3005680",
        overlap="Convertible layouts and transition scheduling are existing capabilities, not a new main contribution",
        no_code_or_data_reproduction=True, source_PDF_redistributed=False), dict(
        title="SceneExplorer", url="https://researchportal.bath.ac.uk/en/publications/sceneexplorer-an-interactive-system-for-expanding-scheduling-and-/",
        read="Author-institution abstract and publication record only; full methods remain unread in this turn",
        venue="IEEE TVCG 31(9), 6008–6021, 2025; early online 2024", full_method_claim=False), dict(
        title="Function2Scene", url="https://arxiv.org/html/2605.30819v1",
        newly_read="Section3 and Table1, Section4.1; prior Section4.2 remains explicitly reused",
        prior_implementation_record=str(repo.parent.parent / "function2scene_source_release_20261103/result.json"),
        scope="HTML text; no new official backend execution or full supplementary implementation review"), dict(
        title="Three primary project descriptions", sources=[dict(id=r["id"], url=r["url"]) for r in primary],
        read="Main project description text through web reader; fetched HTML location checks separate",
        source_kind="Two designer-owned pages and architect-supplied TM8 project text",
        not_read_or_run="Plans not measured, images not independently evaluated, no new scene reconstruction or preference annotation"), dict(
        title="AIRB’N’P", url="https://position-concept.com/portfolio/airbnp-2/",
        read="Main full-story text rechecked; not a new source case and no new action label",
        scope="Open rail for quick access and retained other storage do not establish a converse rule"),
    ])
    previous = json.loads((repo / "runs/autodl/20261111_phase_context_equivalence/book_reading_ledger.json").read_text())
    new_pages = list(range(9, 29))
    covered = sorted(set(previous["covered_text_PDF_pages"]) | set(new_pages))
    if len(covered) != 232:
        raise ValueError("Reading coverage differs")
    write(out / "book_reading_ledger.json", dict(
        source=previous["source"], source_sha256=sha(Path(previous["source"])), PDF_pages=312,
        new_complete_text_PDF_pages=new_pages,
        text_page_sha256={str(n): sha(sources / "book" / f"page_{n}.txt") for n in new_pages},
        covered_text_PDF_pages=covered, covered_count=len(covered),
        remaining_text_pages=sorted(set(range(1, 313)) - set(covered)),
        scope="Introduction and first part of chapter1; actual extracted text read, not every figure or cited product claim",
        document_examples_executed=False))
    write(out / "execution_attempts.json", [dict(
        attempt="v1", status="failed_before_result_write", reason="TM8 local HTML did not contain all later project-text anchors",
        scope="Full prose had been read through web reader; local representation completeness was not established"), dict(
        attempt="v2", status="completed_with_explicit_source_coverage_limit", result="actual_v2/result.json",
        note="Retained all three sources. Missing anchors remain false; no geometric thresholds, task sample or quality labels changed")])
    write(out / "environment_dependency_note.json", dict(
        existing_native_python="/root/autodl-tmp/scene_generation_handoff_20261014_v2/work/envs/hsm_cpu/bin/python",
        resolved_base="/root/autodl-tmp/layout_experience_pilot/envs/eval/bin/python3.11",
        reason_kept="Current native environment resolves to this older interpreter and enables system site packages; age is not evidence that removal is safe",
        deleted_old_files=0, deleted_weights=0))
    write(out / "algorithm_contract.json", dict(
        complete_scene_generation_goal_unchanged=True,
        candidate="Conditional source-principle application with own generated decision and real native actions",
        new_algorithm_established=False,
        newly_verified_policy_advantage=False,
        important_revision="Unselected and counterfactual choices remain unknown unless separately evidenced; fixed and movable strategies are not automatically opposing labels",
        controls="Same U/X/R/tools; direct actions, ordinary CoT, AUX, phase prompting, same-prefix split identity, capability-aware retrieval/feedback/search, prior functional checking and preference learning",
        new_training_labels=0, training_admitted=False,
        next_reading="Remaining book PDF29–108 and SceneExplorer full methods",
        next_data_work="Seek independent applicability/effect evidence rather than manufacture inverse labels from successful project choices",
        paid_API_calls=0, new_model_forwards=0, new_complete_scene_trials=0))
    card_rows = "".join("<tr>" + "".join("<td>" + html.escape(str(card[k])) + "</td>" for k in
        ("title", "reported_context", "reported_strategy", "reported_effect", "source_representation_status")) + "</tr>" for card in cards)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>条件设计来源审阅：避免制造拒用标签</title><style>body{max-width:1100px;margin:32px auto;padding:0 18px;background:#fafbfc;color:#213044;font:17px/1.7 system-ui}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:10px;border:1px solid #c8d2df;text-align:left}th{background:#e8eef6}pre{white-space:pre-wrap;font:15px/1.8 system-ui}.note{padding:15px;background:#eef4fa;border-left:4px solid #527797}a{color:#275b94}</style>
<h1>设计案例支持采用理由，尚未给出拒用边界</h1>
<p class="note">本轮新增3个项目的来源范围审阅、关联8个既有局部动作、完整阅读一项布局转换前作的12页提取文字。没有模型运行、训练、偏好标注或新生成成功房间。</p>
<h2>来源实际说了什么</h2><table><tr><th>项目</th><th>条件</th><th>采用方案</th><th>作者报告的作用</th><th>来源获取范围</th></tr>''' + card_rows + '''</table>
<p>这些都是作者报告，未经本轮独立测量。它们说明不能把灵活用途直接等同于必须移动全部家具，也不能把未选方案当作已证伪。三个项目都没有被转换成新动作或偏好标签。</p>
<p>一手来源：<a href="https://www.studiowok.com/project/batipin-flat-4/">studio wok</a> · <a href="https://www.abrogers.com/portfolio/compact-living/">Ab Rogers Design</a> · <a href="https://www.archdaily.com/1001060/tm8-apartment-dd">TM8建筑师项目说明</a>。TM8完整项目文字通过web reader读到；下载的HTML只含部分正文，缺失定位词仍记录为False。</p>
<h2>对算法的约束</h2><p>保留完整场景生成的目标和条件原则应用候选，但目前不宣称新架构或新loss。相同知识下的普通CoT、明确需求提示、能力检索、反馈和搜索必须作为强对照；拒用/偏好/完整成功仍需相应证据。既有8条局部重建监督不升级为原设计者完整轨迹。</p>
<h2>直接近邻与阅读边界</h2><p><a href="https://graphics.cs.uh.edu/wp-content/papers/2020/2020-TVCG-MotionPlanningforConvertible.pdf">布局转换前作</a>作者接收稿12页提取文字已读；正式出版为TVCG2021。没有执行其系统或逐图检查。<a href="https://researchportal.bath.ac.uk/en/publications/sceneexplorer-an-interactive-system-for-expanding-scheduling-and-/">SceneExplorer</a>本轮仅核对作者机构摘要/出版信息，完整方法待读。书新增PDF9–28，累计232/312页，余80页。</p>
<h2>交付</h2><p><a href="actual_v2/result.json">实际运行记录</a> · <a href="actual_v2/source_review.csv">完整三项目审阅表</a> · <a href="actual_v2/existing_action_source_links.json">八个既有动作的来源边界</a> · <a href="reading_ledger.json">论文/来源阅读记录</a> · <a href="book_reading_ledger.json">书本页码</a> · <a href="source_before_after.json">旧源哈希</a> · <a href="evidence.zip">证据包</a></p>
<details><summary>完整算法与理论修订</summary><pre>''' + html.escape((out / "algorithm_and_theory.txt").read_text()) + '''</pre></details></html>'''
    (out / "stage_report.html").write_text(page)
    old = json.loads((repo / "runs/autodl/20261111_feedback_information_control/source_before_after.json").read_text())
    checked = [dict(path=r["path"], before=r["before"], after=sha(Path(r["path"]))) for r in old["original_inputs"]]
    if not all(r["before"] == r["after"] for r in checked):
        raise ValueError("Protected old source changed")
    write(out / "source_before_after.json", dict(count=len(checked), all_originals_unchanged=True, original_inputs=checked))
    code = [repo / "scripts/autodl" / n for n in ("qualify_conditional_design_evidence.py", "report_conditional_design_evidence.py")]
    for path in code:
        ast.parse(path.read_text())
    write(out / "code_manifest.json", [dict(path=str(p.relative_to(repo)), bytes=p.stat().st_size, sha256=sha(p)) for p in code])
    files = sorted(p for p in out.rglob("*") if p.is_file())
    write(out / "manifest.json", [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p)) for p in files])
    with zipfile.ZipFile(out / "evidence.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in files + [out / "manifest.json"]:
            archive.write(p, p.relative_to(out))
        for p in code:
            archive.write(p, "code/" + p.name)
    write(out / "delivery_verification.json", dict(
        protected_sources=len(checked), all_unchanged=True, evidence_bytes=(out / "evidence.zip").stat().st_size,
        evidence_sha256=sha(out / "evidence.zip"), original_PDF_HTML_or_photos_redistributed=False,
        new_training_labels=0, new_model_forwards=0, research_goal_achieved=False))
    print(json.dumps(dict(out=str(out), protected_sources=len(checked), all_unchanged=True,
                         book_text_pages_covered=232, evidence_bytes=(out / "evidence.zip").stat().st_size)))


if __name__ == "__main__":
    main()
