"""Qualify reported design choices without inventing counterfactual labels.

Reads actual source HTML acquired separately into RAM and the existing frozen
reconstruction corpus. Does not execute third-party code, a model or a simulator.
The output is an evidence ledger, not additional expert action training data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    repo, out = args.repo.resolve(), args.output.resolve()
    stage = repo / "runs/autodl/20261111_conditional_design_evidence"
    if not (out.is_relative_to(stage) or out.is_relative_to(Path("/dev/shm"))):
        raise ValueError("Write this study or RAM only")
    if out.exists():
        raise FileExistsError(out)
    cards_path = stage / "source_cards.json"
    old_path = repo / "runs/autodl/20261111_decision_source_value/principle_source_review.json"
    corpus_path = repo / "runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl"
    paths = [cards_path, old_path, corpus_path]
    before = {str(p): sha(p) for p in paths}
    cards = json.loads(cards_path.read_text())["cards"]
    manifest = json.loads((args.source_root / "primary_sources_manifest.json").read_text())
    source_by_id = {row["id"]: row for row in manifest}
    qualified = []
    for card in cards:
        source = source_by_id[card["id"]]
        if source.get("url") != card["url"] or "sha256" not in source:
            raise ValueError("A web reading alone is not a hash-verified local HTML")
        path = args.source_root / (card["id"] + ".html")
        if sha(path) != source["sha256"]:
            raise ValueError("Primary source changed")
        text = TextOnly()
        text.feed(path.read_text(errors="strict"))
        normalized = " ".join(" ".join(text.parts).split()).casefold()
        anchors = {s: s.casefold() in normalized for s in card["anchors"]}
        for name in ("counterfactual_inapplicability_label", "whole_scene_success_label", "preference_label"):
            if card[name] is not None:
                raise ValueError("This source review must not manufacture new supervision")
        qualified.append(dict(card, source_html_sha256=source["sha256"], anchors_present=anchors,
                              all_locators_present_in_fetched_HTML=all(anchors.values()),
                              source_representation_status="local_HTML_locators_present" if all(anchors.values())
                              else "Full project prose was read via web reader; fetched HTML does not contain every later paragraph",
                              note="Anchor presence checks source location, not truth or independent verification of the authored paraphrase"))

    old = json.loads(old_path.read_text())["cases"]
    source_lookup = {c["decision_id"]: c for c in old}
    rows = [json.loads(s) for s in corpus_path.read_text().splitlines()]
    existing = []
    for row in rows:
        if row["arm"] != "paired_phase_v3" or row["stage"] != "decision":
            continue
        body = row["target_text"].split("<think>", 1)[1].split("</think>", 1)[0].strip()
        decision = json.loads(body)
        application = decision.get("principle_applicability", {})
        source_id = application.get("source_decision_id")
        if not source_id:
            continue
        source = source_lookup[source_id]
        if row["unit_id"] not in source["units"]:
            raise ValueError("Source-to-reconstruction link differs")
        existing.append(dict(unit_id=row["unit_id"], family_id=row["family_id"], principle_id=source_id,
                             source_url=source["URL"], source_support=source["supported"],
                             not_source_demonstrated=source["not_supported"],
                             annotation_kind="Existing researcher reconstruction D label, unchanged",
                             original_designer_action_demonstrated=False,
                             new_inapplicability_label=None, new_full_scene_label=None))
    if len(existing) != 8 or len(qualified) != 3:
        raise ValueError("Frozen review scope changed; do not silently change its denominator")
    after = {str(p): sha(p) for p in paths}
    if after != before:
        raise RuntimeError("Input modified during review")
    out.mkdir(parents=True)
    (out / "qualified_source_cards.json").write_text(json.dumps(qualified, ensure_ascii=False, indent=2) + "\n")
    (out / "existing_action_source_links.json").write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
    with (out / "source_review.csv").open("w", newline="") as stream:
        columns = ["id", "title", "url", "reported_context", "reported_strategy", "reported_effect",
                   "evidence_status", "counterfactual_inapplicability_label", "whole_scene_success_label", "preference_label"]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for c in qualified:
            writer.writerow({k: c[k] for k in columns})
    result = dict(kind="Source-scope qualification, not a trained policy experiment",
                  new_project_source_reports=3, existing_unchanged_principle_action_links=8,
                  existing_source_projects=5, independently_verified_new_functional_successes=0,
                  full_local_HTML_locator_matches=sum(c["all_locators_present_in_fetched_HTML"] for c in qualified),
                  cases_with_partial_fetched_HTML=[c["id"] for c in qualified if not c["all_locators_present_in_fetched_HTML"]],
                  new_preference_labels=0, new_action_training_labels=0, new_inapplicability_labels=0,
                  sources_before=before, sources_after=after, unchanged=True,
                  source_html_bytes=sum(s["bytes"] for s in manifest),
                  full_generation_claim=False, source_report_is_not_a_causal_effect_estimate=True,
                  full_reviewed_current_HTML_is_not_a_fixed_future_web_revision=True,
                  model_forwards=0, training_updates=0, new_scene_trials=0, paid_API_calls=0,
                  seconds=time.monotonic() - tick)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources_before", "sources_after")}))


if __name__ == "__main__":
    main()
