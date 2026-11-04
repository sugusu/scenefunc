"""Prepare a reviewed SFT subset without starting training.

Checks declared provenance, hashes and project-family assignments. It cannot
authenticate a reviewer, establish legal rights or replace scientific review.
Only the explicit manifest is copied, never neighboring JSON fixtures.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from review_samples import checked_ref

ALLOWED_ORIGINS = {"professional_project_decision", "researcher_reconstruction_from_professional_project"}
REVIEW_SCOPES = {"research_use_permission", "source_project_correspondence",
                 "pre_design_input_boundary", "action_replay_or_review_label"}


def _image_paths(episode):
    paths = list(episode.get("images", []))
    for turn in episode.get("turns", []):
        paths.extend(turn.get("images", []))
        single = turn.get("image")
        if single:
            paths.extend(single if isinstance(single, list) else [single])
    return paths


def qualification_findings(manifest, root):
    findings = []
    if manifest.get("purpose") != "professional_sft_preparation":
        findings.append("wrong_manifest_purpose")
    if not manifest.get("samples"):
        findings.append("empty_training_manifest")
    registry = json.loads(checked_ref(root, manifest["project_registry"]).read_text())
    projects = {}
    family_splits = {}
    for project in registry["projects"]:
        pid, family, split = project["project_id"], project["family_id"], project["allocation"]
        if pid in projects:
            findings.append("duplicate_project:" + pid)
        projects[pid] = project
        normalized = "development" if split == "development_source_review" else split
        family_splits.setdefault(family, set()).add(normalized)
    for family, splits in family_splits.items():
        if len(splits) > 1:
            findings.append("family_crosses_splits:" + family)
    for pid in manifest.get("retrieval_project_ids", []):
        if pid not in projects or projects[pid]["allocation"] in ("test", "sealed_test", "quarantine_unallocated"):
            findings.append("unapproved_retrieval_project:" + pid)
    seen_ids, seen_hashes = set(), set()
    for sample in manifest.get("samples", []):
        sid = sample.get("sample_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sid) or sid in seen_ids:
            findings.append("invalid_or_duplicate_sample_id:" + sid)
        seen_ids.add(sid)
        episode_path = checked_ref(root, sample["episode"])
        episode = json.loads(episode_path.read_text())
        if episode.get("stage") not in {"floor_plan", "furniture", "wall", "ceiling", "qa", "critic"}:
            findings.append("unsupported_episode_stage:" + sid)
        episode_hash = sample["episode"]["sha256"]
        if episode_hash in seen_hashes:
            findings.append("duplicate_episode_bytes:" + sid)
        seen_hashes.add(episode_hash)
        provenance = episode.get("provenance", {})
        pid = sample.get("project_id")
        project = projects.get(pid)
        if not project:
            findings.append("unknown_project:" + sid)
            continue
        if project.get("training_eligible") is not True or project["allocation"] != "train":
            findings.append("project_not_qualified_for_training:" + sid)
        if sample.get("family_id") != project["family_id"] or provenance.get("family_id") != project["family_id"]:
            findings.append("family_binding_mismatch:" + sid)
        if provenance.get("project_id") != pid:
            findings.append("project_binding_mismatch:" + sid)
        if provenance.get("kind") not in ALLOWED_ORIGINS or provenance.get("training_eligible") is not True:
            findings.append("episode_not_qualified_professional_data:" + sid)
        try:
            fixed = datetime.fromisoformat(project["split_fixed_at"])
            derived = datetime.fromisoformat(sample["derived_at"])
            if fixed.tzinfo is None or derived.tzinfo is None or derived < fixed:
                raise ValueError("Split not fixed before derivation")
        except (KeyError, TypeError, ValueError):
            findings.append("split_before_derivation_not_established:" + sid)
        dependencies = {str(checked_ref(root, ref)): ref for ref in sample.get("dependencies", [])}
        for image in _image_paths(episode):
            path = str((Path(root) / image).resolve())
            if path not in dependencies:
                findings.append("unlocked_episode_image:" + sid)
        review_ref = sample.get("qualification_review")
        if not review_ref:
            findings.append("missing_qualification_review:" + sid)
            continue
        review = json.loads(checked_ref(root, review_ref).read_text())
        if (review.get("decision") != "approved" or not review.get("reviewer_id") or
                review.get("project_id") != pid or review.get("family_id") != project["family_id"] or
                review.get("episode_sha256") != episode_hash):
            findings.append("review_not_bound_to_approved_sample:" + sid)
        evidence = review.get("evidence", {})
        if not REVIEW_SCOPES <= set(evidence):
            findings.append("incomplete_review_evidence:" + sid)
        for ref in evidence.values():
            checked_ref(root, ref)
    return findings


def _copy_selected_episodes(samples, root, destination):
    """Shared copier; caller must establish qualification or explicit audit use."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=destination.parent))
    try:
        for sample in samples:
            sid = sample["sample_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", sid):
                raise ValueError("Unsafe sample ID")
            path = checked_ref(root, sample["episode"])
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != sample["episode"]["sha256"]:
                raise ValueError("Episode changed while preparing subset")
            target = staging / (sid + ".json")
            with target.open("xb") as stream:
                stream.write(contents)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


def prepare_training_subset(manifest, root, destination):
    findings = qualification_findings(manifest, root)
    if findings:
        raise ValueError("Training subset not prepared: " + "; ".join(findings))
    _copy_selected_episodes(manifest["samples"], root, destination)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    prepare_training_subset(manifest, a.root.resolve(), a.out.resolve())
    print(json.dumps({"prepared_samples": len(manifest["samples"]), "training_started": False,
                      "limitation": "Checks recorded declarations only; training needs separate authorization."}))


if __name__ == "__main__":
    main()
