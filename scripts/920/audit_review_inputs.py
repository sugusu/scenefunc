"""CPU audit of real review loading and fail-closed training-subset preparation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from decision_state import object_table, read_checkpoint, sha
from review_samples import checked_ref, decode_review_answer, export_review
from training_manifest import (_copy_selected_episodes, prepare_training_subset,
                               qualification_findings)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "lock", "critic-lock", "registry", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {"started_at": now(), "status": "running", "model_calls": 0,
        "training_updates": 0, "scene_edits": 0, "qualified_professional_samples": 0,
        "scope": "Engineering judgments over saved scenes, not expert annotations or quality evaluation."}
    print(now(), "CPU review-loader/qualification audit; no model, training or scene execution", flush=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    try:
        lock = json.loads(a.lock.read_text())
        for row in lock["files"]:
            assert sha(a.source / row["source_path"]) == row["sha256"]
        critic_lock = json.loads(a.critic_lock.read_text())
        for row in critic_lock["files"]:
            assert sha(a.source / row["path"]) == row["sha256"]
        spec = importlib.util.spec_from_file_location("official_critic_validation", a.source / "src_critic/validate_sft.py")
        critic_validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(critic_validation)
        sys.path.insert(0, str(a.source / "src"))
        import sft_dataset as upstream
        from scene_observation import configure_image_budget
        import torch
        from transformers import AutoProcessor
        from PIL import Image
        processor = AutoProcessor.from_pretrained(root / "resources/models/Scene-Builder-Qwen3-VL-8B-Instruct/63e235a0c0d6e5bc2f8f3485214df4457fd2e672", local_files_only=True)
        configure_image_budget(processor, 65536)
        refs = {}

        def ref(path):
            value = {"path": str(path.resolve().relative_to(root)), "sha256": sha(path)}
            refs[value["path"]] = value["sha256"]
            return value

        run = root / "runtime/P920-DD-v1_20261104T083214Z"
        start = ref(run / "drafts/B/checkpoint.json")
        brief_ref = ref(root / "inputs_v1/巴黎开间双条件/输入B.json")
        brief = json.loads(checked_ref(root, brief_ref).read_text())["brief"]
        candidates = {}
        for cid, directory in [("draft_B", run / "drafts/B"),
                               ("revision_B_G", run / "revisions/B_G/tools"),
                               ("revision_B_U", run / "revisions/B_U/tools")]:
            receipt = json.loads((directory / "views/receipt.json").read_text())
            candidates[cid] = {"candidate_id": cid, "comparison_start": start,
                "checkpoint": ref(directory / "checkpoint.json"),
                "scene": ref(directory / "partial_scene.glb"),
                "render_receipt": ref(directory / "views/receipt.json"),
                "views": [ref(Path(v["path"])) for v in receipt["views"]]}
        requested = {"dining_chair_0", "dining_chair_1", "dining_table_0", "nightstand_0", "wardrobe_0"}
        retained = {}
        for cid, candidate in candidates.items():
            table = object_table(read_checkpoint(checked_ref(root, candidate["checkpoint"])))
            retained[cid] = sorted(requested & {o["object_id"] for o in table})
        assert len(retained["draft_B"]) == 5 and not retained["revision_B_G"]
        assert retained["draft_B"] == retained["revision_B_U"]
        evidence = [{"evidence_id": cid, "artifact": c["checkpoint"], "kind": "saved_object_state"}
                    for cid, c in candidates.items()]

        def review(name, ids, verdict, groups, question, unknowns):
            return {"review_id": name, "origin": "engineering_review_fixture",
                "project_id": "paris_19_remote_studio", "family_id": "paris_19_remote_studio",
                "original_brief": brief, "brief_source": brief_ref, "question": question,
                "comparison_start": start, "candidates": [candidates[cid] for cid in ids],
                "judgment": {"verdict": verdict, "ranking_groups": groups, "evidence": evidence,
                    "claims": [{"candidate_id": cid, "object_ids": sorted(requested),
                        "activity": "Track explicitly requested existing instances; no inference of whole-room usability.",
                        "assessment": "ENGINEERING_LABEL_ONLY: retained " + str(len(retained[cid])) + " of five selected original IDs.",
                        "evidence_ids": [cid]} for cid in ids], "unknowns": unknowns}}

        strict = review("identity_strict", ["draft_B", "revision_B_G"], "strict_preference",
            [["draft_B"], ["revision_B_G"]],
            "ENGINEERING FIXTURE: rank only retention of the five named original instances. This is not a room-quality or professional preference label.",
            ["whole-room usefulness and professional preference not established"])
        tied = review("identity_tie", ["draft_B", "revision_B_U"], "tie_or_partial_order",
            [["draft_B", "revision_B_U"]],
            "ENGINEERING FIXTURE: compare retention of the five named instances, allowing a tie on this metric only.",
            ["Equal retained IDs do not establish equal quality."])
        unknown = review("room_unknown", ["draft_B", "revision_B_G"], "insufficient_evidence", [],
            "ENGINEERING FIXTURE: is a whole-room professional preference supported by the available evidence? Preserve insufficiency.",
            ["No independent professional room review; retention counts cannot settle overall usability, comfort or aesthetics."])
        source_records = {"strict": strict, "tie": tied, "unknown": unknown}
        for name, record in source_records.items():
            save(out / "review_records" / (name + ".json"), record)
        episodes = {name: export_review(record, root) for name, record in source_records.items()}
        episodes["strict_reversed"] = export_review(strict, root, ["revision_B_G", "draft_B"])
        checks, metrics, tensors = {}, {}, {}
        for name, episode in episodes.items():
            directory = out / "fixtures" / name
            save(directory / "episode.json", episode)
            ds = upstream.StepwiseSFTDataset(directory, root, processor, max_length=16384)
            assert len(ds) == 1
            sample, example = ds.samples[0], ds[0]
            full, prompt = upstream._build_messages(sample)
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            paths = sample.qa_images
            images = []
            for path in paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            try:
                prefix = processor(text=[text], images=images, return_tensors="pt")["input_ids"][0]
            finally:
                for im in images:
                    im.close()
            n = len(prefix)
            assert torch.equal(example["input_ids"][:n], prefix)
            assert bool((example["labels"][:n] == -100).all())
            assert torch.equal(example["labels"][n:], example["input_ids"][n:])
            target = processor.tokenizer.decode(example["labels"][n:], skip_special_tokens=False)
            decoded = decode_review_answer(target, episode["provenance"]["candidate_slots"])
            record = strict if name == "strict_reversed" else source_records[name]
            assert decoded["verdict"] == record["judgment"]["verdict"]
            assert decoded["ranking_groups"] == record["judgment"]["ranking_groups"]
            assert "ENGINEERING_LABEL_ONLY" not in text and "ENGINEERING_LABEL_ONLY" in target
            assert brief in text
            assert len(paths) == (2 if episode["stage"] == "critic" else 10)
            if episode["stage"] == "critic":
                assert full[0]["role"] == "user" and len(full) == 2
            metrics[name] = {"stage": episode["stage"], "images": len(paths), "masked_input_tokens": n,
                "supervised_tokens": len(example["labels"]) - n, "decoded_answer": decoded,
                "episode_sha256": sha(directory / "episode.json"),
                "labels_sha256": hashlib.sha256(example["labels"].numpy().tobytes()).hexdigest(),
                "source_views_have_native_ID_annotations": False}
            tensors[name] = example
            print(name, episode["stage"], len(paths), "images;", n, "masked tokens", flush=True)
        checks["positional_labels_remapped"] = episodes["strict"]["answer"] == "AB" and episodes["strict_reversed"]["answer"] == "BA"
        checks["reversal_changes_actual_image_tensor"] = not torch.equal(tensors["strict"]["pixel_values"], tensors["strict_reversed"]["pixel_values"])
        checks["all_verdicts_round_trip_without_forcing_preference"] = True
        checks["all_inputs_images_masked_and_targets_isolated"] = True
        checks["official_ranking_parser_rejects_explicit_tie"] = critic_validation._extract_answer("<answer>A=B</answer>") == []

        def must_reject(name, operation):
            try:
                operation()
            except (ValueError, FileNotFoundError, FileExistsError):
                checks[name] = True
            else:
                raise AssertionError("Expected rejection: " + name)

        bad = copy.deepcopy(strict)
        bad["candidates"][0]["views"][0] = candidates["revision_B_G"]["views"][0]
        must_reject("view_from_other_candidate_rejected", lambda: export_review(bad, root))
        bad_unknown = copy.deepcopy(unknown)
        bad_unknown["judgment"]["ranking_groups"] = [["draft_B"], ["revision_B_G"]]
        must_reject("unknown_with_fabricated_ranking_rejected", lambda: export_review(bad_unknown, root))
        missing = copy.deepcopy(tied)
        missing["candidates"][0]["views"][0]["path"] = "missing_review_view_7319.png"
        must_reject("QA_adapter_missing_view_rejected", lambda: export_review(missing, root))
        raw_qa = copy.deepcopy(episodes["tie"])
        raw_qa["turns"][0]["image"][0] = "missing_review_view_7319.png"
        save(out / "invalid" / "bare_qa_missing_image.json", raw_qa)
        bare = upstream.build_step_samples(out / "invalid/bare_qa_missing_image.json", root)
        checks["upstream_QA_drops_missing_image_without_error"] = len(bare) == 1 and len(bare[0].qa_images) == 9
        raw_critic = copy.deepcopy(episodes["strict"])
        raw_critic["images"][0] = "missing_review_view_7319.png"
        save(out / "invalid/bare_critic_missing_image.json", raw_critic)
        must_reject("upstream_critic_missing_image_rejected", lambda: upstream.build_step_samples(out / "invalid/bare_critic_missing_image.json", root))

        mutated = copy.deepcopy(episodes["unknown"])
        mutated["evaluation_only"] = {"preferred": "FORBIDDEN_EVAL_SENTINEL", "score": 999}
        save(out / "metadata_variant/episode.json", mutated)
        other = upstream.StepwiseSFTDataset(out / "metadata_variant", root, processor, max_length=16384)[0]
        checks["evaluation_metadata_cannot_change_actual_tensors"] = all(torch.equal(other[k], tensors["unknown"][k]) for k in other)

        selected = [{"sample_id": name, "episode": ref(out / "fixtures" / name / "episode.json")}
                    for name in ("strict", "unknown")]
        save(out / "fixtures/strict/unlisted_neighbor.json", episodes["tie"])
        _copy_selected_episodes(selected, root, out / "audit_selected_loader_inputs")
        selected_ds = upstream.StepwiseSFTDataset(out / "audit_selected_loader_inputs", root, processor, max_length=16384)
        checks["whitelist_excludes_neighboring_episode"] = len(selected_ds) == 2 and {s.source_file for s in selected_ds.samples} == {"strict.json", "unknown.json"}
        for i in range(len(selected_ds)):
            selected_ds[i]  # Exercise actual tokenization, not just file counting.
        dependencies = [v for candidate in candidates.values() for v in candidate["views"]]
        manifest = {"purpose": "professional_sft_preparation", "project_registry": ref(a.registry),
            "retrieval_project_ids": [], "samples": [dict(selected[0], project_id="paris_19_remote_studio",
                family_id="paris_19_remote_studio", derived_at=now(), dependencies=dependencies)]}
        save(out / "qualification/current_engineering_manifest.json", manifest)
        findings = qualification_findings(manifest, root)
        checks["current_engineering_sample_not_qualified"] = any("episode_not_qualified" in x for x in findings)
        must_reject("unqualified_training_directory_not_created", lambda: prepare_training_subset(manifest, root, out / "forbidden_training_subset"))
        assert not (out / "forbidden_training_subset").exists()
        registry = json.loads(a.registry.read_text())
        registry["projects"].append({"project_id": "engineering_alias_TEST_ONLY", "family_id": "paris_19_remote_studio", "allocation": "sealed_test", "training_eligible": False})
        save(out / "qualification/registry_with_deliberate_leak.json", registry)
        leaking = copy.deepcopy(manifest)
        leaking["project_registry"] = ref(out / "qualification/registry_with_deliberate_leak.json")
        leaking["retrieval_project_ids"] = ["engineering_alias_TEST_ONLY"]
        leak_findings = qualification_findings(leaking, root)
        checks["same_family_across_splits_rejected"] = any(x.startswith("family_crosses_splits:") for x in leak_findings)
        checks["sealed_test_retrieval_rejected"] = any(x.startswith("unapproved_retrieval_project:") for x in leak_findings)
        tampered = copy.deepcopy(manifest)
        tampered["samples"][0]["episode"]["sha256"] = "0" * 64
        must_reject("changed_episode_hash_rejected", lambda: prepare_training_subset(tampered, root, out / "changed_hash_subset"))
        checks["cuda_not_initialized"] = not torch.cuda.is_initialized()
        assert all(checks.values()), checks
        report.update(status="passed_engineering_only", exit_code=0, checks=checks, fixtures=metrics,
            current_qualification_findings=findings, deliberate_leak_findings=leak_findings, input_hashes=refs,
            training_subset_created=False, audit_only_subset_samples=2,
            limitation="No real approved professional sample exists. Copy whitelist and rejection behavior tested; positive professional admission and semantic expert judgments remain unverified.")
    except Exception as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_at"] = now()
        save(out / "RESULT.json", report)
        print(now(), report["status"], flush=True)


if __name__ == "__main__":
    main()
