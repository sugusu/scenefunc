"""CPU audit of the pinned upstream SFT loader; no model or parameter updates.

Fixtures deliberately use researcher-written engineering text, not professional
supervision. Reused Paris views only exercise multimodal plumbing. Nothing in
this audit establishes a qualified project sample, a good edit, or training gain.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REVISION = "a0565e50ed7d32c0e09132b9311c0b9b91165405"
MODEL_REVISION = "63e235a0c0d6e5bc2f8f3485214df4457fd2e672"
SOURCE_RUN = "P920-DD-v1_20261104T083214Z"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, source, out = (p.resolve() for p in (args.root, args.source, args.out))
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {
        "started_at": now(), "status": "running", "upstream_revision": REVISION,
        "model_revision": MODEL_REVISION, "training": False,
        "model_weights_loaded": False, "model_calls": 0,
        "qualified_professional_samples": 0,
        "scope": "Engineering loader fixtures only; no scene edits or design-quality conclusions.",
    }
    save(out / "RESULT.json", report)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    sys.dont_write_bytecode = True
    print(now(), "CPU-only official SFT loader audit; no model loading", flush=True)
    try:
        lock = json.loads(args.lock.read_text())
        assert lock["commit"] == REVISION
        for entry in lock["files"]:
            p = source / entry["source_path"]
            assert digest(p) == entry["sha256"], p
        report["source_lock_sha256"] = digest(args.lock)

        import torch
        from transformers import AutoProcessor
        sys.path.insert(0, str(source / "src"))
        import sft_dataset as upstream
        from scene_observation import configure_image_budget

        weights = root / "resources/models/Scene-Builder-Qwen3-VL-8B-Instruct" / MODEL_REVISION
        processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
        configure_image_budget(processor, 65536)
        checkpoint = root / "runtime" / SOURCE_RUN / "drafts/A/checkpoint.json"
        placed = json.loads(checkpoint.read_text())["payload"]["placed_instances"]
        state = [{"object_id": x["object_id"], "asset_id": x["metadata"]["original_asset"],
                  "transform": x["transform"]} for x in placed]
        views = [root / "runtime" / SOURCE_RUN / "drafts/A/views" / ("current_state." + v + ".png")
                 for v in ("topdown", "south", "east", "west", "north")]
        assert len(state) == 9 and all(p.is_file() for p in views)
        report["source_observation"] = {"checkpoint": str(checkpoint.relative_to(root)),
            "checkpoint_sha256": digest(checkpoint),
            "views": [{"path": str(p.relative_to(root)), "sha256": digest(p)} for p in views]}
        task = ("ENGINEERING LOADER FIXTURE; not expert data. Keep wardrobe_0 as the same instance. "
                "The example tool call is not executed or quality validated.\nCurrent objects:\n" +
                json.dumps(state, ensure_ascii=False, sort_keys=True))
        rationale = ("ENGINEERING_REASON_ONLY: Preserve the existing wardrobe_0 instance while "
                     "editing desk_0. A replacement wardrobe with the same shape would lose identity.")
        desk = next(x for x in placed if x["object_id"] == "desk_0")
        tx, ty, _ = desk["transform"]["translation"]
        # No claimed consequence: a legal-shaped proposal for tokenizer inspection only.
        call = {"name": "move_furniture_tool", "arguments":
                {"object_id": "desk_0", "x": tx, "y": ty, "yaw": 0.0}}
        base = {"stage": "furniture", "system": "Inspect the task and emit a furniture tool call.",
                "task_context": {"scene_description": task},
                "project_pair_id": "engineering_pair_A",
                "provenance": {"kind": "researcher_engineering_fixture", "training_eligible": False},
                "evaluation_only": "EVAL_ONLY_SENTINEL_DO_NOT_TRAIN_7319",
                "turns": [{"role": "assistant", "content": "", "images":
                           [str(p.relative_to(root)) for p in views], "tool_calls": [call]}]}
        variants = {"separate_action": base}
        variants["metadata_only_changed"] = copy.deepcopy(base)
        variants["metadata_only_changed"]["project_pair_id"] = "engineering_pair_B"
        variants["metadata_only_changed"]["paired_reason"] = rationale
        variants["paired"] = copy.deepcopy(base)
        variants["paired"]["turns"][0]["content"] = rationale
        variants["separate_reason"] = {
            "stage": "qa", "system": base["system"], "provenance": base["provenance"],
            "turns": [{"role": "user", "content": task,
                       "image": [str(p.relative_to(root)) for p in views]},
                      {"role": "assistant", "content": "", "answer": rationale}]}
        variants["initial_state_metadata_only"] = copy.deepcopy(base)
        variants["initial_state_metadata_only"]["initial_state"] = {
            "objects": state, "marker": "INITIAL_STATE_SENTINEL_7319"}
        datasets, tensors, records = {}, {}, {}
        for name, document in variants.items():
            directory = out / "fixtures" / name
            save(directory / "episode.json", document)
            dataset = upstream.StepwiseSFTDataset(directory, root, processor, max_length=8192)
            assert len(dataset) == 1
            sample = dataset.samples[0]
            example = dataset[0]
            full, prompt = upstream._build_messages(sample)
            prompt_text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            prompt_ids = processor(text=[prompt_text], images=[__import__('PIL').Image.open(p).convert('RGB') for p in views], return_tensors='pt')["input_ids"][0]
            assert torch.equal(example["input_ids"][:len(prompt_ids)], prompt_ids)
            labels = example["labels"]
            assert bool((labels[:len(prompt_ids)] == -100).all())
            assert torch.equal(labels[len(prompt_ids):], example["input_ids"][len(prompt_ids):])
            target = processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
            assert "EVAL_ONLY_SENTINEL" not in full_text
            record = {"fixture_sha256": digest(directory / "episode.json"),
                      "stage": sample.stage, "input_tokens": len(example["input_ids"]),
                      "prompt_tokens_masked": len(prompt_ids),
                      "supervised_tokens": int((labels != -100).sum()),
                      "image_count": len(sample.scene_images or sample.qa_images),
                      "memory": sample.memory_str, "decoded_supervised_target": target,
                      "input_ids_sha256": hashlib.sha256(example["input_ids"].numpy().tobytes()).hexdigest(),
                      "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                      "prompt_is_exact_prefix": True, "loss_mask_matches_assistant_span": True}
            datasets[name], tensors[name], records[name] = dataset, example, record
            print(name, "tokens", record["input_tokens"], "supervised", record["supervised_tokens"], flush=True)
        same = lambda a, b: all(torch.equal(tensors[a][k], tensors[b][k]) for k in tensors[a])
        checks = {
            "metadata_pair_change_has_no_effect": same("separate_action", "metadata_only_changed"),
            "initial_state_metadata_ignored": same("separate_action", "initial_state_metadata_only"),
            "paired_reason_changes_actual_labels": not torch.equal(tensors["paired"]["labels"], tensors["separate_action"]["labels"]),
            "paired_reason_reaches_target": rationale in records["paired"]["decoded_supervised_target"],
            "same_reason_reaches_separate_qa_target": rationale in records["separate_reason"]["decoded_supervised_target"],
            "reason_absent_from_separate_action_target": rationale not in records["separate_action"]["decoded_supervised_target"],
            "same_action_in_both_groups": variants["paired"]["turns"][0]["tool_calls"] == variants["separate_action"]["turns"][0]["tool_calls"],
            "all_five_views_preserved": all(x["image_count"] == 5 for x in records.values()),
        }
        for name, images in [("missing_view", views[:-1]), ("wrong_order", list(reversed(views)))]:
            invalid = copy.deepcopy(base)
            invalid["turns"][0]["images"] = [str(p.relative_to(root)) for p in images]
            p = out / "invalid_fixtures" / (name + ".json"); save(p, invalid)
            try:
                upstream.build_step_samples(p, root)
            except ValueError as error:
                checks[name + "_rejected"] = True
                report[name + "_error"] = str(error)
            else:
                checks[name + "_rejected"] = False
        collator = upstream.StepwiseCollator(processor.tokenizer.pad_token_id)
        batch = collator([tensors["separate_action"], tensors["paired"]])
        length = len(tensors["separate_action"]["input_ids"])
        checks["collator_masks_padding"] = bool((batch["labels"][0, length:] == -100).all())
        checks["cuda_not_initialized"] = not torch.cuda.is_initialized()
        assert all(checks.values()), checks
        report.update(checks=checks, fixtures=records,
            group_budgets={"paired": {"samples": 1, "action_examples": 1,
                "supervised_tokens": records["paired"]["supervised_tokens"]},
                "separate": {"samples": 2, "action_examples": 1,
                "supervised_tokens": records["separate_action"]["supervised_tokens"] + records["separate_reason"]["supervised_tokens"]},
                "matched_training_budget": False,
                "note": "Format overhead and sample counts differ. This is a loader audit, not a frozen training comparison."})

        command = [sys.executable, "-m", "src.train_sft", "--dry-run", "--use-cpu",
                   "--model-path", str(weights), "--sft-root", str(out / "fixtures/paired"),
                   "--repo-root", str(root), "--output-dir", str(out / "official_dry_run"),
                   "--max-pixels", "65536", "--report-to", "none", "--dataloader-workers", "0"]
        start = now()
        dry = subprocess.run(command, cwd=source, env=os.environ.copy(), text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (out / "official_dry_run.log").write_text(dry.stdout)
        report["official_entry_dry_run"] = {"command": command, "cwd": str(source),
            "started_at": start, "ended_at": now(), "exit_code": dry.returncode,
            "first_log": dry.stdout.splitlines()[0] if dry.stdout else None}
        assert dry.returncode == 0, "Official dry-run failed; inspect preserved log"
        report.update(status="passed_engineering_only", exit_code=0,
            limitation="No qualified same-project professional sample is available; that pre-training gate remains open.")
    except Exception as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_at"] = now()
        save(out / "RESULT.json", report)
        print(now(), report["status"], flush=True)


if __name__ == "__main__":
    main()
