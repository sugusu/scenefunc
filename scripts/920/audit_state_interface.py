"""CPU-only audit of saved identities, real receipts and explicit OTA context."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from decision_state import (canonical, engineering_episode, object_table,
                            read_checkpoint, sha, validate_document,
                            verify_removals_or_failures)

SOURCE_RUN = "P920-DD-v1_20261104T083214Z"
MODEL_REVISION = "63e235a0c0d6e5bc2f8f3485214df4457fd2e672"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "lock", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    root, source, out = a.root.resolve(), a.source.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {"started_at": now(), "status": "running", "model_calls": 0,
              "parameter_updates": 0, "scene_edits": 0, "training_eligible": False,
              "scope": "Saved-record consistency and CPU loader checks; no engine replay or design-quality inference."}
    save(out / "RESULT.json", report)
    print(now(), "CPU-only saved-state and authentic-receipt audit; no model/scene execution", flush=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    try:
        lock = json.loads(a.lock.read_text())
        for entry in lock["files"]:
            if sha(source / entry["source_path"]) != entry["sha256"]:
                raise ValueError("Upstream source lock mismatch")
        report["upstream_commit"] = lock["commit"]
        sys.path.insert(0, str(source / "src"))
        import memory
        import sft_dataset as upstream
        import torch
        from transformers import AutoProcessor
        from scene_observation import configure_image_budget
        from PIL import Image
        processor = AutoProcessor.from_pretrained(
            root / "resources/models/Scene-Builder-Qwen3-VL-8B-Instruct" / MODEL_REVISION,
            local_files_only=True)
        configure_image_budget(processor, 65536)
        run = root / "runtime" / SOURCE_RUN
        source_files, records, fixtures = {}, {}, {}

        def read(path):
            source_files[str(path.relative_to(root))] = sha(path)
            return json.loads(path.read_text())

        def replay(history):
            state = memory.empty_memory("furniture")
            for event in history:
                memory.HANDLERS["furniture"](state, event["call"]["name"],
                    event["call"]["arguments"], event["response"])
            return state

        def verify_memory(state, reconstructed):
            placed = {k: v for k, v in reconstructed["assets"].items() if v.get("placed")}
            rows = object_table(state)
            assert set(placed) == {row["object_id"] for row in rows}
            for row in rows:
                entry = placed[row["object_id"]]
                assert entry["source_asset"] == row["template_asset_id"]
                assert all(math.isclose(float(x), float(y), abs_tol=1e-7)
                           for x, y in zip(entry["pos"], row["transform"]["translation"][:2]))
                yaw = math.radians(entry["rot"])
                expected = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                actual = row["transform"]["rotation_wxyz"]
                # q and -q denote the same orientation.
                assert any(all(math.isclose(x, sign*y, abs_tol=1e-7)
                               for x, y in zip(actual, expected)) for sign in (1, -1))
            return sorted(placed)

        for label in ("A_G", "A_U", "B_G", "B_U"):
            arm = label[0]
            before_path = run / "drafts" / arm / "checkpoint.json"
            app_path = run / "revisions" / label / "application.json"
            application = read(app_path)
            assert application["source_checkpoint_sha256"] == sha(before_path)
            after_path = Path(application["final_directory"]) / "checkpoint.json"
            for path in (before_path, after_path):
                source_files[str(path.relative_to(root))] = sha(path)
            before, after = read_checkpoint(before_path), read_checkpoint(after_path)
            calls = read(run / "model_runs" / ("formal_" + label) / "parsed_calls.json")
            events = read(Path(application["final_directory"]) / "tool_responses.json")
            assert events == application["new_tool_events"]
            assert calls == [event["call"] for event in events]
            transition = verify_removals_or_failures(before, after, events)
            before_mem, after_mem = replay(before["tool_history"]), replay(after["tool_history"])
            before_ids, after_ids = verify_memory(before, before_mem), verify_memory(after, after_mem)
            request_path = run / "comparison_inputs" / label / "request.json"
            request = read(request_path)
            receipt = read(request_path.parent / "receipt.json")
            assert receipt["request_sha256"] == sha(request_path)
            assert receipt["checkpoint_sha256"] == sha(before_path)
            views = []
            for view in receipt["encoding"]["images"]:
                path = Path(view["origin"])
                assert sha(path) == view["sha256"]
                source_files[str(path.relative_to(root))] = view["sha256"]
                views.append(str(path.relative_to(root)))
            assert len(views) == 5
            brief_path = root / "inputs_v1/巴黎开间双条件" / ("输入" + arm + ".json")
            brief = read(brief_path)["brief"]
            assert brief in request[0]["content"]
            episode = engineering_episode(state=before, original_brief=brief,
                system=request[0]["content"], calls=calls, views=views,
                provenance={"source_run": SOURCE_RUN, "source_label": label,
                    "project_family": "paris_19_remote_studio", "split": "development",
                    "checkpoint_sha256": sha(before_path), "request_sha256": sha(request_path)})
            directory = out / "episodes" / label
            save(directory / "episode.json", episode)
            dataset = upstream.StepwiseSFTDataset(directory, root, processor, max_length=16384)
            assert len(dataset) == 1  # A six-call batch has one real observation.
            sample, example = dataset.samples[0], dataset[0]
            assert sample.memory_str == "# Memory snapshot (empty)"
            full, prompt = upstream._build_messages(sample)
            prompt_text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            images = []
            for view in views:
                with Image.open(root / view) as im:
                    images.append(im.convert("RGB"))
            try:
                prompt_ids = processor(text=[prompt_text], images=images, return_tensors="pt")["input_ids"][0]
            finally:
                for im in images:
                    im.close()
            n = len(prompt_ids)
            assert torch.equal(example["input_ids"][:n], prompt_ids)
            assert bool((example["labels"][:n] == -100).all())
            target_ids = example["labels"][n:]
            assert torch.equal(target_ids, example["input_ids"][n:])
            target = processor.tokenizer.decode(target_ids, skip_special_tokens=False)
            assert brief in prompt_text
            assert all(row["object_id"] in prompt_text and row["catalog_asset_id"] in prompt_text
                       for row in object_table(before))
            assert "recent_actual_tool_receipts" in prompt_text
            assert "recent_actual_tool_receipts" not in target
            assert "EVAL_ONLY_STATE_SENTINEL" not in full_text
            mutated = copy.deepcopy(episode)
            mutated["evaluation_only"] = {"marker": "EVAL_ONLY_STATE_SENTINEL", "after": object_table(after)}
            mutation_dir = out / "evaluation_metadata_variants" / label
            save(mutation_dir / "episode.json", mutated)
            second = upstream.StepwiseSFTDataset(mutation_dir, root, processor, max_length=16384)[0]
            assert set(example) == set(second)
            assert all(torch.equal(example[k], second[k]) for k in example)
            record = dict(transition, before_count=len(before_ids), after_count=len(after_ids),
                original_batch_tool_calls=len(calls), official_samples=1,
                official_history_replay_matches_saved_ids_and_poses=True,
                all_prestate_ids_and_catalog_assets_reach_prompt=True,
                prompt_and_receipts_masked=True, future_metadata_tensor_invariant=True,
                prompt_tokens=n, supervised_tokens=len(target_ids), input_tokens=len(example["input_ids"]),
                episode_sha256=sha(directory / "episode.json"), decoded_action_target=target,
                labels_sha256=hashlib.sha256(example["labels"].numpy().tobytes()).hexdigest())
            records[label] = record
            save(out / "consequences" / (label + ".json"), {"training_eligible": False,
                "events": events, "before_objects": object_table(before), "after_objects": object_table(after),
                "consistency": transition, "whole_room_quality": "unknown"})
            fixtures[label] = (before, after, events)
            print(label, len(before_ids), "->", len(after_ids), "objects;",
                  len(calls), "calls;", n, "masked tokens", flush=True)

        negative_checks = {}
        b_before, b_after, b_events = fixtures["B_G"]
        corrupt = json.loads((run / "drafts/B/checkpoint.json").read_text())
        corrupt["payload"]["placed_instances"][0]["object_id"] = "tampered_instance"
        try:
            validate_document(corrupt)
        except ValueError:
            negative_checks["tampered_checkpoint_rejected"] = True
        fake_events, fake_after = copy.deepcopy(b_events), copy.deepcopy(b_after)
        fake_events[-1]["response"]["success"] = True
        fake_after["tool_history"] = b_before["tool_history"] + fake_events
        try:
            verify_removals_or_failures(b_before, fake_after, fake_events)
        except ValueError:
            negative_checks["invented_success_for_absent_object_rejected"] = True
        wrong_after = copy.deepcopy(b_after)
        wrong_after["placed_instances"][0]["transform"]["translation"][0] += 0.25
        try:
            verify_removals_or_failures(b_before, wrong_after, b_events)
        except ValueError:
            negative_checks["unrecorded_survivor_move_rejected"] = True
        assert len(negative_checks) == 3
        rows = {r["object_id"]: r for r in object_table(b_before)}
        assert rows["dining_chair_0"]["template_asset_id"] == rows["dining_chair_1"]["template_asset_id"]
        assert rows["dining_chair_0"]["object_id"] != rows["dining_chair_1"]["object_id"]
        assert not torch.cuda.is_initialized()
        report.update(status="passed_engineering_only", exit_code=0, cases=records,
            negative_checks=negative_checks, source_files=source_files,
            shared_asset_instances_kept_distinct=True, cuda_initialized=False,
            limitations=["No new scene execution; memory reconstruction is not an engine replay.",
                "Explicit pre-state enters task_context; upstream memory still starts empty.",
                "Four extracted batches are engineering fixtures, not expert demonstrations.",
                "No intermediate images invented for the six-call batch.",
                "Professional project data, OTC review and fair training comparison remain open."])
    except Exception as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_at"] = now()
        save(out / "RESULT.json", report)
        print(now(), report["status"], flush=True)


if __name__ == "__main__":
    main()
