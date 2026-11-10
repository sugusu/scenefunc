"""Compare actual causal conditions and build a same-prefix split SFT control.

No neural model, trainer, generated labels, or simulator. The optional processor
check uses only source views that are available with verified original hashes.
The split records are teacher-forcing controls, not a new inference algorithm.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from native_multistage_phase import PHASE_SUFFIX


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def load_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def prompt_ids(tok, row):
    text = tok.apply_chat_template(row["messages"], tokenize=False,
                                   add_generation_prompt=True) + row["assistant_prefix"]
    ids = tok.encode(text, add_special_tokens=False)
    if tok.encode(text + row["target_text"], add_special_tokens=False) != ids + row["target_token_ids"]:
        raise ValueError("Tokenizer prompt/target boundary changed")
    return ids


def measure(tok, rows):
    result = {}
    for row in rows:
        prefix = prompt_ids(tok, row)
        images = [item for message in row["messages"] if isinstance(message["content"], list)
                  for item in message["content"] if item.get("type") == "image"]
        state = hashlib.sha256(canonical(images))
        for token in prefix:
            state.update(struct.pack("<q", token))
        for token, weight, origin in zip(row["target_token_ids"], row["token_objective_weights"],
                                         row["source_token_origins"], strict=True):
            if weight > 0:
                key = (row["unit_id"], json.dumps(origin, sort_keys=True))
                if key in result:
                    raise ValueError("Duplicate original positive target")
                result[key] = dict(token=token, weight=weight, context_sha256=state.hexdigest(),
                                   context_tokens=len(prefix), stage=origin["component"])
            state.update(struct.pack("<q", token))
            prefix.append(token)
    return result


def compare(left, right):
    if left.keys() != right.keys():
        raise ValueError("Source targets differ")
    counts = {}
    for key in left:
        a, b = left[key], right[key]
        if (a["token"], a["weight"]) != (b["token"], b["weight"]):
            raise ValueError("Token/coefficient changed")
        unit, _ = key
        entry = counts.setdefault(unit, dict(unit_id=unit, positive_tokens=0,
                                              equal_context_tokens=0, different_context_tokens=0,
                                              by_stage={}))
        entry["positive_tokens"] += 1
        same = a["context_sha256"] == b["context_sha256"]
        entry["equal_context_tokens" if same else "different_context_tokens"] += 1
        stage = entry["by_stage"].setdefault(a["stage"], dict(equal=0, different=0))
        stage["equal" if same else "different"] += 1
    return dict(positive_tokens=len(left), equal_context_tokens=sum(x["equal_context_tokens"] for x in counts.values()),
                different_context_tokens=sum(x["different_context_tokens"] for x in counts.values()),
                units=list(counts.values()))


def matched_split(joint, tokenizer):
    """Preserve joint prompt, all positive targets and ALL conditioning tokens."""
    output = []
    close = tokenizer.convert_tokens_to_ids("</think>")
    for source in joint:
        ids = source["target_token_ids"]
        # Phase boundaries are syntax, NOT original loss-origin categories.
        # Two terminal records have <think> text whose immutable origin is A.
        decision_end = ids.index(close) + 1 if close in ids else 0
        action_start = decision_end
        while action_start < len(ids) and source["token_objective_weights"][action_start] == 0:
            action_start += 1
        if action_start == len(ids):
            raise ValueError("No positive action/end-of-turn target")
        ranges = [("action", action_start, len(ids))]
        if decision_end:
            if any(source["token_objective_weights"][decision_end:action_start]):
                raise ValueError("Would drop a positive interphase target")
            ranges.insert(0, ("decision", 0, decision_end))
        for stage, start, end in ranges:
            row = copy.deepcopy(source)
            row.update(arm="native_phase_v4:joint_prefix_split_control", stage=stage,
                       assistant_prefix=tokenizer.decode(ids[:start], skip_special_tokens=False),
                       assistant_prefix_token_ids=ids[:start],
                       decision_stop_token_id=close if stage == "decision" else None,
                       target_token_ids=ids[start:end],
                       target_text=tokenizer.decode(ids[start:end], skip_special_tokens=False),
                       source_token_origins=source["source_token_origins"][start:end],
                       token_objective_weights=source["token_objective_weights"][start:end],
                       training_admitted=False, image_encoding_checked=False,
                       inference_equivalence_verified=False,
                       phase_stage_does_not_relabel_source_loss_origin=True,
                       control_scope="Same teacher-forced token conditionals; no new deployment policy")
            row["target_sha256"] = digest(json.dumps(row["target_token_ids"]).encode())
            row["response_loss_mask"] = [w > 0 for w in row["token_objective_weights"]]
            row["text_prefix_tokens"] = len(prompt_ids(tokenizer, row))
            output.append(row)
    return output


def processor_check(repo, checkpoint, joint, split, source_path):
    import importlib.util
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from native_feedback_batch import make_scoring_inputs
    from routed_multimodal import resolve_messages, encode_component, tensor_digest

    torch.set_num_threads(1)
    availability_path = repo / "runs/autodl/20261110_multimodal_material/source_image_lookup/image_availability.json"
    availability = json.loads(availability_path.read_text())
    # Explicit path avoids silently replacing a missing original helper.
    if source_path is None:
        raise ValueError("--image-budget-source is required for the processor check")
    if digest(source_path.read_bytes()) != "1a27dd84bdf25ba771c64f59eacc165fd3e5b8365bc5ba08c1c2856cd51734f3":
        raise ValueError("Image budget helper changed")
    spec = importlib.util.spec_from_file_location("pinned_image_budget", source_path)
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    source_module.configure_image_budget(processor, 65536)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    views = {r["unit_id"]: r for r in availability["rows"]}
    records = []
    for source in joint:
        view = views[source["unit_id"]]
        if not view["complete_five_views"]:
            records.append(dict(unit_id=source["unit_id"], status="original_views_unavailable"))
            continue
        messages, paths = resolve_messages(source["messages"], view["images"])
        images = []
        try:
            for path in paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            batch, boundary = encode_component(processor, dict(source, component="joint",
                decision_target_mask=[o["component"] == "decision" for o in source["source_token_origins"]],
                action_target_mask=[o["component"] == "action" for o in source["source_token_origins"]]), messages, images)
            prompt_batch = {k: v[:, :boundary] if k in ("input_ids", "attention_mask") else v
                            for k, v in batch.items() if k not in ("labels", "action_mask", "decision_mask")}
            full_native, _ = make_scoring_inputs(prompt_batch, batch["input_ids"][:, boundary:], config=config)
            pieces = []
            for row in [r for r in split if r["unit_id"] == source["unit_id"]]:
                part, part_boundary = encode_component(processor, dict(row, component=row["stage"]), messages, images)
                pp = {k: v[:, :part_boundary] if k in ("input_ids", "attention_mask") else v
                      for k, v in part.items() if k not in ("labels", "action_mask", "decision_mask")}
                native, _ = make_scoring_inputs(pp, part["input_ids"][:, part_boundary:], config=config)
                length = native["input_ids"].shape[1]
                for name in ("input_ids", "attention_mask", "position_ids"):
                    if not torch.equal(native[name], full_native[name][..., :length]):
                        raise ValueError("Matched split changed multimodal causal input: " + name)
                for name in ("pixel_values", "image_grid_thw"):
                    if not torch.equal(part[name], batch[name]):
                        raise ValueError("Matched split changed image tensors")
                pieces.append(dict(stage=row["stage"], tokens=length, boundary=part_boundary,
                                   position_ids_sha256=tensor_digest(native["position_ids"]),
                                   input_ids_sha256=tensor_digest(native["input_ids"])))
            records.append(dict(unit_id=source["unit_id"], status="exact_processor_and_mRoPE_prefixes_equal",
                                pieces=pieces, image_bytes_verified=True))
        finally:
            for image in images:
                image.close()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only audit initialized CUDA")
    return dict(availability_sha256=digest(availability_path.read_bytes()),
                processor_units=sum(r["status"] == "exact_processor_and_mRoPE_prefixes_equal" for r in records),
                missing_units=sum(r["status"] == "original_views_unavailable" for r in records), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-check", action="store_true")
    parser.add_argument("--image-budget-source", type=Path)
    args = parser.parse_args()
    tick = time.monotonic()
    repo, out = args.repo.resolve(), args.output.resolve()
    if not (out.is_relative_to(repo / "runs/autodl/20261111_phase_context_equivalence") or out.is_relative_to(Path("/dev/shm"))):
        raise ValueError("Write only this study or RAM")
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    root = repo / "runs/autodl/20261111_native_phase_batch_bridge/material_v4_2"
    paths = [root / "prospective_native_baselines_v4.jsonl.gz", root / "prospective_native_phase_v4.jsonl.gz"]
    before = {str(p.relative_to(repo)): digest(p.read_bytes()) for p in paths}
    joint = [r for r in load_rows(paths[0]) if r["arm"].endswith(":native_joint")]
    paired = [r for r in load_rows(paths[1]) if r["arm"].endswith(":decision_conditioned:full_record")]
    normalized = copy.deepcopy(paired)
    for row in normalized:
        suffix = PHASE_SUFFIX[row["stage"]]
        if not row["messages"][0]["content"].endswith(suffix):
            raise ValueError("Prepared phase prompt differs from current source")
        row["messages"][0]["content"] = row["messages"][0]["content"][:-len(suffix)]
    split = matched_split(joint, tok)
    reference = measure(tok, joint)
    comparisons = {"current_P_versus_J": compare(reference, measure(tok, paired)),
                   "phase_suffix_removed_P_versus_J": compare(reference, measure(tok, normalized)),
                   "new_joint_prefix_split_versus_J": compare(reference, measure(tok, split))}
    if comparisons["new_joint_prefix_split_versus_J"]["different_context_tokens"]:
        raise ValueError("Split control failed causal equality")
    processor = processor_check(repo, args.checkpoint, joint, split, args.image_budget_source) if args.processor_check else None
    if before != {str(p.relative_to(repo)): digest(p.read_bytes()) for p in paths}:
        raise RuntimeError("Old source changed")
    out.mkdir(parents=True)
    (out / "matched_joint_prefix_split.jsonl.gz").write_bytes(gzip.compress(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in split).encode(), mtime=0))
    result = dict(source_hashes=before, sources_unchanged=True, units=len(joint), split_components=len(split),
                  comparisons=comparisons, processor=processor,
                  coefficient_mass=math.fsum(r["weight"] for r in reference.values()),
                  materialized_control_inference_status="Not an independently validated two-call generation policy",
                  loss_gradient_claim="Conditional analytical identity with same model/stochasticity/parameters and full-corpus sum before update; no neural gradient measured",
                  model_weights_loaded=False, neural_forwards=0, generations=0, training_updates=0,
                  new_scene_trials=0, paid_API_calls=0, seconds=time.monotonic()-tick)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "processor")}))
    print(json.dumps({k: {x: y for x, y in v.items() if x != "units"} for k, v in comparisons.items()}))
    if processor:
        print(json.dumps({k: v for k, v in processor.items() if k != "records"}))


if __name__ == "__main__":
    main()
