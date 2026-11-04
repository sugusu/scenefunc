"""Validate the approved local snapshot and prepare a CPU-only engineering input."""
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import sys

ROOT = Path(os.environ.get('P920_RUN_ROOT', Path(__file__).resolve().parent))


def main():
    plan = json.loads((ROOT / "MODEL_DOWNLOAD_PLAN.json").read_text())
    receipt = json.loads((ROOT / "MODEL_DOWNLOAD_RECEIPT.json").read_text())
    assert receipt["status"] == "verified", "Download verification is incomplete"
    model = Path(plan["destination"])
    for item in plan["files"]:
        assert (model / item["path"]).stat().st_size == item["bytes"]
        assert receipt["files"][item["path"]]["status"] == "verified"
    index = json.loads((model / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    names = set()
    tensors = 0
    elements = 0
    data_bytes = 0
    dtypes = {}
    headers = []
    item_sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    for shard in sorted(set(mapping.values())):
        path = model / shard
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            assert 0 < length < 100_000_000
            header = json.loads(f.read(length))
        spans = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            assert name not in names and mapping.get(name) == shard
            names.add(name)
            start, end = info["data_offsets"]
            count = math.prod(info["shape"])
            assert end - start == count * item_sizes[info["dtype"]]
            spans.append((start, end))
            elements += count
            data_bytes += end - start
            dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + count
            tensors += 1
        position = 0
        for start, end in sorted(spans):
            assert start == position and end >= start
            position = end
        assert position + 8 + length == path.stat().st_size
        headers.append({"file": shard, "header_bytes": length, "data_bytes": position})
    assert names == set(mapping)
    assert data_bytes == index["metadata"]["total_size"]
    assert elements == index["metadata"]["total_parameters"]
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    assert config.model_type == "qwen3_vl"
    output = ROOT / "preflight/model_local"
    output.mkdir(exist_ok=True)
    work = ROOT / 'runtime/P920-DD-v1_20261104T083214Z/work'
    sys.path.insert(0, str(work / 'native_local_inference_20261104'))
    from local_vlm import LocalQwenEngine
    engine = LocalQwenEngine(model, output / 'cpu_encode', max_pixels=65536, max_input_tokens=8192)
    prepared = []
    for arm in ("A", "B"):
        request = work / 'paris_A_inference_20261104' / ('request_' + arm + '.json')
        messages = json.loads(request.read_text())
        inputs, encoding = engine.encode(messages)
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
        target = output / (arm + "_engineering_inputs.pt")
        torch.save(dict(inputs), target)
        (output / (arm + "_engineering_messages.json")).write_text(json.dumps(messages, ensure_ascii=False, indent=2) + "\n")
        prepared.append({"arm": arm, "input_tokens": int(inputs["input_ids"].shape[-1]),
                         "shapes": {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)},
                         "image_grid_thw": inputs.get("image_grid_thw").tolist(), "path": str(target),
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(), "encoding": encoding})
    result = {"status": "pass", "kind": "cpu_model_and_processor_validation", "model_directory": str(model),
              "revision": plan["revision"], "tensor_count": tensors, "parameter_count": elements,
              "tensor_bytes": data_bytes, "dtype_parameters": dtypes, "shards": headers,
              "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
              "config_class": type(config).__name__, "processor_class": type(processor).__name__,
              "image_processor_settings": engine.processor.image_processor.to_dict(), "inputs": prepared,
              "input_contract": "Engineering capacity probe only: exact materialized native first-round A/B requests, original shared card, 65536 pixels/image, no truncation. Five-view feedback remains unvalidated.",
              "formal_round_count": 0, "cuda_initialized": torch.cuda.is_initialized()}
    assert result["cuda_initialized"] is False
    (ROOT / "MODEL_CPU_VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "image_processor_settings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
