"""One bounded engineering generation using the approved model and actual input.

This uses the exact native first-round input with a 32-token engineering cap.
Never use this truncated output as an A/B scene result or execute its tools.
"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get('P920_RUN_ROOT', Path(__file__).resolve().parent))
UUID = "GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1"
PYTHON = "/data/wyh/conda/envs/c15_genrec/bin/python"
ATTEMPT = os.environ.get('P920_FORWARD_ATTEMPT', 'initial')
assert ATTEMPT in ('initial', 'utf8'), 'Only the documented engineering retry is allowed'
SUFFIX = '' if ATTEMPT == 'initial' else '_UTF8'
RESULT = ROOT / ('MODEL_FORWARD' + SUFFIX + '_PREFLIGHT.json')
LAUNCH = ROOT / ('MODEL_FORWARD' + SUFFIX + '_LAUNCH.json')
LOG = ROOT / ('model_forward_preflight' + SUFFIX.lower() + '.log')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
    os.replace(temp, path)


def child():
    import resource
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.94, 0)
    plan = json.loads((ROOT / "MODEL_DOWNLOAD_PLAN.json").read_text())
    model_dir = plan["destination"]
    result = {"kind": "engineering_model_generation_only", "status": "running", "started_at": now(),
              "model": model_dir, "revision": plan["revision"], "precision": "float16",
              "attention": "sdpa", "batch_size": 1, "seed": 0, "max_new_tokens": 32,
              "do_sample": False, "use_cache": True, "cuda_allocator_fraction_cap": 0.94,
              "formal_round": False, "tool_execution": False, "full_2048_token_budget_validated": False,
              "five_view_feedback_budget_validated": False, "process_id": os.getpid(), "attempt": ATTEMPT}
    target = RESULT
    start = time.monotonic()
    try:
        input_path = ROOT / "preflight/model_local/A_engineering_inputs.pt"
        result["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
        inputs = torch.load(input_path, weights_only=True, map_location="cpu")
        result["input_tokens"] = int(inputs["input_ids"].shape[1])
        atomic(target, result)
        print(now(), "loading fixed FP16 model", flush=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map={"": 0},
            attn_implementation="sdpa", local_files_only=True, low_cpu_mem_usage=True).eval()
        torch.cuda.synchronize()
        result["load_seconds"] = time.monotonic() - start
        result["loaded_parameter_dtype"] = str(next(model.parameters()).dtype)
        result["device"] = str(next(model.parameters()).device)
        result["allocated_after_load_bytes"] = torch.cuda.memory_allocated()
        assert next(model.parameters()).dtype == torch.float16
        atomic(target, result)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        before_generation = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        new_ids = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
        atomic(ROOT / 'preflight/model_local' / ('A_engineering_token_ids' + SUFFIX + '.json'), new_ids)
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        text = processor.decode(new_ids, skip_special_tokens=False)
        raw = ROOT / 'preflight/model_local' / ('A_engineering_raw_output' + SUFFIX + '.txt')
        raw.write_text(text, encoding='utf-8')
        result.update(status="pass", generation_seconds=time.monotonic() - before_generation,
                      output_tokens=len(new_ids), output_token_ids=new_ids, raw_output=str(raw),
                      output_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result.update(ended_at=now(), total_seconds=time.monotonic() - start,
                      max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                      max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                      process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(target, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "pass" else 1


def guard():
    receipt_path = LAUNCH
    assert not receipt_path.exists(), "Prior attempt exists: inspect it before authorizing a retry"
    assert json.loads((ROOT / "MODEL_CPU_VALIDATION.json").read_text())["status"] == "pass"
    plan = json.loads((ROOT / "MODEL_DOWNLOAD_PLAN.json").read_text())
    inputs = json.loads((ROOT / "MODEL_CPU_VALIDATION.json").read_text())["inputs"][0]
    tensor_bytes = 17534247392
    kv_bound = (inputs["input_tokens"] + 32) * 147456
    # Explicit engineering estimate, not a reduction of any full-round guard.
    required_free = tensor_bytes + kv_bound + 4 * 1024**3
    mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemAvailable:", "MemTotal:"))}
    required_host = tensor_bytes * 2 + 8 * 1024**3
    assert mem["MemAvailable"] >= required_host, "Insufficient host-memory estimate for model load"
    with (ROOT.parent / "work/gpu0.lock").open("r") as legacy, (ROOT / "gpu0.lock").open("a") as local:
        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = os.environ.copy(); env.pop("LD_LIBRARY_PATH", None)
        def query(*args):
            return subprocess.check_output(["nvidia-smi", "-i", "0", *args], env=env, text=True, timeout=15).strip()
        row = [v.strip() for v in next(csv.reader([query("--query-gpu=index,uuid,memory.used,memory.free,memory.total,driver_version", "--format=csv,noheader,nounits")]))]
        index, uuid, used, free, total, driver = row
        assert index == "0" and uuid == UUID
        assert not query("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader") and int(used) <= 512, "Authorized GPU occupied"
        assert int(free) * 1024**2 >= required_free, "Insufficient engineering memory headroom"
        env.update(CUDA_VISIBLE_DEVICES=UUID, CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", P920_RUN_ROOT=str(ROOT), PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        command = [PYTHON, str(Path(__file__).resolve()), "--child"]
        receipt = {"started_at": now(), "status": "running", "command": command, "pid": os.getpid(),
                   "start_ticks": Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19],
                   "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   "uuid": uuid, "gpu": 0, "total_mib": int(total), "free_mib": int(free), "driver": driver,
                   "estimated_required_free_bytes": required_free, "estimate_workspace_reserve_bytes": 4 * 1024**3,
                   "required_host_available_bytes": required_host, "host_memory": mem,
                   "wall_timeout_seconds": 1200, "log": str(LOG), "attempt": ATTEMPT,
                   "formal_capacity_gate": "not evaluated; this gate covers engineering input and 32 output tokens only"}
        atomic(receipt_path, receipt)
        status = json.loads((ROOT / "STATUS.json").read_text())
        status.update(stage="preflight", milestone="model_forward_engineering_running", updated_at=now(), process={"pid": os.getpid(), "start_ticks": receipt["start_ticks"]}, command=command)
        status["logs"].append(receipt["log"])
        atomic(ROOT / "STATUS.json", status)
        with LOG.open("x", encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt["child_pid"] = process.pid
            receipt["child_start_ticks"] = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()[19]
            atomic(receipt_path, receipt)
            try:
                exit_code = process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                exit_code = 124
        receipt.update(ended_at=now(), exit_code=exit_code, status="pass" if exit_code == 0 else "failed")
        atomic(receipt_path, receipt)
        print(json.dumps(receipt, indent=2))
        return exit_code


if __name__ == "__main__":
    raise SystemExit(child() if "--child" in sys.argv else guard())
