"""Engineering CUDA check, not a policy forward or an experiment result.

Retains gpu_guard.py's authorized UUID, project-lock, no-compute-process,
and <=512 MiB idle-usage gates. Uses the working system driver instead of
the legacy project's 580.173.02 private libraries. Does not overwrite old receipts.
"""
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
UUID = "GPU-6df2660c-5a77-4e41-0b1a-c6208368fcd1"
PYTHON = "/data/wyh/conda/envs/c15_genrec/bin/python"


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt_path = ROOT / "CUDA_PREFLIGHT.json"
    if receipt_path.exists():
        raise SystemExit("Existing CUDA receipt: inspect before retrying")
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    receipt = {
        "kind": "engineering_cuda_only", "started_at": started,
        "pid": os.getpid(), "pid_start_ticks": Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "authorized_uuid": UUID, "physical_gpu": 0,
        "legacy_guard_sha256": hashlib.sha256((PARENT / "gpu_guard.py").read_bytes()).hexdigest(),
        "driver_selection": "system libraries; legacy private driver not injected",
        "model_forward_performed": False, "formal_capacity_validated": False,
        "status": "preflight", "log": str(ROOT / "cuda_preflight.log"),
    }
    atomic(receipt_path, receipt)
    try:
        with (PARENT / "work/gpu0.lock").open("r") as legacy, (ROOT / "gpu0.lock").open("a") as local:
            fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(local, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def query(*args):
                return subprocess.check_output(["nvidia-smi", "-i", "0", *args], env=env, text=True, timeout=15).strip()
            row = next(csv.reader([query("--query-gpu=index,uuid,memory.used,memory.free,memory.total,driver_version", "--format=csv,noheader,nounits")]))
            index, actual_uuid, used, free, total, driver = [x.strip() for x in row]
            assert index == "0" and actual_uuid == UUID, "GPU authorization mismatch"
            apps = query("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader")
            assert not apps and int(used) <= 512, "GPU occupied; no process stopped"
            assert int(free) >= 1024, "Insufficient memory for tiny engineering probe"
            receipt["gpu_before"] = dict(index=index, uuid=actual_uuid, used_mib=int(used), free_mib=int(free), total_mib=int(total), driver=driver)
            env.update(CUDA_VISIBLE_DEVICES=UUID, CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            code = """
import json, sys, torch
from torch.nn import functional as F
assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
x=torch.ones((64,64),device='cuda',dtype=torch.float16)
y=x@x
assert bool(torch.all(y==64).item())
q=torch.ones((1,2,16,32),device='cuda',dtype=torch.float16)
z=F.scaled_dot_product_attention(q,q,q,is_causal=True)
assert bool(torch.isfinite(z).all().item())
torch.cuda.synchronize()
print(json.dumps({'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'fp16_matmul':'pass','sdpa_tiny_tensor':'pass','max_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'max_memory_reserved_bytes':torch.cuda.max_memory_reserved(),'model_forward':False,'capacity_for_8b': 'not_tested'},indent=2))
"""
            command = [PYTHON, "-c", code]
            receipt.update(status="running", command=command)
            atomic(receipt_path, receipt)
            with (ROOT / "cuda_preflight.log").open("x") as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt["child_pid"] = process.pid
                receipt["child_start_ticks"] = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()[19]
                atomic(receipt_path, receipt)
                try:
                    result = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise RuntimeError("Own engineering probe exceeded 120-second timeout")
            receipt.update(exit_code=result, status="pass" if result == 0 else "failed")
            if result != 0:
                raise RuntimeError("CUDA probe failed; inspect log")
    except Exception as error:
        receipt.update(status="failed", error=str(error))
    receipt["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
