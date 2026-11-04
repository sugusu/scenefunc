"""Download only the explicitly approved fixed manifest, with durable receipts."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "MODEL_DOWNLOAD_PLAN.json"
RECEIPT = ROOT / "MODEL_DOWNLOAD_RECEIPT.json"
MUTEX = threading.RLock()
STATE = {}
DEADLINE = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, obj):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def persist():
    with MUTEX:
        STATE["updated_at"] = now()
        STATE["present_bytes"] = sum(r.get("present_bytes", 0) for r in STATE["files"].values())
        atomic(RECEIPT, STATE)
        status_path = ROOT / "STATUS.json"
        status = json.loads(status_path.read_text())
        status.update(updated_at=now(), stage="preflight", milestone="fixed_model_download_" + STATE["status"],
                      process=STATE.get("process") if STATE["status"] == "running" else None,
                      download_receipt=str(RECEIPT),
                      command=[sys.executable, str(Path(__file__).resolve())],
                      next_step="Finish fixed model integrity checks and local processor/model preflight; executor/assets remain pending")
        status["locks"]["model_download"] = {"path": str(ROOT / "model_download.lock"), "held": STATE["status"] == "running"}
        log = str(ROOT / "model_download.log")
        if log not in status["logs"]:
            status["logs"].append(log)
        atomic(status_path, status)


def update(name, **values):
    with MUTEX:
        STATE["files"][name].update(values)
        persist()


def digest(path, item):
    if path.stat().st_size != item["bytes"]:
        raise ValueError("size mismatch for " + item["path"])
    sha256 = hashlib.sha256()
    gitsha = hashlib.sha1(b"blob " + str(item["bytes"]).encode() + b"\0")
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            gitsha.update(block)
    if item["sha256"] and sha256.hexdigest() != item["sha256"]:
        raise ValueError("SHA256 mismatch for " + item["path"])
    if item["git_blob_sha1"] and gitsha.hexdigest() != item["git_blob_sha1"]:
        raise ValueError("Git blob SHA1 mismatch for " + item["path"])
    return {"actual_sha256": sha256.hexdigest(), "actual_git_blob_sha1": gitsha.hexdigest()}


def transfer(item, destination):
    name = item["path"]
    target = destination / name
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        hashes = digest(target, item)
        update(name, status="verified", present_bytes=item["bytes"], source="existing_verified", **hashes)
        return
    reference = ROOT / "preflight/model_reference" / name
    if reference.is_file():
        hashes = digest(reference, item)
        shutil.copyfile(reference, partial)
        digest(partial, item)
        os.replace(partial, target)
        update(name, status="verified", present_bytes=item["bytes"], source="verified_preflight_metadata", **hashes)
        return
    for attempt in range(1, 9):
        if time.monotonic() > DEADLINE:
            raise TimeoutError("Overall download deadline exceeded")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item["bytes"]:
            raise ValueError("Oversized existing partial; preserve and inspect " + name)
        update(name, status="downloading", attempt=attempt, present_bytes=offset)
        try:
            if offset < item["bytes"]:
                headers = {"User-Agent": "P920-DD-v1-fixed-manifest/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(item["url"] + "?download=true", headers=headers)
                with urllib.request.urlopen(request, timeout=45) as response:
                    code = response.status
                    if offset:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if code != 206 or not match or int(match[1]) != offset or int(match[3]) != item["bytes"]:
                            raise ValueError("Server did not honor exact resume range; partial preserved")
                    elif code != 200:
                        raise ValueError("Unexpected initial HTTP status " + str(code))
                    last = time.monotonic()
                    with partial.open("ab") as f:
                        while True:
                            if time.monotonic() > DEADLINE:
                                raise TimeoutError("Overall download deadline exceeded")
                            data = response.read(1024 * 1024)
                            if not data:
                                break
                            if offset + len(data) > item["bytes"]:
                                raise ValueError("Response exceeds authorized file size")
                            f.write(data)
                            offset += len(data)
                            if time.monotonic() - last >= 10:
                                f.flush()
                                update(name, present_bytes=offset)
                                last = time.monotonic()
                        f.flush(); os.fsync(f.fileno())
                if offset < item["bytes"]:
                    raise ConnectionError("Premature EOF; resume verified byte range")
            update(name, status="verifying", present_bytes=offset)
            hashes = digest(partial, item)
            os.replace(partial, target)
            update(name, status="verified", present_bytes=item["bytes"], source="920_local_https", **hashes)
            print(now(), "verified", name, item["bytes"], flush=True)
            return
        except Exception as error:
            safe_error = type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                safe_error += " HTTP " + str(error.code)
            elif isinstance(error, ValueError):
                safe_error += ": " + str(error)
            fatal = isinstance(error, ValueError) or attempt == 8
            update(name, status="failed" if fatal else "retry_pending", error=safe_error,
                   present_bytes=partial.stat().st_size if partial.exists() else 0)
            print(now(), name, safe_error, "attempt", attempt, flush=True)
            if fatal:
                raise RuntimeError(name + ": " + safe_error) from None
            time.sleep(2 * attempt)


def main():
    global STATE, DEADLINE
    with (ROOT / "model_download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads(PLAN_PATH.read_text())
        assert plan["state"] == "authorized", "Explicit approval not recorded"
        assert plan["revision"] == "63e235a0c0d6e5bc2f8f3485214df4457fd2e672"
        assert len(plan["files"]) == 12 and sum(x["bytes"] for x in plan["files"]) == 17541458531
        destination = Path(plan["destination"]).resolve()
        assert ROOT in destination.parents
        for item in plan["files"]:
            assert Path(item["path"]).name == item["path"]
            assert item["url"] == "https://huggingface.co/" + plan["model"] + "/resolve/" + plan["revision"] + "/" + item["path"]
        destination.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(destination).free > plan["total_bytes"] + 2 * 1024**3
        if RECEIPT.exists():
            previous = json.loads(RECEIPT.read_text())
            archive = ROOT / ("MODEL_DOWNLOAD_RECEIPT_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
            atomic(archive, previous)
        STATE = {"handoff_id": "P920-DD-v1", "status": "running", "started_at": now(),
                 "process": {"pid": os.getpid(), "start_ticks": Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19], "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()},
                 "plan_sha256": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest(),
                 "destination": str(destination), "revision": plan["revision"], "total_bytes": plan["total_bytes"],
                 "files": {x["path"]: {"status": "pending", "expected_bytes": x["bytes"], "present_bytes": 0} for x in plan["files"]},
                 "maximum_concurrent_transfers": 4, "wall_timeout_seconds": 14400, "model_or_gpu_execution": False}
        DEADLINE = time.monotonic() + 14400
        persist(); print(now(), "started fixed-manifest download", flush=True)
        errors = []
        small = [x for x in plan["files"] if x["bytes"] < 10_000_000]
        large = [x for x in plan["files"] if x["bytes"] >= 10_000_000]
        for item in small:
            try: transfer(item, destination)
            except Exception as error: errors.append(str(error))
        if not errors:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(transfer, item, destination) for item in large]
                for future in concurrent.futures.as_completed(futures):
                    try: future.result()
                    except Exception as error: errors.append(str(error))
        STATE.update(status="verified" if not errors else "failed", ended_at=now(), errors=errors)
        persist()
        print(now(), STATE["status"], json.dumps(errors), flush=True)
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
