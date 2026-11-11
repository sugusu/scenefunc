"""Fetch pinned public research source/data to RAM; no model or backend runs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

CODE_PIN = "778be78bdac582b51a975ff819046583aad383e0"
DATA_PIN = "257d3d5a9a056fd357ac4a92c6edbc20dc0cb3b1"
CODE = ["LICENSE", "README.md", "RESULTS.md", "docs/implementation.md",
        "psd/training/river_psd.py", "psd/training/turn_kl.py", "psd/targets/river_topk.py",
        "psd/repair/hints.py", "psd/repair/turn_recovery.py",
        "experiments/bfcl/agentic/attempt_runner.py", "experiments/bfcl/agentic/build_river_native_rows.py",
        "experiments/bfcl/agentic/leak_audit.py", "experiments/bfcl/repair/slate.py",
        "experiments/bfcl/repair/harvest.py", "scripts/build_repair_browser_traces.py"]
DATA = ["README.md", "MANIFEST.sha256", "traces/river_native/trace_manifest.json",
        "rollouts/river_native/summary.json", "repairs/river_native/repair_rows_clean.jsonl",
        "traces/river_native/repair_search_traces.jsonl", "targets/river_native/target_manifest.json",
        "targets/river_native/top20_targets.jsonl"]


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--output", type=Path, required=True)
    out = a.parse_args().output.resolve()
    assert out.is_relative_to(Path("/dev/shm"))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("psd_tree.json", f"https://api.github.com/repos/essamsleiman/psd/git/trees/{CODE_PIN}?recursive=1"),
            ("data_repo.json", f"https://huggingface.co/api/datasets/essamsleiman/psd-bfcl/revision/{DATA_PIN}")]
    jobs += [("repo/"+n, f"https://raw.githubusercontent.com/essamsleiman/psd/{CODE_PIN}/{n}") for n in CODE]
    jobs += [("data/"+n, f"https://huggingface.co/datasets/essamsleiman/psd-bfcl/resolve/{DATA_PIN}/{n}") for n in DATA]

    def fetch(job):
        name, url = job
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name+".partial")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "public-research-source-audit"})
            with urllib.request.urlopen(req, timeout=90) as r, tmp.open("wb") as f:
                while block := r.read(1024*1024):
                    f.write(block)
            tmp.replace(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return dict(path=name, url=url, bytes=path.stat().st_size, sha256=digest)
        except Exception as e:
            return dict(path=name, url=url, error=type(e).__name__+": "+str(e))

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, jobs))
    (out / "download_manifest.json").write_text(json.dumps(dict(code_pin=CODE_PIN, data_pin=DATA_PIN, files=rows), indent=2)+"\n")
    failures = [r for r in rows if "error" in r]
    print(json.dumps(dict(files=len(rows), bytes=sum(r.get("bytes", 0) for r in rows), failures=failures)))
    if failures:
        raise RuntimeError("Some source fetches failed; inspect manifest, do not infer source correctness")


if __name__ == "__main__":
    main()
