"""Prepare isolated pinned runtime source checkouts and the official integration.

No package installation, service, model, asset download or experiment execution.
Existing directories are refused; partial failures remain available for inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORIES = (
    ("Scene-Designer", "https://github.com/ZJU-OmniAI/Scene-Designer.git",
     "a0565e50ed7d32c0e09132b9311c0b9b91165405",
     ["src", "Designer", "integration", "scripts", "prompts", "tools",
      "trajectory_compression", "src_critic"]),
    ("scenesmith", "https://github.com/nepfaff/scenesmith.git",
     "e888297a8b0a2f3acbb8634edf9def59cceab836",
     ["scenesmith", "configurations", "scripts", "materials"]),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--proxy", default="http://127.0.0.1:19998")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"started_at": now(), "status": "running", "commands": [],
               "repositories": [], "models_or_services_started": False,
               "packages_changed": False, "private_ssh_key_configured": False}

    def save():
        (out / "SOURCE_CHECKOUTS.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(command, cwd=None):
        record = {"command": list(map(str, command)), "cwd": str(cwd or out),
                  "started_at": now()}
        receipt["commands"].append(record)
        save()
        print(record["started_at"], " ".join(record["command"]), flush=True)
        start = time.monotonic()
        try:
            result = subprocess.run(record["command"], cwd=cwd or out,
                                    capture_output=True, text=True, timeout=180)
            record.update(exit_code=result.returncode, output=result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError("Command failed: " + " ".join(record["command"]))
            return result.stdout.strip()
        finally:
            record.update(ended_at=now(), elapsed_seconds=time.monotonic() - start)
            save()

    git = ["git", "-c", "http.proxy=" + args.proxy]
    try:
        for name, url, revision, folders in REPOSITORIES:
            target = out / name
            run([*git, "init", "--quiet", str(target)])
            run([*git, "remote", "add", "origin", url], target)
            run([*git, "config", "remote.origin.promisor", "true"], target)
            run([*git, "config", "remote.origin.partialclonefilter", "blob:none"], target)
            run([*git, "fetch", "--depth=1", "--filter=blob:none", "origin", revision], target)
            run([*git, "sparse-checkout", "init", "--cone"], target)
            run([*git, "sparse-checkout", "set", *folders], target)
            run([*git, "checkout", "--detach", revision], target)
            head = run([*git, "rev-parse", "HEAD"], target)
            if head != revision or run([*git, "status", "--porcelain"], target):
                raise ValueError("Pinned checkout is not clean")
            receipt["repositories"].append({"name": name, "url": url, "commit": head,
                "directory": str(target), "runtime_sparse_directories": folders,
                "scope": "Complete selected runtime source directories; demo media, website and test data excluded."})
        designer, smith = out / "Scene-Designer", out / "scenesmith"
        script = designer / "scripts/setup_scenesmith.py"
        patch = designer / "integration/scenesmith.patch"
        receipt["official_setup_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        receipt["official_patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
        run([sys.executable, str(script), "--path", str(smith), "--check"], designer)
        run([sys.executable, str(script), "--path", str(smith)], designer)
        run([sys.executable, str(script), "--path", str(smith), "--check"], designer)
        receipt["scene_designer_clean"] = not bool(run([*git, "status", "--porcelain"], designer))
        receipt["scenesmith_integration_changes"] = run([*git, "status", "--porcelain"], smith)
        receipt.update(status="runtime_sources_integrated", exit_code=0)
    except Exception as error:
        receipt.update(status="failed_preserved_partial_sources", exit_code=1,
                       error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["ended_at"] = now()
        save()
        print(receipt["ended_at"], receipt["status"], flush=True)


if __name__ == "__main__":
    main()
