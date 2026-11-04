"""Recompute bounded local source inventory; never infer professional provenance from a folder name."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    root, out = a.root.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    shell = root / "inputs_v1/同项目原柜实验/冻结开发输入/19_remote_studio/initial.json"
    released = root / "scenefunc/paris/config/brief/initial.json"
    data = json.loads(shell.read_text())
    inputs = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p)}
              for p in sorted((root / "inputs_v1").rglob("*")) if p.is_file()]
    roots = [root, Path("/data/wyh/external"), Path("/data/wyh/myproject")]
    patterns = ["专业设计", "知识卡", "同项目", "19_remote", "nursery", "数据与真实", "训练说明"]
    command = ["rg", "--files", *map(str, roots)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    names = result.stdout.splitlines()
    matches = [p for p in names if any(term in p for term in patterns)]
    audit = {"checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Known project inputs plus bounded filename search; no full-disk absence claim",
        "search_roots": list(map(str, roots)), "filename_patterns": patterns,
        "search_command": command, "enumerated_file_count": len(names), "matching_paths": matches,
        "inputs": inputs, "shell_furniture_count": len(data.get("furniture", [])),
        "shell_sha256": sha(shell), "released_paris_shell_sha256": sha(released),
        "same_bytes_as_paris_shell": shell.read_bytes() == released.read_bytes(),
        "finding": "The similarly named cabinet directory contains the Paris empty shell, not a verified nursery initial state.",
        "claimed_external_inventory": {"knowledge_cards": 7, "residential_images": 165,
            "development_rooms": 29, "furniture_items": 545, "training_2d_reviews": 99,
            "verification": "Historical user-provided description; package not located by this search."},
        "external_original_cabinet_replay": "Windows result.json has no verified local/repository path",
        "qualified_replayable_professional_decisions_verified": 0,
        "limitation": "Filename search is not full content search; data may exist elsewhere."}
    assert audit["same_bytes_as_paris_shell"] and audit["shell_furniture_count"] == 0
    (out / "LOCAL_SOURCE_INVENTORY.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"shell_is_paris_empty_shell": True, "source_inputs": len(inputs),
        "filename_matches": len(matches), "qualified_training_decisions": 0}), flush=True)


if __name__ == "__main__":
    main()
