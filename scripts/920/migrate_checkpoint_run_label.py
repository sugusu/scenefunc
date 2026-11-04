"""Explicitly migrate the verified run-label hashing defect; never replay tools."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil

from native_checkpoint import canonical_hash, config_hash, runtime_fingerprint
from native_project import read_json, sha256, validate_bundle


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--configuration", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--previous-helper-sha", required=True)
    p.add_argument("--designer", type=Path, required=True)
    p.add_argument("--scenesmith", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.destination.exists() or args.out.exists():
        raise ValueError("Migration outputs must be new")
    manifest, state, _, _ = validate_bundle(args.source)
    old_record = read_json(args.source / manifest["inputs"]["session"])
    expected = read_json(args.expected)
    source_hash = sha256(args.source / "manifest.json")
    if expected["manifest_sha256"] != source_hash or expected["house_state"] != state:
        raise ValueError("Expected evidence does not match source checkpoint")
    current = runtime_fingerprint(args.designer, args.scenesmith)
    previous = old_record["runtime_fingerprint"]
    differences = sorted(key for key in set(current) | set(previous) if current.get(key) != previous.get(key))
    if differences != ["adapter/native_checkpoint.py"] or previous[differences[0]] != args.previous_helper_sha:
        raise ValueError("Migration only permits the explicitly identified checkpoint helper change")
    configuration = read_json(args.configuration)["effective_config"]
    old_directory = Path(configuration["experiment"]["output_dir"])
    if not re.fullmatch(r"chained_replay_[0-9]{6}", str(configuration.get("name", ""))):
        raise ValueError("No automatically generated label to migrate")
    normalized = json.loads(json.dumps(configuration).replace(str(old_directory), "{PROJECT_DIRECTORY}"))
    if canonical_hash(normalized) != old_record["effective_config_sha256"]:
        raise ValueError("Configuration evidence differs from original saved hash")
    new_record = copy.deepcopy(old_record)
    new_record["effective_config_sha256"] = config_hash(configuration, old_directory)
    new_record["runtime_fingerprint"] = current
    # State, memory, all real receipts and counters remain unchanged.
    for key in old_record:
        if key not in {"effective_config_sha256", "runtime_fingerprint"}:
            assert new_record[key] == old_record[key]
    shutil.copytree(args.source, args.destination)
    relative = manifest["inputs"]["session"]
    record_path = args.destination / relative
    write(record_path, new_record)
    new_manifest = copy.deepcopy(manifest)
    new_manifest["files"][relative] = {"bytes": record_path.stat().st_size, "sha256": sha256(record_path)}
    migration = {"source_manifest_sha256": source_hash,
                 "reason": "Ignore only the upstream automatically generated logging label in the config hash",
                 "changed_session_fields": ["effective_config_sha256", "runtime_fingerprint"],
                 "old_config_sha256": old_record["effective_config_sha256"],
                 "new_config_sha256": new_record["effective_config_sha256"],
                 "old_checkpoint_helper_sha256": args.previous_helper_sha,
                 "new_checkpoint_helper_sha256": current["adapter/native_checkpoint.py"],
                 "tools_replayed": 0}
    new_manifest["checkpoint_migration"] = migration
    write(args.destination / "manifest.json", new_manifest)
    validate_bundle(args.destination)
    for file in manifest["files"]:
        if file != relative:
            assert sha256(args.destination / file) == sha256(args.source / file)
    assert sha256(args.source / "manifest.json") == source_hash
    args.out.mkdir()
    expected["migration_source_manifest_sha256"] = source_hash
    expected["manifest_sha256"] = sha256(args.destination / "manifest.json")
    write(args.out / "EXPECTED.json", expected)
    write(args.out / "MIGRATION.json", migration)
    print(json.dumps({"status": "completed", "tools_replayed": 0, "source_unchanged": True}))


if __name__ == "__main__":
    main()
