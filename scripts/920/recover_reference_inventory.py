"""Inventory the official Function2Scene viewer's embedded reference JSON.

No JavaScript execution, remote media, model calls or training. Full source data
stay outside Git. Generated layouts are not designer-approved action evidence.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_embedded(html):
    matches = re.findall(r"window\.EMBEDDED_DATA\s*=\s*JSON\.parse\(('(?:\\.|[^'\\])*')\);", html)
    if len(matches) != 1:
        raise ValueError("Expected exactly one literal embedded-data assignment")
    # Literal parsing only: never evaluate the surrounding downloaded script.
    value = ast.literal_eval(matches[0])
    if not isinstance(value, str):
        raise ValueError("Embedded payload must be a string literal")
    data = json.loads(value)
    if not isinstance(data, dict) or not isinstance(data.get("folders"), dict):
        raise ValueError("Unexpected embedded payload shape")
    return data


def source_urls(text):
    urls = []
    for raw in re.findall(r"https?://[^\s<>\"']+", text):
        parsed = urlsplit(raw.rstrip(").,;]"))
        url = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
        if url not in urls:
            urls.append(url)
    return urls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    report = {"started_at": now(), "status": "running", "model_calls": 0,
              "training": False, "qualified_professional_decisions": 0}
    print(now(), "Recover official public reference inventory; no JavaScript/model execution", flush=True)
    try:
        source = a.html.read_bytes()
        receipt = json.loads(a.receipt.read_text())
        if digest_bytes(source) != receipt["sha256"] or len(source) != receipt["bytes"]:
            raise ValueError("Downloaded HTML does not match its retrieval receipt")
        data = parse_embedded(source.decode("utf-8"))
        rows, article_groups, normalized_groups = [], defaultdict(list), defaultdict(list)
        for folder, entry in data["folders"].items():
            files = entry.get("files")
            if not isinstance(files, dict):
                raise ValueError("Malformed files for " + folder)
            states = []
            for filename, scene in files.items():
                if not isinstance(scene, dict):
                    raise ValueError("Scene is not a JSON object")
                furniture = scene.get("furniture", [])
                if not isinstance(furniture, list):
                    raise ValueError("Invalid furniture list")
                states.append({"filename": filename,
                    "canonical_sha256": digest_bytes(canonical(scene).encode()),
                    "furniture_count": len(furniture),
                    "native_object_ids_present": all("object_id" in x for x in furniture) if furniture else None,
                    "native_asset_ids_present": all("asset_id" in x for x in furniture) if furniture else None})
            ad = entry.get("ad", "")
            if not isinstance(ad, str):
                raise ValueError("Unexpected functional-brief type")
            urls = source_urls(ad)
            for url in urls:
                article_groups[url].append(folder)
            normalized_groups[folder.replace("-", "_")].append(folder)
            final = next((x for x in states if x["filename"] == "final.json"), None)
            rows.append({"reference_id": folder, "source_article_urls": urls,
                "allocation": "quarantine_unallocated", "training_eligible": False,
                "evidence_role": "generated reference layout and adapted requirements; not professional action supervision",
                "has_adapted_brief": bool(ad.strip()),
                "adapted_brief_sha256": digest_bytes(ad.encode()) if ad else None,
                "parsed_description_sha256": digest_bytes(canonical(entry.get("scene_description")).encode()),
                "constraints_sha256": digest_bytes(canonical(entry.get("constraints")).encode()),
                "files": states, "final_furniture_count": final["furniture_count"] if final else None,
                "professional_object_identity": "not_verified",
                "native_tool_receipts": "absent_from_embedded_schema",
                "research_use_permission": "unverified", "designer_review": "unverified"})
        finals = [r for r in rows if r["final_furniture_count"] is not None]
        matched = [r for r in finals if r["has_adapted_brief"]]
        inventory = {"source": receipt, "source_file": str(a.html.resolve()),
            "source_format": "window.EMBEDDED_DATA literal JSON; no page scripts executed",
            "counts": {"folders": len(rows), "scene_snapshots": sum(len(r["files"]) for r in rows),
                "final_layouts": len(finals), "final_furniture": sum(r["final_furniture_count"] for r in finals),
                "brief_and_final_layouts": len(matched),
                "brief_and_final_furniture": sum(r["final_furniture_count"] for r in matched),
                "distinct_article_urls": len(article_groups)},
            "historical_count_match": {"rooms": len(matched) == 29,
                "furniture": sum(r["final_furniture_count"] for r in matched) == 545,
                "interpretation": "Count correspondence only; exact identity with the absent Windows package is unverified."},
            "shared_article_groups": [{"url": k, "references": v} for k, v in article_groups.items() if len(v) > 1],
            "possible_name_aliases": [v for v in normalized_groups.values() if len(v) > 1],
            "alias_policy": "Do not merge automatically. Review original source correspondence before family allocation.",
            "references": rows,
            "limits": ["Recovery does not recreate the 7 knowledge cards, 165 reviewed images or original cabinet experiment.",
                "Generated box layouts have names and dimensions, not verified real-object identities or Scene-Designer action receipts.",
                "Adapted briefs and parsed descriptions are not verified original occupant submissions or surveyed dimensions.",
                "Public access does not establish training/redistribution permission; raw text/scene data retained only on server.",
                "Do not allocate same-article/alias variants across splits or silently treat references as sealed tests."]}
        (a.out / "REFERENCE_INVENTORY.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
        # Preserve the recovered payload locally for reproducible source review.
        (a.out / "embedded_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        report.update(status="recovered_reference_data_only", exit_code=0,
            source_sha256=receipt["sha256"], inventory_sha256=digest_bytes((a.out / "REFERENCE_INVENTORY.json").read_bytes()),
            counts=inventory["counts"], historical_count_match=inventory["historical_count_match"],
            possible_name_aliases=inventory["possible_name_aliases"])
        print(json.dumps(report["counts"]), flush=True)
    except Exception as error:
        report.update(status="failed", exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["ended_at"] = now()
        (a.out / "RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(now(), report["status"], flush=True)


if __name__ == "__main__":
    main()
