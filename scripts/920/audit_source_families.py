"""Bounded source-metadata comparison; never inspect recovered scene geometry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote


def canonical(url):
    p = urlsplit(url.rstrip('.,);'))
    return p.netloc.lower().removeprefix('www.') + unquote(p.path).rstrip('/').lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--registry', type=Path, required=True)
    ap.add_argument('--sources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    raw = a.reference.read_bytes()
    folders = json.loads(raw)['folders']
    registry = json.loads(a.registry.read_text())
    candidates = json.loads(a.sources.read_text())['case_candidates']
    candidate_ids = {c['candidate_family_id'] for c in candidates}
    old_projects = [p for p in registry['projects'] if p['family_id'] not in candidate_ids]
    metadata = {name: json.dumps({k: v for k, v in doc.items() if k != 'files'},
                                 ensure_ascii=False) for name, doc in folders.items()}
    urls = {name: sorted({canonical(u) for u in re.findall(r'https?://[^\s<>"\\]+', text)})
            for name, text in metadata.items()}
    aliases = {
        'singapore_bukit_batok_interim_spaces': ['Bukit Batok', 'Interim Spaces', 'Dess Chew', 'Three-D Conceptwerke'],
        'milan_taaac': ['TAAAC', 'ATOMAA'],
        'singapore_newton_design_pair': ['Newton', 'Design Pair', 'Mikolaj Scibisz', 'Joyce Li'],
    }
    results = []
    for c in candidates:
        family = c['candidate_family_id']
        target = canonical(c['primary_project_url'])
        matches = {name: [s for s in aliases[family] if s.casefold() in text.casefold()]
                   for name, text in metadata.items()}
        results.append({'family_id': family, 'canonical_primary_url': target,
                        'exact_reference_url_matches': [name for name, us in urls.items() if target in us],
                        'reference_alias_matches': {k: v for k, v in matches.items() if v},
                        'existing_registry_id_matches': [p['family_id'] for p in old_projects if p['family_id'] == family],
                        'aliases_checked': aliases[family]})
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference': str(a.reference.resolve()), 'reference_sha256': hashlib.sha256(raw).hexdigest(),
              'metadata_folders': len(metadata), 'folders_with_source_urls': sum(bool(u) for u in urls.values()),
              'existing_registered_families': [p['family_id'] for p in old_projects],
              'reference_url_index': urls, 'candidates': results,
              'scope': 'Source URLs and non-file brief metadata only. No recovered geometry/target scene was inspected by this audit.',
              'limitation': 'No exact URL/name hit is not proof of universal physical-project uniqueness. Registry physical identity also needs source-based manual review.',
              'sealed_test_eligible': False}
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'metadata_folders': len(metadata), 'candidates': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
