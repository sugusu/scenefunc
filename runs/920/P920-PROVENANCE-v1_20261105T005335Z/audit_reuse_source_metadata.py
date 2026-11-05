"""Inspect a captured public article's JSON metadata without executing its scripts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--html', type=Path, required=True)
    ap.add_argument('--fetch-receipt', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    raw = args.html.read_bytes()
    receipt = json.loads(args.fetch_receipt.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    assert receipt['status'] == 'completed' and receipt['sha256'] == digest
    match = re.search(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', raw.decode(), re.S)
    assert match is not None, 'Captured article has no expected JSON metadata'
    values = json.loads(match.group(1))

    def resolve(index, depth=0):
        if not isinstance(index, int) or index < 0 or index >= len(values) or depth > 20:
            raise ValueError('Invalid or excessive metadata reference')
        value = values[index]
        if isinstance(value, dict):
            return {key: resolve(ref, depth+1) for key, ref in value.items()}
        if isinstance(value, list):
            return [resolve(ref, depth+1) for ref in value]
        return value

    stores = [x for x in values if isinstance(x, dict) and 'currentArticle' in x and 'hasSupplementalData' in x]
    assert len(stores) == 1
    article = values[stores[0]['currentArticle']]
    assert resolve(article['doi']) == '10.3389/fbuil.2025.1727219'
    files = resolve(article['files'])
    supplements = [x for x in files if x['type']['code'] == 'SUPPLEMENTARY_MATERIAL']
    assert len(supplements) == 4
    xml_refs = [resolve(x[':@']) for x in values if isinstance(x, dict) and 'supplementary-material' in x and ':@' in x]
    assert len(xml_refs) == 4
    report = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'article_sha256': digest, 'article_bytes': len(raw),
        'doi': resolve(article['doi']),
        'current_article_files': files,
        'xml_supplement_references': xml_refs,
        'server_rendered_has_supplemental_data': resolve(stores[0]['hasSupplementalData']),
        'supplement_content_downloaded': False,
        'limitations': 'File names/versions and relative references only. A false UI state is not evidence that files do not exist. SM3 refers to file4 and SM4 to file3; content numbering cannot be verified without PDFs. No script execution or independent professional review.',
    }
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'metadata_audit_passed', 'supplement_records': len(supplements),
                      'downloaded_supplements': 0, 'model_calls': 0, 'training_updates': 0}))


if __name__ == '__main__':
    main()
