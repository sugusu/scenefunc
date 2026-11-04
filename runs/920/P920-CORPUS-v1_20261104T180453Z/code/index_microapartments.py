"""Index licensed source evidence without creating training episodes or geometry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit, unquote

import pymupdf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def norm(value):
    value = unicodedata.normalize('NFKD', value).casefold()
    return ' '.join(re.sub(r'[^\w]+', ' ', ''.join(c for c in value if not unicodedata.combining(c))).split())


def urlkey(value):
    p = urlsplit(unicodedata.normalize('NFKC', value))
    return p.netloc.removeprefix('www.').casefold() + unquote(p.path).rstrip('/').casefold()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); plan = json.loads(a.plan.read_text()); a.out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    result = {'started_at': now(), 'status': 'running', 'plan_sha256': sha(a.plan),
              'model_calls': 0, 'training_updates': 0, 'training_episodes_created': 0}
    save(a.out/'RESULT.json', result)
    print(now(), 'begin source-index extraction; no training or scene generation', flush=True)
    try:
        for key in ['pdf', 'table1', 'recovered_metadata']:
            assert sha(plan[key]['path']) == plan[key]['sha256'], key
        assert sha(__file__) == plan['script_sha256']
        doc = pymupdf.open(plan['pdf']['path']); source = json.loads(Path(plan['table1']['path']).read_text())
        entries = source['entries']; ids = {e['source_case_id'] for e in entries}
        assert len(entries) == len(ids) == 30
        assert all(e['allocation'] == 'quarantine_source_review' and not e['training_eligible'] for e in entries)
        dimensions = {}
        for line in doc[5].get_text(sort=True).splitlines():
            tokens = line.split()
            if not tokens or tokens[0] not in ids:
                continue
            assert tokens[0] not in dimensions
            area, volume, living, ancillary = tokens[1:5]
            dimensions[tokens[0]] = {'source_pdf_page': 6, 'reported_area_m2': float(area),
                'reported_volume_m3': None if volume == 'nd' else float(volume),
                'reported_living_m2': float(living), 'reported_ancillary_m2': float(ancillary),
                'reported_living_percent': float(tokens[5].removesuffix('%')),
                'reported_users': tokens[6], 'raw_table_row': line.strip()}
        assert set(dimensions) == ids
        spans = {}
        for page in range(14, 22):
            text = doc[page].get_text(sort=True)
            matches = list(re.finditer(r'(?m)^\s*(M\.(?:S|XS|XXS)\.\d+)\s+[^\n]+', text))
            for i, m in enumerate(matches):
                cid = m.group(1); assert cid not in spans
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                raw = text[m.start():end]
                refnums = sorted({int(n) for group in re.findall(r'\[([\d,\s]+)\]', raw)
                                  for n in re.findall(r'\d+', group)})
                path = a.out/'source_fragments'/(cid.replace('.', '_') + '.txt')
                path.parent.mkdir(exist_ok=True); path.write_text(raw)
                spans[cid] = {'pdf_page': page+1, 'char_start_in_sorted_page_text': m.start(),
                    'char_end_in_sorted_page_text': end, 'source_references': refnums,
                    'local_fragment': str(path.resolve()), 'sha256': sha(path),
                    'role': 'Published final-design description. NOT raw brief, tool trajectory or SFT target.'}
        assert set(spans) == ids
        bibtext = '\n'.join(doc[i].get_text(sort=True) for i in [28, 29])
        matches = list(re.finditer(r'(?m)^\s*(\d{1,2})\.\s+', bibtext)); bibliography = {}
        for i, m in enumerate(matches):
            number = int(m.group(1)); end = matches[i+1].start() if i+1 < len(matches) else len(bibtext)
            raw = bibtext[m.start():end]
            u = re.search(r'Available online:\s*(.*?)\s*\(accessed', raw, re.S)
            url = re.sub(r'\s+', '', unicodedata.normalize('NFKC', u.group(1))) if u else None
            if url and url.startswith('www.'):
                url = 'https://' + url
            bibliography[number] = {'url_as_indexed': url,
                'kind': 'author_correspondence_not_acquired' if 'E-Mail' in raw else 'published_source',
                'reference_text_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                'normalization': 'PDF line wrapping removed, Unicode ligatures normalized; URL access not implied.'}
        recovered = json.loads(Path(plan['recovered_metadata']['path']).read_text())['folders']
        metadata = {name: json.dumps({k:v for k,v in data.items() if k != 'files'}, ensure_ascii=False)
                    for name,data in recovered.items()}
        normalized = {name:norm(text) for name,text in metadata.items()}
        oldurls = {name:{urlkey(u.rstrip('.,);')) for u in re.findall(r'https?://[^\s<>"\\]+', text)}
                   for name,text in metadata.items()}
        indexed = []
        for e in entries:
            cid = e['source_case_id']; assert sorted(e['bibliography_ids']) == spans[cid]['source_references'], cid
            refs = {str(k):bibliography[k] for k in e['bibliography_ids']}
            urls = {urlkey(v['url_as_indexed']) for v in refs.values() if v['url_as_indexed']}
            # A shared architect is only a review flag; never merge physical projects automatically.
            terms = [e['reported_architect']]
            if len(norm(e['reported_name']).split()) >= 2:
                terms.append(e['reported_name'])
            hints = {name:[t for t in terms if norm(t) in text] for name,text in normalized.items()}
            m = dimensions[cid]; group = cid.split('.')[1]
            bounds = {'S':(25,35),'XS':(15,25),'XXS':(0,15)}[group]
            audit = {'outside_declared_group': not bounds[0] <= m['reported_area_m2'] < bounds[1],
                'living_plus_ancillary_minus_total_m2': round(m['reported_living_m2'] + m['reported_ancillary_m2'] - m['reported_area_m2'], 3),
                'computed_minus_reported_living_percent': round(100*m['reported_living_m2']/m['reported_area_m2'] - m['reported_living_percent'], 3),
                'policy': 'Preserve reported numbers; differences are source audit notes, not automatically corrected or survey precision.'}
            indexed.append({**e, 'dimensions': m, 'source_fragment': spans[cid], 'bibliography': refs,
                'source_number_audit': audit,
                'possible_prior_metadata_relations': {k:v for k,v in hints.items() if v},
                'exact_prior_url_matches': [k for k,v in oldurls.items() if urls & v]})
        correlations = {}
        for e in entries:
            correlations.setdefault(e['reported_architect'], []).append(e['source_case_id'])
        index = {'source_doi': source['source_doi'], 'source_sha256': source['source_sha256'],
            'created_at': now(), 'license': 'Authored article CC BY 4.0; attribution in TABLE1_INDEX.json. Original third-party source rights remain separate.',
            'common_source_correlation_group': 'buildings11120654', 'entries': indexed,
            'shared_architect_groups': {k:v for k,v in correlations.items() if len(v)>1},
            'scope': 'Metadata and exact local source fragments only. 30 entries are not 30 verified independent families or replayable training decisions.'}
        save(a.out/'INDEX.json', index)
        result.update(status='completed_source_index_only', exit_code=0, indexed_entries=len(indexed),
            source_fragments=len(spans), exact_reference_url_matches=sum(bool(e['exact_prior_url_matches']) for e in indexed),
            possible_metadata_relations={e['source_case_id']:e['possible_prior_metadata_relations'] for e in indexed if e['possible_prior_metadata_relations']},
            source_band_inconsistencies=[e['source_case_id'] for e in indexed if e['source_number_audit']['outside_declared_group']],
            new_qualified_training_decisions=0)
    except Exception as e:
        result.update(status='failed', exit_code=1, error_type=type(e).__name__, error=str(e)); raise
    finally:
        result['ended_at'] = now(); save(a.out/'RESULT.json', result)
        print(now(), result['status'], flush=True)


if __name__ == '__main__':
    main()
