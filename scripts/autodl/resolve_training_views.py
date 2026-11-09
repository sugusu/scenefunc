"""Locate exact corpus images in a pinned public Git snapshot, never rerender.

The bounded search hashes published PNG/JPEG blobs. Only images matching a
declared corpus SHA256 are materialized. Missing views remain missing.
No model weights, account credentials or remote runtime are accessed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

from audit_material_routing import PIN, CORPUS, digest, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-bytes', type=int, default=100 * 1024 ** 2)
    ap.add_argument('--source-images-only', action='store_true')
    ap.add_argument('--prior', type=Path)
    args = ap.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    corpus_raw = subprocess.check_output(['git', 'show', f'{PIN}:{CORPUS}'], cwd=args.repo)
    corpus = json.loads(corpus_raw)
    wanted = {image['sha256'] for row in corpus['rows'] for image in row['images']}
    source_runs = {image['path'].split('/runtime/', 1)[1].split('/')[0]
                   for row in corpus['rows'] for image in row['images']}
    source_names = {(image['path'].split('/runtime/', 1)[1].split('/')[0], Path(image['path']).name)
                    for row in corpus['rows'] for image in row['images']}
    # No -l: asking for every blob size in a partial clone fetches unrelated data.
    tree = subprocess.check_output(['git', 'ls-tree', '-r', PIN, 'runs/920'], cwd=args.repo, text=True)
    entries = []
    for line in tree.splitlines():
        meta, path = line.split('\t', 1)
        if Path(path).suffix.lower() in ('.png', '.jpg', '.jpeg'):
            entries.append((meta.split()[2], path))
    if args.source_images_only:
        entries = [item for item in entries if item[1].split('/')[2] in source_runs]
    entries.sort(key=lambda item: ((item[1].split('/')[2], Path(item[1]).name) not in source_names,
                                  item[1].split('/')[2] not in source_runs, item[1]))
    seen, matches, scanned = {}, {}, []
    if args.prior:
        prior = json.loads(args.prior.read_text())
        if prior['pin'] != PIN or prior['corpus_sha256'] != digest(corpus_raw):
            raise ValueError('Prior lookup uses different sources')
        scanned = prior['scanned'][:]
        seen = {r['git_blob']: r['sha256'] for r in scanned}
        for row in prior['rows']:
            for ref in row['images']:
                if ref['match']:
                    match = ref['match']
                    if digest(Path(match['local_path']).read_bytes()) != ref['sha256']:
                        raise ValueError('Prior cache integrity failure')
                    matches[ref['sha256']] = match
    total_bytes = 0
    stopped = None
    for blob, path in entries:
        if blob in seen:
            value = seen[blob]
            if value in matches and path not in matches[value]['git_paths']:
                matches[value]['git_paths'].append(path)
            continue
        if total_bytes >= args.max_bytes:
            stopped = 'Read budget reached; unsurveyed images are not declared absent.'
            break
        # One published image at a time. Bound single-file reads as well.
        proc = subprocess.Popen(['git', 'show', f'{PIN}:{path}'], cwd=args.repo,
                                stdout=subprocess.PIPE)
        data = proc.stdout.read(min(8 * 1024 ** 2, args.max_bytes - total_bytes) + 1)
        if len(data) > min(8 * 1024 ** 2, args.max_bytes - total_bytes):
            proc.terminate()
            proc.wait()
            stopped = 'One image exceeded the remaining or per-image read budget.'
            break
        if proc.wait() != 0:
            raise RuntimeError('Failed Git image read: ' + path)
        total_bytes += len(data)
        value = digest(data)
        seen[blob] = value
        scanned.append({'git_blob': blob, 'path': path, 'bytes': len(data), 'sha256': value})
        if value in wanted:
            target = args.cache / (value + Path(path).suffix.lower())
            if target.exists() and digest(target.read_bytes()) != value:
                raise ValueError('Cache integrity failure')
            if not target.exists():
                target.write_bytes(data)
            matches[value] = {'sha256': value, 'git_blob': blob, 'git_paths': [path],
                              'local_path': str(target.resolve()), 'bytes': len(data)}
        if len(scanned) % 40 == 0:
            print(f'scanned {len(scanned)} unique images; exact matches {len(matches)}/{len(wanted)}', flush=True)
    rows = []
    for unit in corpus['rows']:
        images = [{**source, 'match': matches.get(source['sha256'])} for source in unit['images']]
        rows.append({'unit_id': unit['id'], 'family_id': unit['family_id'], 'images': images,
                     'complete_five_views': len(images) == 5 and all(i['match'] for i in images)})
    result = {'pin': PIN, 'corpus_path': CORPUS, 'corpus_sha256': digest(corpus_raw),
              'source_units': len(rows), 'image_references': sum(len(r['images']) for r in rows),
              'unique_required_images': len(wanted), 'unique_exact_matches': len(matches),
              'complete_units': sum(r['complete_five_views'] for r in rows),
              'complete_unit_ids': [r['unit_id'] for r in rows if r['complete_five_views']],
              'published_image_paths': len(entries), 'unique_blobs_scanned': len(scanned),
              'selected_image_search_complete': stopped is None,
              'published_image_search_complete': stopped is None and not args.source_images_only,
              'scope': ('Original source run images only' if args.source_images_only else 'All pinned runs/920 images') + '; not a search of the remote runtime.',
              'prior_lookup': str(args.prior) if args.prior else None,
              'stop_reason': stopped, 'bytes_read': total_bytes,
              'wall_seconds': time.perf_counter() - start,
              'rows': rows, 'scanned': scanned, 'model_calls': 0, 'training_updates': 0}
    write(args.output / 'image_availability.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'scanned')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
