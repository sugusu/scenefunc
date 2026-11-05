"""Resolve the frozen priority list; no candidate search or credential logging."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
from huggingface_hub import HfApi

P = Path(__file__).resolve().parent.parent
R = P.parents[2]
request = json.loads((P/'ASSET_REQUEST.json').read_text())
print(datetime.now(timezone.utc).isoformat(), 'resolve four frozen metadata paths', flush=True)
result = {'status': 'running', 'found': [], 'credential_values_logged': False}
code = 0
try:
    api = HfApi(token=True)
    for repo, rev, suffix, role in [
        ('hssd/hssd-models', '208ea6bc22d863bbc1cbaef1829b3e646600dfb6', '.glb', 'mesh'),
        ('hssd/hssd-hab', '4369cb9876214c7fbebcf552eb532380e4d287e4', '.object_config.json', 'coordinate_frame'),
    ]:
        paths = {f"objects/{r['id'][0]}/{r['id']}{suffix}": r for r in request['candidates']}
        for entry in api.get_paths_info(repo, list(paths), repo_type='dataset', revision=rev):
            row = paths[entry.path]; lfs = getattr(entry, 'lfs', None)
            result['found'].append({'repository': repo, 'revision': rev, 'path': entry.path,
                'bytes': entry.size, 'asset_id': row['id'], 'asset_name': row['name'], 'role': role,
                **({'sha256': lfs.sha256} if lfs else {'git_blob': entry.blob_id})})
    eligible = []
    for candidate in request['candidates']:
        files = [f for f in result['found'] if f['asset_id'] == candidate['id']]
        if len(files) == 2 and sum(f['bytes'] for f in files) <= request['budget']['max_bytes']:
            eligible.append(files)
    if not eligible:
        raise ValueError('No frozen candidate has both pinned files within budget')
    plan = {'files': eligible[0], 'cache_directory': str(R.parent/'resources/research_support/decision_asset_cache'),
            'download_script_sha256': hashlib.sha256((R/'scripts/920/fetch_asset_bundle.py').read_bytes()).hexdigest(),
            'budget': {'files': 2, 'max_bytes': request['budget']['max_bytes']}}
    with (P/'ASSET_PLAN.json').open('x') as f: json.dump(plan, f, indent=2); f.write('\n')
    result.update(status='resolved', selected_asset=eligible[0][0]['asset_id'])
except Exception as exc:
    code = 1
    result.update(status='failed', error_type=type(exc).__name__,
                  http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
finally:
    result.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=code)
    with (P/'ASSET_RESOLUTION.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(result['status'], result.get('selected_asset'), flush=True)
raise SystemExit(code)
