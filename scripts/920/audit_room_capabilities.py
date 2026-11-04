"""Read-only coverage audit of the existing locked catalog and native tool scope.

Category matches are leads, not semantic, dimensional or usability acceptance.
This does not acquire assets, import a scene, call a model or create SFT records.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = datetime.now(timezone.utc).isoformat()
    root = args.root.resolve(); repo = root / 'scenefunc'
    lock = repo / 'paris/config/assets.lock.json'
    rows = json.loads(lock.read_text())['rows']
    requirements = {
        'seating': {'chair', 'sofa', 'armchair'},
        'table_surface': {'table', 'dining_table', 'coffee_table', 'desk'},
        'sleeping': {'bed', 'double_bed', 'king_bed', 'single_bed'},
        'storage': {'wardrobe', 'cabinet', 'chest_of_drawers'},
        'kitchen_storage': {'kitchen_cabinet'},
        'cooking_appliance': {'stove', 'oven', 'cooker'},
        'cold_food_storage': {'refrigerator'},
        'sink_ambiguous_kitchen_or_bath': {'sink', 'sink_cabinet'},
        'toilet': {'toilet', 'toilet_bowl', 'water_closet'},
        'bathing': {'shower', 'shower_stall', 'bathtub', 'bath'},
        'lighting': {'lamp', 'floor_lamp', 'table_lamp', 'ceiling_lamp'},
    }
    coverage = {}
    for function, terms in requirements.items():
        matches = []
        for row in rows:
            categories = {c.split('.')[0] for c in row['categories']}
            if not categories & terms:
                continue
            possible = [root/'resources/scenefunc/hssd-models'/row['path'],
                root/'resources/research_support/decision_asset_cache'/
                'datasets--hssd--hssd-models'/'snapshots'/row['revision']/row['path']]
            found = next((p for p in possible if p.is_file()), None)
            verified = False
            if found:
                verified = found.stat().st_size == row['bytes'] and digest(found) == row['sha256']
                if not verified:
                    raise ValueError('Local locked asset hash mismatch: ' + row['id'])
            matches.append({'asset_id': row['id'], 'name': row['name'],
                'categories': row['categories'], 'sha256': row['sha256'],
                'local_mesh_path': str(found) if found else None,
                'mesh_hash_verified': verified, 'function_accepted': False})
        coverage[function] = {'category_candidates': len(matches),
            'local_hash_verified_meshes': sum(m['mesh_hash_verified'] for m in matches),
            'matches': matches}
    helper = repo/'scripts/920/existing_project_session.py'
    assignment = next(n for n in ast.parse(helper.read_text()).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TOOLS' for t in n.targets))
    tools = list(ast.literal_eval(assignment.value))
    smith = root/'resources/full_runtime_20261104/scenesmith'
    native_files = [smith/'scenesmith/furniture_agents/tools/furniture_tools.py',
                    smith/'scenesmith/agent_utils/asset_registry.py']
    report = {'kind': 'read_only_source_and_cache_audit', 'started_at': started,
        'ended_at': datetime.now(timezone.utc).isoformat(), 'exit_code': 0,
        'script_sha256': digest(Path(__file__)),
        'catalog': {'path': str(lock), 'sha256': digest(lock), 'entries': len(rows),
            'scope': 'Existing experiment lock only, not the entire HSSD collection.'},
        'coverage': coverage,
        'native_scope': {'exposed_tools': tools, 'addition_exposed': 'add_furniture_to_scene_tool' in tools,
            'implementation_evidence': [{'path': str(p), 'sha256': digest(p)} for p in [helper, *native_files]],
            'source_review': 'Native add obtains an asset template from registry, copies to a unique object instance, applies placement noise, and records actual pose. Current session excludes it.',
            'next_contract': 'A separate explicit catalog-backed construction scope must bind registry, asset IDs, tool schemas, checkpoint restore and original requirements. Existing revision checkpoints must not silently change scope.',
            'full_generation_ready': False},
        'limits': ['Category membership and verified bytes do not establish scale, orientation, fit, support or usability.',
            'Mesh availability does not establish placement-tool execution or functional plumbing/electrical simulation.',
            'Privacy, partitions, windows, doors and material properties need independent structural/tool contracts.',
            'Bath/toilet proxies must remain explicit and cannot count as completed functional facilities.'],
        'model_calls': 0, 'engine_calls': 0, 'downloads': 0, 'training_updates': 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {x:v[x] for x in ['category_candidates', 'local_hash_verified_meshes']} for k,v in coverage.items()}))


if __name__ == '__main__':
    main()
