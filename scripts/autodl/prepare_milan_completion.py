"""Prepare a fresh RAM workspace from frozen metadata and local licensed data.

No network, model, training, source mutation or quality-dependent retry. The
separate asset recovery and XML restoration commands verify all71 old hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--fixture-cache', type=Path, default=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/milan_completion_artifacts_20261111/raw_assets'))
    a = p.parse_args()
    if a.scratch.exists():
        raise FileExistsError('Use fresh scratch; do not overwrite an experiment')
    source = a.root/'source/runs/920/P920-MILAN-TASK-v1_20261105T224015Z/input_metadata'
    manifest = json.loads((source/'manifest.json').read_text())
    metadata = ['house_state.json','runtime.json','catalog.json','task.json','provenance.json']
    for name in metadata:
        assert sha(source/name) == manifest['files'][name]['sha256']
    fixture = a.root/'source/runs/920/P920-WHOLE-v1_20261104T191014Z/COMBINED_DOWNLOAD.json'
    meshes = [x for x in json.loads(fixture.read_text())['files'] if x['role']=='mesh']
    for row in meshes:
        path = a.fixture_cache/(row['asset_id']+'.glb')
        assert path.is_file() and sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
    bundle = a.scratch/'bundle';bundle.mkdir(parents=True)
    for name in ['manifest.json',*metadata]:
        shutil.copyfile(source/name,bundle/name)
    raw = a.scratch/'raw_assets';raw.mkdir()
    for row in meshes:
        name=row['asset_id']+'.glb';shutil.copyfile(a.fixture_cache/name,raw/name)
    receipt=dict(metadata_copied=6,licensed_fixture_meshes_copied=len(meshes),
        total_fixture_bytes=sum(x['bytes'] for x in meshes),new_downloads=0,
        source_metadata_unchanged=True, model_calls=0,training_updates=0,
        next_commands=['recover_milan_assets.py','restore_milan_bundle.py','complete_milan_workspace.py'])
    (a.scratch/'prepare_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
