"""Restore a persisted actual pilot scene and checkpoint, without a model."""
import argparse
import gzip
from pathlib import Path
import shutil
from cpu_continuation import read, sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--arm',choices=['B0','B1_repaired','B2','B2_selected_recorded_state'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.arm=='B2_selected_recorded_state':
        row=read(a.root/'mechanical_checkpoint_selection.json')
        packed=Path(row['durable_scene_gzip']);expected=row['uncompressed_sha256']
        assert sha(packed)==row['compressed_sha256']
    else:
        row=next(x for x in read(a.root/'summary.json')['rows'] if x['arm']==a.arm)
        packed=Path(row['durable_scene_gzip']);expected=row['scene_sha256']
        manifest=next(x for x in read(a.root/'artifact_manifest.json')['files'] if x['path']==str(packed))
        assert sha(packed)==manifest['sha256']
    cp=packed.parent/'checkpoint.json'
    for source,expected_asset in read(cp)['payload']['geometry_hashes'].items():
        assert sha(source)==expected_asset,source
    a.output.mkdir(parents=True,exist_ok=False)
    with gzip.open(packed,'rb') as f,(a.output/'partial_scene.glb').open('wb') as g:
        shutil.copyfileobj(f,g)
    assert sha(a.output/'partial_scene.glb')==expected
    shutil.copyfile(cp,a.output/'checkpoint.json')
    print(str((a.output/'partial_scene.glb').resolve()))


if __name__=='__main__':main()
