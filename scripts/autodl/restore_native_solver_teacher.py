"""Reconstruct the tool-produced native bundle from persisted metadata and assets.

No tool action is re-executed and no geometry/model is generated. The complete
original resource allowlist is validated before copying, then final metadata is
verified against the saved manifest. Use a new RAM destination for large assets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

R=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source-bundle',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();source=a.source_bundle.resolve();dest=a.destination.resolve()
    if dest.exists():raise FileExistsError('Restoration destination must be new')
    sys.path.insert(0,str(R/'runs/autodl/20261110_public_function_search/original_evaluator'))
    from native_project import validate_bundle
    im,house,_,_=validate_bundle(source)
    protocol=read(root/'protocol.json');saved=read(root/'final_bundle_metadata.json')
    assert sha(source/'manifest.json')==protocol['source_manifest_sha256']
    assert saved['manifest']['checkpoint_parent']['manifest_sha256']==sha(source/'manifest.json')
    shutil.copytree(source,dest)
    house['rooms'][im['selected_room_id']]=read(root/'state_002.json')
    dump(dest/'house_state.json',house);dump(dest/'runtime.json',saved['runtime'])
    dump(dest/'manifest.json',saved['manifest'])
    fm,fh,_,_=validate_bundle(dest)
    assert fh['rooms'][im['selected_room_id']]==read(root/'state_002.json')
    assert all((dest/name).read_bytes()==(source/name).read_bytes() for name in im['files']
               if name not in ['house_state.json','runtime.json'])
    result=dict(status='persisted_native_teacher_bundle_reconstructed',
        original_manifest_sha256=sha(source/'manifest.json'),
        final_manifest_sha256=sha(dest/'manifest.json'),
        final_state_sha256=sha(root/'state_002.json'),source_resources_exact=True,
        objects=len(fh['rooms'][im['selected_room_id']]['objects']),
        model_calls=0,paid_API_calls=0,training_updates=0,tool_reexecutions=0,
        destination=str(dest),limits='Reconstruction and checksums, not new scene quality evidence')
    dump(root/'reconstruction_receipt.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
