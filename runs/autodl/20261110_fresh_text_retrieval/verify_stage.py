"""Verify the saved execution without rerunning a model or mesh pipeline."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def f32_array(p):
    """Minimal strict NPY v1/v2 reader for these plain float32 output arrays."""
    raw=Path(p).read_bytes()
    if raw[:6]!=b'\x93NUMPY':raise ValueError('Not NPY')
    version=tuple(raw[6:8])
    if version==(1,0):size=struct.unpack('<H',raw[8:10])[0];offset=10
    elif version==(2,0):size=struct.unpack('<I',raw[8:12])[0];offset=12
    else:raise ValueError('Unexpected array version')
    header=ast.literal_eval(raw[offset:offset+size].decode('latin1'))
    body=raw[offset+size:]
    if header['descr']!='<f4' or header['fortran_order']:
        raise ValueError('Unexpected scalar type/order; no pickle fallback')
    n=math.prod(header['shape'])
    if len(body)!=4*n:raise ValueError('Array length mismatch')
    return header['shape'],body,struct.unpack('<'+str(n)+'f',body)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--require-live-weight',action='store_true')
    args=ap.parse_args()
    manifest=read(HERE/'artifact_manifest.json')
    for row in manifest['files']:
        p=HERE/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('Artifact changed: '+row['path'])
    with zipfile.ZipFile(HERE/'evidence.zip') as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC mismatch')
        for name in z.namelist():
            if z.read(name)!=(HERE/name).read_bytes():raise ValueError('ZIP content differs')
    s=read(HERE/'fresh_v1/summary.json');calls=read(HERE/'fresh_v1/embedding_calls.json')
    rows=read(HERE/'fresh_v1/requests.json');protocol=read(HERE/'protocol.json')
    for p,h in {**s['input_hashes'],**s['source_module_hashes']}.items():
        if sha(p)!=h:raise ValueError('Old input/source changed: '+p)
    if len(calls)!=6 or len(rows)!=3 or s['server_counters']!={'total':3,'completed':3,'failed':0}:
        raise ValueError('Frozen call/request membership differs')
    if s['new_8B_VLM_calls'] or s['paid_API_calls'] or s['training_updates'] or s['new_complete_rooms']:
        raise ValueError('Execution exceeded its declared scope')
    saved=[]
    for i,row in enumerate(calls,1):
        shape,body,values=f32_array(HERE/f'fresh_v1/embedding_{i:02d}.npy')
        if shape!=(1024,) or hashlib.sha256(body).hexdigest()!=row['sha256']:
            raise ValueError('Encoder receipt differs from saved vector')
        if not all(math.isfinite(x) for x in values) or abs(math.sqrt(sum(x*x for x in values))-1)>1e-5:
            raise ValueError('Invalid saved normalized embedding')
        saved.append(values)
    archive_path=next(Path(x) for x in s['input_hashes'] if x.endswith('/text_embeddings.npy'))
    shape,_,archive=f32_array(archive_path)
    if shape!=(3,1024):raise ValueError('Original parity archive changed')
    parity=read(HERE/'fresh_v1/historical_parity.json')
    for i,row in enumerate(parity):
        error=max(abs(a-b) for a,b in zip(saved[i],archive[i*1024:(i+1)*1024]))
        if abs(error-row['max_absolute_error'])>1e-12:
            raise ValueError('Parity receipt does not match actual saved values')
    if [r['text'] for r in rows]!=protocol['fresh_descriptions'] or [r['text'] for r in calls[3:]]!=protocol['fresh_descriptions']:
        raise ValueError('Fresh request-to-forward mapping differs')
    root=HERE.parents[2]
    native_protocol=read(root/'runs/autodl/20261110_live_manager_request/protocol.json')
    assets={r['mesh_id']:r for r in read(native_protocol['inventory'])['local_assets']}
    for row in rows:
        asset=assets[row['hssd_id']]
        if sha(asset['path'])!=row['original_glb_sha256'] or asset['sha256']!=row['original_glb_sha256']:
            raise ValueError('Original selected asset changed')
        if not row['embedding_row']<protocol['excluded_mapping_suffix_starts_at_row']:
            raise ValueError('Selected asset outside declared prefix pool')
        if row['semantic_correctness_verified'] or row['physical_properties_verified']:
            raise ValueError('Unsupported semantic/physical claim')
    restoration=read(HERE/protocol['restoration_receipt']);weights=Path(restoration['weight_path'])
    cleanup=read(HERE/'bytecode_cache_cleanup.json')
    for row in cleanup['rows']:
        if sha(row['source_path'])!=row['source_sha256_before'] or row['source_sha256_after']!=row['source_sha256_before']:
            raise ValueError('Retained source differs after cache cleanup')
    weight_status='ephemeral_weight_no_longer_present'
    if weights.is_file():
        if sha(weights)!=restoration['actual_sha256']:raise ValueError('RAM checkpoint changed')
        weight_status='live_RAM_original_official_full_hash_verified'
    elif args.require_live_weight:raise ValueError('Ephemeral weight not present')
    print(json.dumps({'status':'verified','artifact_files':len(manifest['files']),
        'old_input_and_native_source_hashes':len({**s['input_hashes'],**s['source_module_hashes']}),
        'saved_embeddings_independently_parsed':len(saved),
        'historical_parity_recomputed_without_model':len(parity),
        'selected_original_GLBS_unchanged':len(rows),'ZIP_CRC_and_byte_equality':True,
        'cache_cleanup_original_sources_verified':len(cleanup['rows']),
        'weight_status':weight_status,'new_model_calls':0,'new_mesh_pipeline_calls':0,
        'semantic_or_whole_room_success_inferred':False}))


if __name__=='__main__':main()
