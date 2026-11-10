"""Load existing HSM data with the exact native loader; audit acquisition scope.

No CLIP inference or generated geometry. A corpus index is not a local asset
catalogue: enumerate the native mesh paths that actually exist and quantify
orientation metadata without pretending it determines material or mass.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main(a):
    start = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    source_hash = sha(a.loader)
    inputs = {str(p): sha(p) for p in a.data.iterdir() if p.name in
              ('hssd_wnsynsetkey_index.json', 'clip_hssd_embeddings.npy',
               'clip_hssd_embeddings_index.yaml', 'object_categories.json')}
    spec = importlib.util.spec_from_file_location('qualified_native_hssd_loader', a.loader)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    data = loader.load_preprocessed_data(a.data)
    tree = ast.parse(a.similarity.read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'filter_meshes_by_category')
    ns = dict(HssdPreprocessedData=loader.HssdPreprocessedData,
              console_logger=logging.getLogger('hssd_qualification'))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(a.similarity), 'exec'), ns)
    assert data.clip_embeddings.ndim == 2 and data.clip_embeddings.shape[1] == 1024
    cardinality_equal = len(data.embedding_index) == data.clip_embeddings.shape[0]
    if len(data.embedding_index) > data.clip_embeddings.shape[0]:
        raise ValueError('Index references nonexistent embedding rows')
    assert len(set(data.embedding_index)) == len(data.embedding_index)
    assert np.isfinite(data.clip_embeddings).all()
    existing = {}
    for index, mesh_id in enumerate(data.embedding_index):
        try:
            path = loader.construct_hssd_mesh_path(a.meshes, mesh_id)
        except FileNotFoundError:
            continue
        m = data.get_metadata(mesh_id)
        existing[index] = dict(mesh_id=mesh_id, path=str(path), bytes=path.stat().st_size,
            sha256=sha(path), metadata=None if m is None else
            dict(name=m.name, up=m.up, front=m.front, wordnet_key=m.wordnet_key))
    categories = []
    for category in data.object_categories:
        indices = ns['filter_meshes_by_category'](data, category)
        available = [i for i in indices if i in existing]
        categories.append(dict(category=category, indexed_candidates=len(indices),
            local_candidates=len(available),
            local_ids=[existing[i]['mesh_id'] for i in available]))
    for path, value in inputs.items():
        assert sha(Path(path)) == value
    assert sha(a.loader) == source_hash
    summary = dict(status='native_loader_and_index_audit_completed',
        index_shape=list(data.clip_embeddings.shape), dtype=str(data.clip_embeddings.dtype),
        wordnet_categories=len(data.metadata_by_wordnet), categories=categories,
        indexed_assets=len(data.embedding_index), local_indexed_assets=len(existing),
        cardinality_equal=cardinality_equal,
        row_identity_verified=False,
        unused_tail_rows=list(range(len(data.embedding_index), data.clip_embeddings.shape[0])),
        cardinality_interpretation='Native category search addresses only indexed rows. '
        'An extra row is not proof of shifted identities; no row is truncated or relabeled. '
        'This audit enumerates local files, not semantic retrieval correctness.',
        local_mesh_bytes=sum(v['bytes'] for v in existing.values()),
        missing_local_assets=len(data.embedding_index)-len(existing),
        local_assets=list(existing.values()),
        input_hashes=inputs, source_sha256=source_hash,
        category_source_sha256=sha(a.similarity),
        old_sources_unchanged=True, actual_clip_inference=False,
        text_retrieval_qualified=False, physical_assets_qualified=False,
        seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Native data/index compatibility only. Availability does not '
        'establish semantic suitability, SDF readiness or complete description coverage.')
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in
        ('local_assets', 'input_hashes', 'categories')}, indent=2))
    print(json.dumps(categories, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('loader', 'similarity', 'data', 'meshes', 'out'):
        ap.add_argument('--'+key, type=Path, required=True)
    main(ap.parse_args())
