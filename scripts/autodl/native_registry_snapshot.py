"""Lossless native SceneObject registry snapshot for isolated continuation.

Uses existing native object serialization. Does not derive support/physics,
normalize placed objects into asset templates, authorize services, or certify
geometry resources. Resource hashes/closure must be checked separately.
"""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def save_registry_snapshot(registry,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    records=[obj.to_dict() for obj in registry.list_all()]
    ids=[r['object_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate registry identity')
    payload={'format':'native_full_object_registry_v1','records':records}
    document={'payload':payload,'sha256':hashlib.sha256(canonical(payload).encode()).hexdigest(),
        'scope':'State serialization only; direct paths retained, resource existence/closure and physics not certified.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return document


def load_registry_snapshot(path,*,registry_class,scene_object_class):
    doc=json.loads(Path(path).read_text());payload=doc['payload']
    if hashlib.sha256(canonical(payload).encode()).hexdigest()!=doc['sha256']:
        raise ValueError('Snapshot content changed')
    if payload['format']!='native_full_object_registry_v1':raise ValueError('Unknown snapshot format')
    objects=[scene_object_class.from_dict(r) for r in payload['records']]
    if len({str(o.object_id) for o in objects})!=len(objects):raise ValueError('Duplicate registry identity')
    # Return a new registry; never clear or partially overwrite a live one.
    # Native autosave uses a narrower serializer, so it is intentionally off.
    restored=registry_class(auto_save_path=None)
    for obj in objects:restored.register(obj)
    return restored
