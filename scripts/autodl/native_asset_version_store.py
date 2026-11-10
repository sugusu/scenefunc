"""Ordinary content-addressed versions of owned mutable native asset files.

Qualified purpose: bind a saved SceneJSON to its SDF/registry bytes. This is not a
complete simulator snapshot, a rollout policy, or a new append-only memory method.
Immutable meshes, observations and source data need their separate provenance.
"""
import hashlib
import json
from pathlib import Path


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()


class NativeAssetVersionStore:
    def __init__(self,root,store):
        self.root=Path(root).resolve();self.store=Path(store).resolve()
        if not self.store.is_relative_to(self.root):raise ValueError('Store must be in the owned run tree')
        (self.store/'blobs').mkdir(parents=True,exist_ok=False)
        (self.store/'versions').mkdir()
    def _blob(self,data):
        digest=hashlib.sha256(data).hexdigest();path=self.store/'blobs'/digest
        if path.exists():assert path.read_bytes()==data
        else:
            with path.open('xb') as f:f.write(data)
        return digest
    def capture(self,scene,registry,*,label):
        # Public registry serialization makes the autosaved registry and the
        # actual in-memory registration agree at the recorded checkpoint.
        if not registry.auto_save_path:raise ValueError('Actual registry autosave path required')
        registry_path=Path(registry.auto_save_path).resolve()
        if not registry_path.is_relative_to(self.root):raise ValueError('Registry outside owned run')
        registry.save_to_file(registry_path)
        files=[]
        candidates=set(self.root.rglob('*.sdf'))|set(self.root.rglob('asset_registry.json'))
        candidates.add(registry_path)
        for path in sorted(candidates):
            actual=path.resolve()
            if not actual.is_relative_to(self.root) or path.is_symlink():raise ValueError('Unowned mutable path')
            data=path.read_bytes();files.append(dict(path=str(path.relative_to(self.root)),
                                                   sha256=self._blob(data),bytes=len(data)))
        state=scene.to_state_dict()
        record=dict(label=label,recorded_owned_root=str(self.root),scene_state=state,files=files,
                    registry_path=str(registry_path.relative_to(self.root)),
                    scope='Owned SDF and registry bytes + SceneJSON only; no worker/cache/RNG or immutable geometry snapshot.')
        data=canonical(record);digest=hashlib.sha256(data).hexdigest()
        p=self.store/'versions'/(digest+'.json')
        if p.exists():assert p.read_bytes()==data
        else:
            with p.open('xb') as f:f.write(data)
        return digest
    def restore(self,digest,scene,registry):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid version ID')
        data=(self.store/'versions'/(digest+'.json')).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Version metadata hash mismatch')
        record=json.loads(data)
        if record['recorded_owned_root']!=str(self.root):
            raise ValueError('New-root migration requires explicit path relocation and a separate proof')
        prepared=[]
        for item in record['files']:
            path=(self.root/item['path']).resolve()
            if not path.is_relative_to(self.root):raise ValueError('Version path escapes owned run')
            value=(self.store/'blobs'/item['sha256']).read_bytes()
            if len(value)!=item['bytes'] or hashlib.sha256(value).hexdigest()!=item['sha256']:
                raise ValueError('Version payload hash mismatch')
            prepared.append((path,value))
        # Validate every payload before changing any file. Whole-worker atomicity
        # and deletion of files created after a snapshot are not claimed.
        for path,value in prepared:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
        scene.restore_from_state_dict(record['scene_state'])
        registry.load_from_file(self.root/record['registry_path'])
        return record
