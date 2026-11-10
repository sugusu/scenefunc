"""Ordinary checkpoint of native scene, memory, registry and mutable SDF bytes.

Qualified scope: rescaling already-registered furniture in an owned work copy.
Not a new learning algorithm, not a snapshot of arbitrary acquisition workers,
GPU/RNG state or external services. Registry/scene object identities stay intact.
"""
import copy
import hashlib
from pathlib import Path


class NativeEditSnapshot:
    admitted_tools=frozenset({'rescale_furniture_tool'})

    def __init__(self,env,*,owned_root,snapshot_dir):
        if env.stage!='furniture':raise ValueError('Only furniture editing qualified')
        self.env=env;self.root=Path(owned_root).resolve()
        self.snapshot_dir=Path(snapshot_dir).resolve();self.snapshot_dir.mkdir(parents=True,exist_ok=False)
        self.scene_state=copy.deepcopy(env.scene.to_state_dict());self.memory=copy.deepcopy(env.mem)
        self.step_idx=env.step_idx;self.registry=env.agent.asset_manager.registry
        self.registry_path=self.snapshot_dir/'registry_snapshot.json'
        self.registry.save_to_file(self.registry_path)
        paths={o.sdf_path.resolve() for o in list(env.scene.objects.values())+self.registry.list_all() if o.sdf_path}
        if self.registry.auto_save_path:paths.add(self.registry.auto_save_path.resolve())
        self.file_bytes={}
        for path in sorted(paths):
            if not path.is_relative_to(self.root) or not path.is_file():raise ValueError('Mutation outside owned tree: '+str(path))
            self.file_bytes[path]=path.read_bytes()
        self.scalar_scene_state={name:getattr(env.scene,name) for name in ('_surface_id_counter','room_type') if hasattr(env.scene,name)}
        self.identities=(id(env.scene),id(self.registry))

    def restore(self,*,restore_assets=True):
        env=self.env
        if (id(env.scene),id(env.agent.asset_manager.registry))!=self.identities:
            raise RuntimeError('Native scene/registry alias changed')
        if restore_assets:
            for path,data in self.file_bytes.items():path.write_bytes(data)
            self.registry.load_from_file(self.registry_path)
        env.scene.restore_from_state_dict(copy.deepcopy(self.scene_state))
        env.mem=copy.deepcopy(self.memory);env.step_idx=self.step_idx
        for name,value in self.scalar_scene_state.items():setattr(env.scene,name,value)
        env.agent.rendering_manager.clear_cache()

    def check_action(self,call):
        if call.get('name') not in self.admitted_tools:
            raise ValueError('Snapshot qualification does not admit this operation')

    def file_fingerprint(self):
        return {str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.file_bytes}

