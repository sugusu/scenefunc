"""Compare serializers on retained native records, without generating scenes."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

from native_registry_snapshot import save_registry_snapshot,load_registry_snapshot,canonical


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    n=load(a.base/'native_room_bridge_20261104/native_kernels.py','actual_registry_classes');G=n.G
    helper=a.base/'native_state_and_supervision_20261104/source/scenesmith/utils/path_utils.py'
    G['safe_relative_path']=load(helper,'native_asset_paths').safe_relative_path
    cache=json.loads(a.checkpoint.read_text())['payload']['asset_templates']
    house=json.loads(a.paper_state.read_text());rooms=house['rooms']
    # This is the existing researcher-authored counter, not a new HSSD surface
    # annotation or a new design. Keep original placed pose in checkpoint test.
    supported=[r for room in rooms.values() for r in room['objects'].values() if r['support_surfaces']]
    if not supported:raise ValueError('No retained source surface available for this comparison')
    source_hashes={str(p):sha(p) for p in [a.checkpoint,a.paper_state,helper]}
    for r in n.READ:source_hashes[r['path']]=sha(Path(r['path']))
    groups={'actual_mesh_cache':cache,'retained_supported_records':supported}
    rows=[]
    for name,records in groups.items():
        registry=G['AssetRegistry']()
        for r in records:registry.register(G['SceneObject'].from_dict(copy.deepcopy(r)))
        before={str(o.object_id):o.to_dict() for o in registry.list_all()}
        native_path=a.out/(name+'_native.json');registry.save_to_file(native_path)
        native_restored=G['AssetRegistry']();native_restored.load_from_file(native_path)
        native_after={str(o.object_id):o.to_dict() for o in native_restored.list_all()}
        path=a.out/(name+'_complete.json');save_registry_snapshot(registry,path)
        full=load_registry_snapshot(path,registry_class=G['AssetRegistry'],scene_object_class=G['SceneObject'])
        after={str(o.object_id):o.to_dict() for o in full.list_all()}
        if canonical(before)!=canonical(after):raise AssertionError('Complete snapshot changed native object state')
        native_differences=[{'object_id':oid,'field':k} for oid in before for k in before[oid]
                            if canonical(before[oid][k])!=canonical(native_after[oid][k])]
        rows.append({'source_group':name,'objects':len(before),
            'source_surface_count':sum(len(r['support_surfaces']) for r in before.values()),
            'native_loaded_surface_count':sum(len(r['support_surfaces']) for r in native_after.values()),
            'complete_loaded_surface_count':sum(len(r['support_surfaces']) for r in after.values()),
            'native_record_differences':native_differences,'complete_record_exact':True,
            'historical_pose_preserved_not_a_generation_input':True})
    if source_hashes!={p:sha(Path(p)) for p in source_hashes}:raise AssertionError('Old inputs changed')
    result={'status':'completed','results':rows,'source_hashes':source_hashes,
        'native_serializers_executed':True,'mesh_renderer_executed':False,
        'new_model_calls':0,'training_updates':0,'new_scene_generations':0,
        'new_support_labels':0,'old_sources_unchanged':True,
        'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'Serialization fidelity using actual native classes, not a generation or physics result. Full snapshot retains existing poses; the separate geometry catalogue removes historical poses.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('base','checkpoint','paper-state','out'):ap.add_argument('--'+k,type=Path,required=True)
    main(ap.parse_args())
