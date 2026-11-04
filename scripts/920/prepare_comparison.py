"""Freeze four same-start G/U requests before any formal model generation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    run = root / 'runtime/P920-DD-v1_20261104T083214Z'
    work = run / 'work'
    out = run / 'comparison_inputs'
    out.mkdir(exist_ok=False)
    sys.path.insert(0, str(work / 'native_local_inference_20261104'))
    from local_vlm import LocalQwenEngine
    import torch
    model = read(root/'MODEL_DOWNLOAD_PLAN.json')['destination']
    engine = LocalQwenEngine(model, out/'encode', max_pixels=65536, max_input_tokens=12288, max_new_tokens=2048)
    spec = importlib.util.spec_from_file_location('native_memory', work/'professional_data_ingest_20261104/native_src/memory.py')
    memory_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory_module)
    common = '''\n\n## Frozen 920 single-revision protocol
Continue from the actual furnished checkpoint below. The attached views, ordered
topdown, south, east, west, north, show its actual geometry with a ceiling cutaway;
there are no object-ID overlays. The task, source shell and assets are unchanged.
Use actual object IDs and dimensions in the state evidence. You have ONE revision
with at most 12 emitted tool calls, including invalid or unsupported calls.
A program with more than 12 calls is rejected without executing any call.
Executable tools are generate_assets, add_furniture_to_scene_tool,
move_furniture_tool and remove_furniture_tool. Other tools return explicit failures.
generate_assets searches only the same frozen 23-mesh selected HSSD inventory,
using category filtering and size ranking; it is not external asset generation.
Assets are not rescaled. Missing requested categories return failures.
Preserve the original requirements. An intersection count is not a usability
score, and deleting a needed object is not automatically an improvement.
The native placement reference is z=0 while the shell floor top is z=0.02 m;
this adapter discrepancy is reported, not repaired by unrelated XY movement.
Report remaining unknowns honestly. Do not infer complete room success.
'''
    records = []
    for arm in ('A', 'B'):
        draft = run/'drafts'/arm
        state = read(draft/'checkpoint.json')['payload']
        assert hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == read(draft/'checkpoint.json')['payload_sha256']
        for path, digest in state['geometry_hashes'].items():
            assert sha(path) == digest
        views = read(draft/'views/receipt.json')
        assert views['status'] == 'completed' and views['source_sha256'] == sha(draft/'partial_scene.glb')
        for image in views['views']:
            assert sha(image['path']) == image['sha256']
        audit = read(draft/'audit.json')
        measurement = read(draft/'use_measurement.json')
        assert audit['scene_sha256'] == measurement['scene_sha256'] == views['source_sha256']
        memory = memory_module.empty_memory('furniture')
        for event in state['tool_history']:
            call = event['call']
            memory_module.update_furniture(memory, call['name'], call['arguments'], event['response'])
        memory_text = memory_module.render_memory('furniture', memory)
        original = read(work/'paris_A_inference_20261104'/f'request_{arm}.json')
        task = original[1]['content'][-1]['text'].split('## Memory')[0].strip()
        assets = [{k: row[k] for k in ['object_id', 'actual_name', 'categories', 'width', 'depth', 'height', 'rescaled']} for row in state['asset_records']]
        geometry = {k: audit[k] for k in ['furniture_surface_intersection_pairs', 'architecture_surface_intersection_pairs', 'out_of_rectangular_floor_objects', 'objects', 'floor_reference']}
        use = {'scope': 'Measured planar footprint evidence; no target coordinates or repair actions',
               'work_materials': 'Not explicit mesh objects; tested footprint is 0.80 x 0.50 m on the candidate surfaces below.',
               'dining_footprint_m': [0.60, 0.40],
               'surfaces': [{k: s[k] for k in ['object_id', 'z_m', 'surface_m2', 'free_section_m2', 'blockers', 'uncertain_section_objects', 'candidate_counts']} for s in measurement['surfaces']],
               'separate_use': measurement['results']['scene_section_free']['separate_use']['status'],
               'persistent_work_simultaneous_dining': measurement['results']['scene_section_free']['simultaneous_use']['status'],
               'search_limit': '1 cm grid, 0/90 degrees, at most 100000 pair comparisons; not_found does not prove global infeasibility.',
               'unknown': ['seated posture', 'chair pullout', 'walking reachability', 'storage capacity and transition', 'sleep access', 'door swing', 'whole-room success']}
        for method in ('G', 'U'):
            label = arm+'_'+method
            directory = out/label
            directory.mkdir()
            content = [{'type': 'image', 'image': x['path']} for x in views['views']]
            content.append({'type':'text', 'text': task+'\n\n## Current ordinary memory\n'+memory_text+'\n\n## Actual assets\n'+json.dumps(assets, ensure_ascii=False)+'\n\n## Measured ordinary geometry feedback\n'+json.dumps(geometry, ensure_ascii=False)})
            if method == 'U':
                content.append({'type':'text', 'text':'## Additional measured functional-use feedback\n'+json.dumps(use, ensure_ascii=False)})
            messages = [{'role':'system', 'content':original[0]['content']+common}, {'role':'user', 'content':content}]
            dump(directory/'request.json', messages)
            dump(directory/'feedback.json', {'geometry':geometry, 'functional_use': use if method == 'U' else None})
            tensors, encoded = engine.encode(messages)
            torch.save(dict(tensors), directory/'inputs.pt')
            record = {'label':label, 'arm':arm, 'method':method, 'input_tokens':encoded['input_tokens'],
                      'request_sha256':sha(directory/'request.json'), 'input_tensor_sha256':sha(directory/'inputs.pt'),
                      'checkpoint_sha256':sha(draft/'checkpoint.json'), 'scene_sha256':views['source_sha256'],
                      'encoding':encoded, 'feedback_sha256':sha(directory/'feedback.json')}
            dump(directory/'receipt.json', record)
            records.append(record)
    assert not torch.cuda.is_initialized()
    for arm in ('A','B'):
        g = read(out/f'{arm}_G/request.json'); u = read(out/f'{arm}_U/request.json')
        assert g[0] == u[0] and g[1]['content'] == u[1]['content'][:-1]
    protocol = {'status':'frozen_before_formal_calls', 'records':records,
                'probe_label':max(records,key=lambda x:x['input_tokens'])['label'],
                'max_input_tokens':12288, 'input_cap_reason':'Common prospective cap allowing the additional measured feedback without truncation; actual encoded lengths recorded for all four branches.',
                'max_pixels_per_image':65536, 'max_new_tokens':2048, 'seed':0, 'batch_size':1,
                'precision':'float16', 'attention':'native torch SDPA', 'use_cache':True,
                'max_tool_calls':12, 'over_budget_policy':'Reject entire emitted program without tool execution',
                'formal_model_calls_planned':4, 'draft_model_calls_planned':0,
                'same_start_same_views_same_assets_checked':True}
    dump(out/'PROTOCOL.json', protocol)
    print(json.dumps({'tokens':{r['label']:r['input_tokens'] for r in records}, 'probe_label':protocol['probe_label']}))


if __name__ == '__main__':
    main()
