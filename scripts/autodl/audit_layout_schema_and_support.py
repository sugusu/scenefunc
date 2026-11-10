"""Inspect original sampled layouts and empirical decision-context support.

No geometry is loaded or inferred; no training, rationale authoring or preference
labels. The support witness is an analytical counterexample, not a trained model.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
                                     separators=(',',':')).encode()).hexdigest()


def main(repo,stage,out_name):
    out=stage/out_name;out.mkdir(parents=True,exist_ok=False)
    sample=stage/'il3d_static_layout_sample_v2'
    receipt=json.loads((sample/'result.json').read_text())
    assert receipt['status']=='selected8_static_layout_payloads_read_CRC_checked'
    selection=json.loads((sample/'SELECTION.json').read_text())
    assert [x['archive_member'] for x in receipt['samples']]==selection['entries']
    old_inventory=repo/'runs/autodl/20261110_sdk_asset_qualification/native_data_v2/summary.json'
    original_assets=json.loads(old_inventory.read_text())['local_assets']
    # IDs alone do not verify converted USDZ/GLB frame, shape or semantic identity.
    current_assets={x['mesh_id']:x for x in original_assets if Path(x['path']).is_file()}
    objects=[];rooms=[]
    for record in receipt['samples']:
        p=sample/record['local_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
        layout=json.loads(p.read_text());floor=layout['meshes'][0]['xyz']
        coords=[[float(v) for v in x] for x in floor]
        room=dict(sample=record['local_file'],archive_member=record['archive_member'],
            dataset_tag=layout['dataset'],object_count=len(layout['objects']),
            room_type_tags=sorted({x['roomId'] for x in layout['objects']}),
            floor_vertices_y_equal_zero=all(x[1]==0 for x in coords),
            floor_sha256=digest(floor),
            original_user_description_present=any(k in layout for k in ('prompt','instruction','description')),
            original_rationale_or_action_trace_present=any(k in layout for k in ('rationale','trajectory','tool_calls','messages')),
            geometry_examined=False,semantic_quality='unknown',train_split='not_established')
        rooms.append(room)
        for obj in layout['objects']:
            row=dict(sample=record['local_file'],object_name=obj['object_name'],
                category=obj.get('category'),label=obj.get('label'),asset_id=obj['assetId'],
                original_asset_path=obj['object_path'],
                currently_matching_local_HSSD_ID=obj['assetId'] in current_assets,
                local_mesh_reference=current_assets.get(obj['assetId'],{}).get('path'),
                position_Y_up_m=obj['position'],rotation_original=obj['rotation'],
                scale_original=obj['scale'],bbox_original=obj['bbox'],
                geometry_and_USDZ_to_GLB_equivalence_verified=False,
                object_name_not_a_semantic_certificate=True)
            objects.append(row)
    dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    dump(out/'sample_rooms.json',rooms);dump(out/'sample_objects.json',objects)
    with (out/'sample_objects.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(objects[0]));writer.writeheader();writer.writerows(objects)
    components=repo/'runs/autodl/20261110_decision_phase_study/actual_v3/prospective_phase_components.jsonl'
    rows=[x for x in map(json.loads,components.read_text().splitlines())
          if x['arm']=='paired_phase_v3' and x['stage']=='action']
    def text_only(value):
        if isinstance(value,list):
            return [text_only(x) for x in value if not (isinstance(x,dict)
                    and x.get('type') in ('image','image_url'))]
        if isinstance(value,dict):
            return {k:text_only(v) for k,v in value.items() if k not in ('image','image_url')}
        return value
    grouped=defaultdict(list)
    for x in rows:grouped[digest(x['messages'])].append(x)
    # On unique observed X values, both an X-only lookup and an X,D lookup can
    # reproduce the same finite target table. This is not optimization or evidence
    # that the actual VLM memorizes, and says nothing about held-out performance.
    unique_targets=all(len({x['target_sha256'] for x in g})==1 for g in grouped.values())
    witness=[]
    for xhash,g in grouped.items():
        witness.append(dict(X_sha256=xhash,
            text_without_image_references_sha256=digest(text_only(g[0]['messages'])),
            unit_ids=[x['unit_id'] for x in g],
            family_ids=sorted({x['family_id'] for x in g}),
            observed_A_sha256=sorted({x['target_sha256'] for x in g}),
            observed_D_prefix_sha256=sorted({digest(x['assistant_prefix_token_ids']) for x in g}),
            no_D_prefix=all(not x['assistant_prefix_token_ids'] for x in g)))
    dump(out/'empirical_support.json',dict(components=len(rows),unique_X=len(grouped),
        input_identity_scope='Serialized messages; text-only hashes remove image references. No model token/pixel activation equality asserted.',
        unique_text_only_X=len({digest(text_only(x['messages'])) for x in rows}),
        observed_X_with_multiple_D=sum(len({digest(x['assistant_prefix_token_ids']) for x in g})>1 for g in grouped.values()),
        X_only_table_can_represent_all_observed_action_targets=unique_targets,
        hypothetical_predictor_not_trained=True,
        causal_role_of_D_identified_from_training_transcripts_alone=False,
        no_new_counterfactual_ground_truth_created=True,records=witness))
    result=dict(status='original_layout_schema_and_empirical_support_read',
        author_static_layouts=len(rooms),object_instances=len(objects),
        dataset_tags=dict(Counter(x['dataset_tag'] for x in rooms)),
        room_type_tags=dict(Counter(t for x in rooms for t in x['room_type_tags'])),
        unique_floor_geometries=len({x['floor_sha256'] for x in rooms}),
        layouts_are_independent_user_tasks='not_established',
        raw_original_task_descriptions=sum(x['original_user_description_present'] for x in rooms),
        raw_original_reasoning_action_trajectories=sum(x['original_rationale_or_action_trace_present'] for x in rooms),
        matching_available_local_HSSD_ID_instances=sum(x['currently_matching_local_HSSD_ID'] for x in objects),
        unique_matching_available_HSSD_IDs=len({x['asset_id'] for x in objects if x['currently_matching_local_HSSD_ID']}),
        complete_exact_geometry_replay_qualified=False,
        code_coordinates='Author sft_dataset.py explicitly states Y-up and metres. get_usda_rooms.py multiplies translation by100 for USD export; raw scale and rotateXYZ/orient are applied.',
        source_SFT='Author constructs JSON-only final placements from layouts and external assets.json descriptions. The8raw layouts do not themselves contain diagnostic rationales or original full user tasks.',
        geometry_quality_labels_created=0,training_updates=0,model_forwards=0,paid_API_calls=0,
        input_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [components,old_inventory,stage/'il3d_source/sft_dataset.py',
                                stage/'il3d_source/get_usda_rooms.py']})
    dump(out/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--out-name',default='actual_input_audit_v1')
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve(),a.out_name)
