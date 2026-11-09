"""Independent source-token accounting and publication verification, no models."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT=Path('runs/autodl/20261110_decision_phase_study')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for path,want in manifest['source_sha256'].items():assert sha(Path(path))==want,path
    for path,want in manifest['output_sha256'].items():assert sha(ROOT/path)==want,path
    old=[json.loads(s) for s in Path('runs/autodl/20261110_material_routing/actual/prospective_components.jsonl').read_text().splitlines()]
    old={(r['unit_id'],r['component']):r for r in old if r['arm']=='paired_tagged'}
    new=[json.loads(s) for s in (ROOT/'actual_v3/prospective_phase_components.jsonl').read_text().splitlines()]
    expected={(u,c,i):(r['target_token_ids'][i],r['target_component_weight'])
              for (u,c),r in old.items() for i in range(len(r['target_token_ids']))}
    assert len(expected)==2975 and len(new)==68
    seen=defaultdict(dict);paired={};stage_counts=Counter()
    for row in new:
        arm=row['arm'];key=row['unit_id'],row['stage'];stage_counts[arm,row['stage']]+=1
        for tok,weight,origin in zip(row['target_token_ids'],row['token_objective_weights'],row['source_token_origins'],strict=True):
            k=row['unit_id'],origin['component'],origin['index']
            assert k not in seen[arm]
            seen[arm][k]=(tok,weight)
        invariant=(row['target_token_ids'],row['token_objective_weights'],row['messages'])
        if key in paired:assert paired[key]==invariant
        else:paired[key]=invariant
        if row['stage']=='decision' or arm=='separate_phase_v3':assert row['assistant_prefix']==''
        if row['target_kind']=='terminal_control' and row['stage']=='action':
            assert row['target_token_ids']==[151645,198]
    assert set(seen)=={'paired_phase_v3','separate_phase_v3'}
    assert all(v==expected for v in seen.values())
    assert all(stage_counts[arm,'decision']==10 and stage_counts[arm,'action']==24 for arm in seen)
    encoded=json.loads((ROOT/'phase_encoding_and_loss_v3.json').read_text())
    assert len(encoded['records'])==68
    assert Counter(r['status'] for r in encoded['records'])=={'encoded':16,'missing_original_views':52}
    assert not encoded['full_corpus_training_ready'] and not encoded['cuda_initialized']
    assert encoded['loss_probe']['gradient_exact'] and encoded['loss_probe']['microbatch_sum_exact']
    gate=json.loads((ROOT/'ordinary_stop_gate.json').read_text())
    assert len(gate['rows'])==24
    assert sum(r['edit_stop_label_agreement'] is True for r in gate['rows'])==6
    assert sum(r['edit_stop_label_agreement'] is None for r in gate['rows'])==18
    assert all(r['gate'].get('whole_room_success','unknown')=='unknown' for r in gate['rows'])
    reference=json.loads((ROOT/'objective_and_reference_check_v3.json').read_text())
    assert reference['reference_margin_probe']['actual_gradients']==[-1.,1.]
    assert not reference['cuda_initialized'] and reference['trained_parameter_updates']==0
    for f in manifest['source_sha256']:compile(Path(f).read_text(),f,'exec')
    result={'status':'passed','manifest_source_files':len(manifest['source_sha256']),
            'manifest_output_files':len(manifest['output_sha256']),
            'source_token_positions_verified_per_arm':len(expected),
            'phase_components_verified':len(new),'per_arm_D_A_components':[10,24],
            'original_targets_and_coefficients_preserved':True,
            'ordinary_gate_recomputed':{'supported':6,'agreement':6,'unknown':18},
            'full_training_ready':False,'new_model_calls':0,'new_training_updates':0,
            'scope':'File/hash and independent accounting verification. Actual encoding/CE gradients in prior execution receipt; no independent re-run of models or mesh physics.'}
    (ROOT/'verification_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
