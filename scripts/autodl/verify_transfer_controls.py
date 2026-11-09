"""Recheck actual material, request pairs, frozen sources and published receipts.

This does not rerun the processor, inference, training, or physical executor.
Processor evidence comes from encode_transfer_control.py's saved real run.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    coverage=read(args.root/'actual_v3/coverage.json')
    attribution=read(args.root/'conditioning/attribution.json')
    encoding=read(args.root/'conditioning/actual_encoding.json')
    rewrites=read(args.root/'actual_v3/request_rewrites.json')
    # A fresh source read verifies the archived data used before transformation.
    for path,expected in coverage['source_sha256'].items():
        raw=subprocess.check_output(['git','show',coverage['pin']+':'+path])
        assert sha(raw)==expected,path
    local=dict(coverage['local_source_sha256'])
    local[attribution['source_path']]=attribution['source_sha256']
    local.update(encoding['source_sha256'])
    for path,expected in local.items():assert sha(Path(path).read_bytes())==expected,path
    original=[json.loads(s) for s in Path(attribution['source_path']).read_text().splitlines()]
    mixed=[json.loads(s) for s in (args.root/'conditioning/optional_mixed_components.jsonl').read_text().splitlines()]
    def masses(rows):
        mass=defaultdict(float)
        for r in rows:
            origins=r['source_token_origins'];tokens=r['target_token_ids'];weights=r['token_objective_weights']
            assert len(origins)==len(tokens)==len(weights)
            for o,t,w in zip(origins,tokens,weights):
                assert w>0
                mass[r['unit_id'],o['component'],o['index'],t]+=w
        return mass
    baseline=[r for r in original if r['arm']=='paired_phase_v3']
    expected=masses(baseline);observed=masses(mixed)
    assert expected.keys()==observed.keys()
    maxerr=max(abs(expected[k]-observed[k]) for k in expected)
    assert maxerr<1e-15
    keyed={(r['unit_id'],r['stage'],r['variant']):r for r in mixed}
    assert len(keyed)==len(mixed)==44
    for r in encoding['rows']:
        c=keyed[r['unit_id'],r['stage'],r['variant']]
        assert r['target_sha256']==c['target_sha256']
        if r['status']=='encoded':
            assert r['target_tokens']==len(c['target_token_ids'])
            assert abs(r['objective_weight_mass']-sum(c['token_objective_weights']))<1e-12
            assert r['original_vision_tensors_exact'] and r['all_context_labels_masked']
    assert len(encoding['rows'])==44
    assert sum(r['status']=='encoded' for r in encoding['rows'])==21
    folds={f['fold_id']:f for f in coverage['folds']}
    unit_ids={u['unit_id'] for u in coverage['units']}
    assert len(rewrites)==24*len(folds)==264
    for f in folds.values():
        assert set(f['train_unit_ids']).isdisjoint(f['held_unit_ids'])
        assert set(f['train_unit_ids'])|set(f['held_unit_ids'])==unit_ids
        if f['split_kind']=='source_and_structural_template':
            assert not f['shared_structural_signatures']
        rows=[r for r in rewrites if r['fold_id']==f['fold_id']]
        assert {r['unit_id'] for r in rows}==unit_ids
        for r in rows:
            assert not r['residual_exact_source_markers']
            assert set(r['removed_card_ids'])==set(f['source_excluded_card_ids'])
            assert r['paired_canonicalization_verified']
    example=read(args.root/'export_example/manifest.json')
    for name,expected_hash in example['files'].items():
        assert sha((args.root/'export_example'/name).read_bytes())==expected_hash
    manifest={}
    for path in sorted(args.root.rglob('*')):
        if not path.is_file() or 'source_read' in path.parts or path==args.output:continue
        if path.name=='artifact_manifest.json':continue
        manifest[str(path)]=sha(path.read_bytes())
    scripts=['audit_project_knowledge_splits.py','audit_conditioning_attribution.py',
             'export_project_control.py','encode_transfer_control.py','verify_transfer_controls.py']
    for name in scripts:
        path=Path('scripts/autodl')/name
        compile(path.read_text(),str(path),'exec')
        manifest[str(path)]=sha(path.read_bytes())
    result={'frozen_git_sources_rechecked':len(coverage['source_sha256']),
        'local_source_files_rechecked':len(local),'target_token_coefficients_checked':len(expected),
        'maximum_original_mass_error':maxerr,'folds':len(folds),'request_rewrites':len(rewrites),
        'exported_request_pair_hashes_match':True,'processor_receipts_checked':21,
        'scope':'Source and saved-evidence integrity; not a new processor/model/physics execution.',
        'old_tracked_worktree_changes':subprocess.check_output(['git','diff','--name-only','HEAD']).decode().splitlines(),
        'experiment_model_calls':0,'training_updates':0,'new_scenes':0}
    # Existing tracked outputs are intended to remain immutable this stage.
    assert not result['old_tracked_worktree_changes']
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest[str(args.output)]=sha(args.output.read_bytes())
    (args.root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
