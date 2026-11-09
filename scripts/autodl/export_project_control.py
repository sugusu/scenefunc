"""Export one actual matched request pair, without model calls or invented labels.

The full-card and source-filtered arms use the SAME JSON serialization. The
untouched archive is referenced separately. Caller selects a recorded fold
and unit; this does not select good outcomes or promote development to test.
"""
import argparse
import json
from pathlib import Path
import subprocess

from audit_material_routing import digest, write
from audit_project_knowledge_splits import exclude_source_cards


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coverage',type=Path,required=True)
    ap.add_argument('--fold',required=True)
    ap.add_argument('--unit',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.coverage.read_text())
    folds={f['fold_id']:f for f in report['folds']}
    units={u['unit_id']:u for u in report['units']}
    fold=folds[args.fold];unit=units[args.unit]
    raw=subprocess.check_output(['git','show',report['pin']+':'+unit['request_path']])
    assert digest(raw)==report['source_sha256'][unit['request_path']]
    original=json.loads(raw)
    excluded=set(fold['source_excluded_card_ids'])
    texts={report['knowledge'][p]['card']['principle'] for p in excluded}
    canonical,_=exclude_source_cards(original,set(),set())
    filtered,receipt=exclude_source_cards(original,excluded,texts)
    repeated,_=exclude_source_cards(canonical,excluded,texts)
    assert repeated==filtered
    assert set(receipt['removed_card_ids'])==excluded
    if receipt['residual_exact_source_markers']:
        raise ValueError('Source markers remain; do not export as a clean literal-card ablation')
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'all_cards.json',canonical)
    write(args.output/'source_filtered.json',filtered)
    files={p.name:digest(p.read_bytes()) for p in args.output.iterdir()}
    write(args.output/'manifest.json',{
        'unit_id':args.unit,'fold_id':args.fold,'pin':report['pin'],
        'source_request_path':unit['request_path'],'source_request_sha256':digest(raw),
        'coverage_sha256':digest(args.coverage.read_bytes()),'files':files,
        'canonical_all_cards_differs_from_original_format':canonical!=original,
        'paired_serialization_verified':True,**receipt,
        'image_paths':'Original image references retained; the explicit recovery manifest is still required to load them.',
        'scope':'Development request preprocessing only; source filtering is not absence of all project cues or model pretraining.',
        'model_calls':0,'training_updates':0,'scene_edits':0})
    print(json.dumps({'unit':args.unit,'files':files,'removed':sorted(excluded)}))


if __name__=='__main__':main()
