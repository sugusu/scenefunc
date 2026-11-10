"""Evaluate an ordinary public-input preview on ALL frozen EVALFIX proposals.

Predictions are persisted before outcome rows are loaded. Old source files and
the evaluator are never edited. This is retrospective off-policy prediction,
not a new policy rollout or an estimate of rejection's causal benefit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from public_function_preview import functional_checks, predict

PIN = '75fc783f86d4fea78714639e110fbe77476ee106'
GEOMETRY_MARKER = '## SUPPLIED ROOM GEOMETRY (native room coordinates, metres):'


def canonical(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def dump(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def public_rows(path):
    """Strip saved future measurements before the predictor loop."""
    out=[]
    for line in path.read_text().splitlines():
        r=json.loads(line)
        # The original proposal belongs to the pre-dispatch observation.
        calls=r['execution']['calls']
        if len(calls)>1:raise ValueError('Unexpected multi-action proposal')
        out.append({'id':r['id'],'parent':r['parent'],
                    'call':calls[0] if calls else None,
                    'request_sha256':r['original_request_sha256']})
    return out


def geometry_of_request(raw):
    messages=json.loads(raw);texts=[]
    for m in messages:
        if m['role']!='user':continue
        if isinstance(m['content'],str):texts.append(m['content'])
        else:texts.extend(c['text'] for c in m['content'] if c.get('type')=='text')
    text='\n'.join(texts)
    if text.count(GEOMETRY_MARKER)!=1:raise ValueError('No unique public geometry envelope')
    return json.JSONDecoder().raw_decode(text.split(GEOMETRY_MARKER)[1].lstrip())[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--effects',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();sources={};cache={};predictions=[]
    files=[args.effects/'source_fragments.jsonl',args.effects/'transitions.jsonl']
    files += [Path(__file__),Path(__file__).with_name('public_function_preview.py'),
              Path(__file__).with_name('audit_prospective_guard.py')]
    before={str(p):digest(p.read_bytes()) for p in files}
    try:
        public=public_rows(files[0])
        for i,r in enumerate(public):
            cell,turn=r['id'].rsplit('__',1)
            req=f'runs/920/P920-EVALFIX-v1_20261107T130916Z/evaluation/{cell}/recorded/round_{int(turn):03d}/request.json'
            raw=subprocess.check_output(['git','show',PIN+':'+req])
            h=digest(raw);sources[req]=h
            if h!=r['request_sha256']:raise ValueError('Request differs from frozen record')
            geometry=geometry_of_request(raw);parent=r['parent']
            key=digest(canonical([parent['objects'],geometry,parent['declared_requirements'],parent['requested_mode']]).encode())
            if key not in cache:
                cache[key]=functional_checks(parent['objects'],geometry,parent['declared_requirements'],parent['requested_mode'])
            calibration={k:{'computed':v,'recorded':parent['checks'].get(k)} for k,v in cache[key].items()
                         if v!=parent['checks'].get(k)}
            tick=time.monotonic()
            pred=predict(parent,r['call'],geometry)
            predictions.append({'id':r['id'],'prediction':pred,'prediction_seconds':time.monotonic()-tick,
                                'request_path':req,'request_sha256':h,
                                'public_predictor_input_sha256':digest(canonical([parent,r['call'],geometry]).encode()),
                                'pre_functional_checks':cache[key],
                                'pre_calibration_mismatches':calibration})
            print(f'{i+1}/{len(public)} {r["id"]} {pred["status"]} calibration_mismatches={len(calibration)}',flush=True)
        pred_path=args.out/'predictions.json';dump(pred_path,predictions)
        prediction_sha=digest(pred_path.read_bytes())
        # Only now load measured outcome/after-state checks. No feedback to predict.
        outcomes={r['id']:r for r in map(json.loads,files[1].read_text().splitlines())}
        fragments={r['id']:r for r in map(json.loads,files[0].read_text().splitlines())}
        rows=[];known=correct=0
        for pred in predictions:
            r=outcomes[pred['id']]; p=pred['prediction'];given=p.get('predicted_checks',{})
            comparison={k:{'predicted':v,'observed':r['after'][k]} for k,v in given.items()
                        if v is not None and r['after'][k] is not None}
            mismatch={k:v for k,v in comparison.items() if v['predicted']!=v['observed']}
            known+=len(comparison);correct+=len(comparison)-len(mismatch)
            pre=fragments[r['id']]['before_measurements']['geometry']
            post=fragments[r['id']]['after_measurements']['geometry']
            pairs=lambda g:{tuple(sorted((x['a'],x['b']))) for x in g['positive_3d_AABB_pairs']}
            primitive=bool(pairs(post)-pairs(pre) or
                           set(post['outside_declared_wall_inner_faces'])-set(pre['outside_declared_wall_inner_faces']))
            damage=bool(r['delta']['damaged']) or primitive
            rows.append({'id':r['id'],'arm':r['arm'],'condition':r['condition'],'cell':r['cell'],
                         'native_called':r['native_called'],'tool_success':r['tool_success'],
                         'prediction_status':p['status'],'box_guard_status':p['box_guard']['status'],
                         'actual_damaged_checks':r['delta']['damaged'],
                         'actual_primitive_or_check_damage':damage,
                         'actual_repaired_checks':r['delta']['repaired'],
                         'actual_repair_without_primitive_or_check_damage':bool(r['delta']['repaired']) and not damage,
                         'predicted_lost_checks':p.get('predicted_lost_checks'),
                         'predicted_check_comparisons':len(comparison),'prediction_mismatches':mismatch})
        old=json.loads(Path('runs/autodl/20261110_supervision_horizon/guard_v2/rows.json').read_text())
        old_byid={r['id']:r for r in old}
        assert len(rows)==72 and set(outcomes)=={r['id'] for r in rows}==set(old_byid)
        assert all(p['prediction']['box_guard']==old_byid[p['id']]['prediction'] for p in predictions)
        extra=[r for r in rows if r['box_guard_status']=='allow' and r['prediction_status']=='reject']
        result={'status':'all_archived_proposals_scored','source_pin':PIN,'decisions':len(rows),
                'trajectories':len({r['cell'] for r in rows}),'independent_project_families':1,
                'prediction_status_counts':dict(Counter(r['prediction_status'] for r in rows)),
                'pre_calibration_mismatch_rows':sum(bool(p['pre_calibration_mismatches']) for p in predictions),
                'before_unique_functional_inputs':len(cache),
                'predicted_check_comparisons':known,'correct_predicted_checks':correct,
                'after_prediction_mismatch_rows':sum(bool(r['prediction_mismatches']) for r in rows),
                'native_called_rows':sum(r['native_called'] for r in rows),
                'damage_rows':sum(r['actual_primitive_or_check_damage'] for r in rows),
                'rejected_damage_rows':sum(r['actual_primitive_or_check_damage'] and r['prediction_status']=='reject' for r in rows),
                'allowed_damage_rows':sum(r['actual_primitive_or_check_damage'] and r['prediction_status']=='allow' for r in rows),
                'rejected_no_new_recorded_damage_rows':sum(not r['actual_primitive_or_check_damage'] and r['prediction_status']=='reject' for r in rows),
                'rejected_already_no_native_call_rows':sum(not r['native_called'] and r['prediction_status']=='reject' for r in rows),
                'extra_rejections_over_box_guard':[r['id'] for r in extra],
                'extra_rejections_with_measured_damage':sum(r['actual_primitive_or_check_damage'] for r in extra),
                'actual_repairs_without_damage':sum(r['actual_repair_without_primitive_or_check_damage'] for r in rows),
                'rejected_actual_repairs_without_damage':sum(r['actual_repair_without_primitive_or_check_damage'] and r['prediction_status']=='reject' for r in rows),
                'unchanged_box_control_on_all_rows':True,
                'predictions_persisted_before_outcome_scoring_sha256':prediction_sha,
                'prediction_seconds':sum(p['prediction_seconds'] for p in predictions),
                'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'model_calls':0,'training_updates':0,'new_native_actions':0,'new_scene_generations':0,
                'whole_scene_success':None,'training_ready':False,
                'scope':'Ordinary same-information pre-action proxy prediction on all saved development proposals; not independent generalization, causal avoided-damage evidence, human preference, or a full policy run.'}
        dump(args.out/'outcome_comparison.json',rows);dump(args.out/'summary.json',result)
        dump(args.out/'source_hashes.json',{'local':before,'git_requests':sources})
        if any(digest(Path(p).read_bytes())!=h for p,h in before.items()):raise ValueError('Old source changed')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(args.out/'failure.json',{'error':type(exc).__name__+': '+str(exc),'completed_predictions':len(predictions),
                                     'source_hashes':sources,'wall_seconds':time.monotonic()-start})
        raise


if __name__=='__main__':main()
