"""Posthoc read-only reference-intermediate context; zero native/model calls."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts/920'))
from native_project import sha256, validate_bundle
from native_checkpoint import canonical_hash
from conditional_room_metrics import measure_endpoint
from score_activity_transition import score_transition

pub=Path(__file__).resolve().parent
plan=json.loads((pub/'PLAN.json').read_text())
for name,digest in plan['code_sha256'].items():
    assert sha256(Path(__file__).resolve().parents[3]/'scripts/920'/name)==digest
base=Path(plan['identity_reference_bundle']);m,h,_,_=validate_bundle(base);rid=m['selected_room_id'];initial=h['rooms'][rid]
root=base.parents[1]
rows=[]
for sid,before,after,mode in [
    ('waterloo_edit_02',root/'replay/checkpoint_1',root/'replay/checkpoint_2','joined_four_place_hosting'),
    ('waterloo_edit_03',root/'replay/checkpoint_2',root/'replay/checkpoint_3','separate_work'),
    ('waterloo_initial_work_stop',base,base,'separate_work')]:
    values=[]
    for bundle in (before,after):
        bm,bh,_,_=validate_bundle(bundle);state=bh['rooms'][rid]
        metric=measure_endpoint(state,initial,bundle/('room_'+rid),bundle,plan['program'],mode)
        values.append(dict(bundle=str(bundle),manifest_sha256=sha256(bundle/'manifest.json'),state_sha256=canonical_hash(state),evaluation=metric))
    cost=int(before!=after)
    rows.append(dict(state_id=sid,before=values[0],after=values[1],transition=score_transition(values[0]['evaluation']['review'],values[1]['evaluation']['review'],attempts=cost,native_calls=cost,state_changes=cost)))
result=dict(scope='Posthoc context from already-saved researcher demonstration checkpoints. No new scene execution. Historical action costs only; intermediate reference failure is not a terminal task failure.',native_calls_this_analysis=0,model_calls=0,training_updates=0,script_sha256=sha256(Path(__file__)),rows=rows)
(pub/'REFERENCE_CONTEXT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for x in rows:print(x['state_id'],x['transition']['outcome'],x['transition']['final_failed'])
