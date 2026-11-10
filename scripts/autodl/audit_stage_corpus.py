"""Inventory actual target tools and paired supervision, never infer success."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--material',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    material=[json.loads(s) for s in a.material.read_text().splitlines()]
    known={'add_furniture_to_scene_tool':'add','move_furniture_tool':'move',
           'remove_furniture_tool':'remove'}
    ds={r['unit_id'] for r in material if r['stage']=='decision'}
    paired={r['unit_id']:r for r in material if r['stage']=='action' and r['arm']=='paired_phase_v3'}
    rows=[];families=defaultdict(list)
    for r in material:
        if r['stage']!='action' or r['arm']!='separate_phase_v3':continue
        p=paired[r['unit_id']]
        assert r['target_token_ids']==p['target_token_ids']
        assert r['token_objective_weights']==p['token_objective_weights']
        assert r['messages']==p['messages']
        text=r['target_text'];tool=None
        if '<tool_call>' in text:
            call,end=json.JSONDecoder().raw_decode(text.split('<tool_call>',1)[1].lstrip())
            tool=call['name'];kind=known.get(tool,'unclassified')
        else:
            assert text.strip()=='<|im_end|>'
            kind='stop'
        row=dict(unit_id=r['unit_id'],family_id=r['family_id'],action_kind=kind,
                 tool=tool or '',has_decision=r['unit_id'] in ds,
                 paired_action_has_decision_prefix=bool(p['assistant_prefix']),
                 target_tokens=len(r['target_token_ids']),
                 action_objective_mass=sum(r['token_objective_weights']),
                 stage_scope='furniture' if kind!='unclassified' else 'unclassified')
        assert row['has_decision']==row['paired_action_has_decision_prefix']
        rows.append(row);families[row['family_id']].append(row['unit_id'])
    masses={arm:{stage:sum(sum(r['token_objective_weights']) for r in material
                 if r['arm']==arm and r['stage']==stage) for stage in ('action','decision')}
            for arm in ('separate_phase_v3','paired_phase_v3')}
    assert masses['separate_phase_v3']==masses['paired_phase_v3']
    summary=dict(units=len(rows),families=len(families),decision_units=len(ds),
                 missing_decision_units=[r['unit_id'] for r in rows if not r['has_decision']],
                 action_counts=dict(Counter(r['action_kind'] for r in rows)),
                 stage_counts=dict(Counter(r['stage_scope'] for r in rows)),
                 family_units=dict(families),objective_mass=masses,
                 source_sha256=hashlib.sha256(a.material.read_bytes()).hexdigest(),
                 interpretation='Target tools cover furniture edits/additions and STOP only. Families are existing development sources, not independent held-out full-generation episodes.',
                 full_generation_success='not evaluated',model_calls=0,training_updates=0)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.out/'units.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('missing_decision_units','family_units')}))


if __name__=='__main__':main()
