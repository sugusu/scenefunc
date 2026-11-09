"""Append the live observation envelope to the unchanged 24-unit phase corpus.

Uses only CURRENT NATIVE STATE and advertised tools already in each request.
All old task/source/feedback text stays verbatim. No post-action or target field
is passed to the formatter. New material is prospective, not authorized training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from scene_evidence import append_host_evidence, canonical_host_evidence


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def public_facts(messages):
    marker = 'Available executable tools:\n'
    system = messages[0]['content']
    if system.count(marker) != 1:
        raise ValueError('Expected exactly one source tool schema block')
    tools, _ = json.JSONDecoder().raw_decode(system.split(marker,1)[1])
    schemas = {t['name']:t['parameters'] for t in tools}
    if len(schemas) != len(tools):raise ValueError('Duplicate source tool name')
    state_marker = 'CURRENT NATIVE STATE:\n'
    texts = [p['text'] for m in messages if isinstance(m.get('content'),list)
             for p in m['content'] if p.get('type')=='text' and state_marker in p['text']]
    if len(texts) != 1:raise ValueError('Expected one current state block')
    state, _ = json.JSONDecoder().raw_decode(texts[0].split(state_marker,1)[1])
    return schemas, state


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--format', choices=['append_v1', 'canonical_v2'], default='append_v1')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    source=[json.loads(x) for x in a.source.read_text().splitlines()]
    rows=[];audit=[];states={};schemas_by_unit={}
    for c in source:
        schemas,state=public_facts(c['messages'])
        before=copy.deepcopy(c['messages'])
        new=copy.deepcopy(c)
        formatter=canonical_host_evidence if a.format=='canonical_v2' else append_host_evidence
        new['messages']=formatter(before,schemas=schemas,before_state=state,host_history=[])
        if a.format=='append_v1':assert new['messages'][:-1]==before
        # Labels, inherited coefficients and D prefills are byte-for-byte data
        # identical; the appended observation has no target access.
        assert {k:v for k,v in c.items() if k!='messages'}=={k:v for k,v in new.items() if k!='messages'}
        uid=c['unit_id']
        if uid in states:assert states[uid]==state and schemas_by_unit[uid]==schemas
        states[uid]=state;schemas_by_unit[uid]=schemas
        audit.append(dict(unit_id=uid,arm=c['arm'],stage=c['stage'],
            original_messages_sha256=digest(before),new_messages_sha256=digest(new['messages']),
            before_state_sha256=digest(state),available_tools=sorted(schemas),
            visible_object_count=len(state['objects']),targets_and_coefficients_unchanged=True,
            source_message_fields_unchanged=a.format=='append_v1',source_memory_retained=True,
            structured_source_values_preserved=True,
            new_host_history='empty at isolated example boundary; does not erase source memory'))
        rows.append(new)
    assert len(rows)==68 and len(states)==24
    paired={(r['unit_id'],r['stage']):r for r in rows if r['arm']=='paired_phase_v3'}
    for r in rows:
        if r['arm']=='separate_phase_v3':assert r['messages']==paired[(r['unit_id'],r['stage'])]['messages']
    (a.out/'components.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    record=dict(components=len(rows),units=len(states),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
        corpus_targets_or_weights_changed=False,all_original_messages_retained=a.format=='append_v1',
        structured_source_values_preserved=True,source_text_outside_moved_json_preserved=True,
        identical_observations_across_arms=True,host_envelope_shared_with_live_code=True,
        stage_phase_revision_unchanged='v3',material_revision='prospective-host-'+a.format,
        not_established=['identical tasks, system prompts and tools across archived and new native rooms',
          'complete original five-view availability','full native multi-stage rollout','learning or generation benefit'],
        new_model_calls=0,training_updates=0,scene_actions=0,wall_seconds=time.monotonic()-start)
    (a.out/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
