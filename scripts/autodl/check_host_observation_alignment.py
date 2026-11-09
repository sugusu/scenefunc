"""Verify new formatting on archived observations, without executing actions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from materialize_host_evidence import public_facts
from phase_host import NativeRoomPort
from scene_evidence import (HOST_MARKER, STATE_MARKER, TOOLS_MARKER,
                            append_host_evidence, canonical_host_evidence, text_slots)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(messages):
    return json.loads(messages[-1]['content'][0]['text'].split(HOST_MARKER, 1)[1])


def visible_images(messages):
    return [p['image'] for m in messages if isinstance(m.get('content'), list)
            for p in m['content'] if p.get('type') == 'image']


def plain(messages):
    """Remove only structured spans/placeholders for prose equality checking."""
    out = copy.deepcopy(messages)
    for slot, key in text_slots(out):
        for marker in (TOOLS_MARKER, STATE_MARKER):
            if marker not in slot[key]:
                continue
            pre, post = slot[key].split(marker)
            if post.startswith('[See '):
                end = post.index(']') + 1
            else:
                _, end = json.JSONDecoder().raw_decode(post)
            slot[key] = pre + marker + '<STRUCTURED>' + post[end:]
    return out


def port_observe(base_messages, state, schemas, fmt, history):
    env = SimpleNamespace(stage='furniture', scene=SimpleNamespace(to_state_dict=lambda: state),
                          mem={}, tool_map={k: SimpleNamespace(params_json_schema=v) for k, v in schemas.items()})
    env.observation = lambda: {'messages': base_messages, 'step_idx': 0}
    port = NativeRoomPort(env, allowed_tools=tuple(schemas), load_images=visible_images,
                          acceptance=lambda _: None, evidence_format=fmt)
    return port.observe(history)['messages']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--material', type=Path, required=True)
    ap.add_argument('--native-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    inputs = [json.loads(s) for s in a.source.read_text().splitlines()]
    outputs = [json.loads(s) for s in a.material.read_text().splitlines()]
    assert len(inputs) == len(outputs) == 68
    rows = []
    for old, new in zip(inputs, outputs):
        assert {k:v for k,v in old.items() if k != 'messages'} == {k:v for k,v in new.items() if k != 'messages'}
        schemas, state = public_facts(old['messages'])
        actual = port_observe(old['messages'], state, schemas, 'canonical_v2', [])
        assert actual == new['messages']
        e = decoded(actual)
        c = e['source_context']; objs = [dict(p, **x) for p,x in zip(e['actual_scene_state']['objects'], c['object_extra_fields'])]
        if c['object_layout']['kind'] == 'dict':
            objs = dict(zip(c['object_layout']['keys'], objs))
        assert dict(c['non_object_fields'], objects=objs) == state
        assert plain(old['messages']) == plain(actual[:-1])
        assert visible_images(old['messages']) == visible_images(actual)
        rows.append({'unit_id': old['unit_id'], 'arm': old['arm'], 'stage': old['stage'],
                     'all_source_json_values_recovered': True, 'surrounding_prose_exact': True,
                     'live_port_matches_material': True, 'targets_weights_images_unchanged': True})
    native_rows = []
    for current, previous in [('actual_v2_resume','actual_v1'), ('actual_v3_stop','actual_v2_resume')]:
        req = a.native_root/current/'native_stage/phase_host/turn_000/request.json'
        st = a.native_root/previous/'native_stage/phase_host/final_state.json'
        messages = json.loads(req.read_text()); state = json.loads(st.read_text());e=decoded(messages)
        for fmt in ('append_v1', 'canonical_v2'):
            observed = port_observe(messages[:-1], state, e['executable_tools'], fmt, e['host_execution_history'])
            assert observed == messages  # No legacy blocks: both formats exactly match.
        native_rows.append({'request': str(req), 'request_sha256':sha(req), 'state':str(st),
                            'state_sha256':sha(st), 'cached_observation_exact_in_both_formats':True})
    base = inputs[0]['messages']; schemas,state=public_facts(base)
    failures=[]
    for formatter in (append_host_evidence, canonical_host_evidence):
        once=formatter(base, schemas=schemas, before_state=state, host_history=[])
        try:formatter(once, schemas=schemas, before_state=state, host_history=[])
        except ValueError:failures.append(formatter.__name__+': duplicate rejected')
        else:raise AssertionError('Duplicate evidence accepted')
    wrong=copy.deepcopy(state);obj=next(iter(wrong['objects'].values())) if isinstance(wrong['objects'],dict) else wrong['objects'][0]
    obj['immutable']=not obj['immutable']
    for label, badstate,badschemas in [('stale_state',wrong,schemas), ('wrong_schema',state,{'unavailable':{}})]:
        try:canonical_host_evidence(base,schemas=badschemas,before_state=badstate,host_history=[])
        except ValueError:failures.append(label+': rejected')
        else:raise AssertionError('Disagreement silently normalized')
    summary={'components':len(rows),'units':len({r['unit_id'] for r in rows}), 'rows':rows,
             'native_cached_observations':native_rows,'negative_cases':failures,
             'model_calls':0,'native_calls':0,'render_calls':0,'training_updates':0,
             'source_sha256':{str(a.source):sha(a.source),str(a.material):sha(a.material)},
             'scope':'Current NativeRoomPort.observe on archived input/state fixtures. No renderer, SDK service or new trajectory executed.'}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','source_sha256')}))


if __name__=='__main__':main()
