"""Repair the observed ordering defect from original envelopes and bound states.

No new labels, images, geometry, model inference or training. Old artifacts are
read-only. Do not use the inconsistent overwritten envelope as the ID mapping.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from native_state_override import canonical
from scene_evidence import HOST_MARKER, refresh_visible_evidence


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def envelope(row):
    return json.loads(row['messages'][-1]['content'][0]['text'][len(HOST_MARKER):])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('original','rebuilt','bindings','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args(); start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    inputs={str(p):sha(p) for p in (a.original,a.rebuilt,a.bindings)}
    old=[json.loads(s) for s in a.original.read_text().splitlines()]
    recent=[json.loads(s) for s in a.rebuilt.read_text().splitlines()]
    assert len(old)==len(recent)==68
    bindings={b['unit_id']:b for b in json.loads(a.bindings.read_text())}
    output=[];audit=[]
    for archived, rebuilt in zip(old,recent):
        assert {k:v for k,v in archived.items() if k!='messages'}=={k:v for k,v in rebuilt.items() if k!='messages'}
        unit=archived['unit_id'];new=copy.deepcopy(rebuilt)
        if unit in bindings:
            binding=bindings[unit];path=Path(binding['export_receipt']).parent/'actual_room_state.json'
            state=json.loads(path.read_text());assert canonical(state)==binding['runtime_state_sha256']
            inputs[str(path)]=sha(path)
            expected=refresh_visible_evidence(envelope(archived),state)
            actual=envelope(rebuilt)
            assert actual['actual_scene_state']==expected['actual_scene_state']
            bad_keys=actual['source_context']['object_layout'].get('keys',[])
            good_keys=expected['source_context']['object_layout'].get('keys',[])
            assert {k:v for k,v in actual.items() if k!='source_context'}=={
                k:v for k,v in expected.items() if k!='source_context'}
            new['messages'][-1]['content'][0]['text']=HOST_MARKER+json.dumps(
                expected,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
            audit.append(dict(unit_id=unit,arm=archived['arm'],stage=archived['stage'],
                wrong_key_positions=sum(x!=y for x,y in zip(bad_keys,good_keys)),
                state_path=str(path),runtime_state_sha256=canonical(state),
                before_envelope_sha256=canonical(actual),after_envelope_sha256=canonical(expected)))
        else:
            assert archived==rebuilt
        # Only the evidence envelope can change; every target/image stays exact.
        assert {k:v for k,v in new.items() if k!='messages'}=={
            k:v for k,v in rebuilt.items() if k!='messages'}
        assert new['messages'][:-1]==rebuilt['messages'][:-1]
        output.append(new)
    (a.out/'components.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    assert inputs=={p:sha(p) for p in inputs}
    summary=dict(status='identity_order_repaired_not_reencoded',components=len(output),
        changed_components=len(audit),changed_units=len({r['unit_id'] for r in audit}),
        source_hashes=inputs,output_sha256=sha(a.out/'components.jsonl'),audit=audit,
        source_extra_metadata='Preserved by object ID from archived requests; not claimed live runtime metadata',
        target_tokens_weights_images_unchanged=True,model_calls=0,training_updates=0,
        new_renders=0,full_generation_runs=0,training_ready=False,
        encoding_status='Prior encoding receipts do not certify the modified message tokens; re-encoding required',
        seconds=time.monotonic()-start)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('audit','source_hashes')}))


if __name__=='__main__':main()
