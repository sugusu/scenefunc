"""Read cached training tensors and actual rollout logs; no model or optimization."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    plan = read(args.study/'ENCODING_PLAN.json')
    tok = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True)
    template_file = Path(plan['model'])/'chat_template.json'
    template = read(template_file)['chat_template']
    # Use the processor's pinned template explicitly, not a tokenizer fallback.
    tok.chat_template = template
    start = tok.encode('<|im_start|>assistant\n', add_special_tokens=False)
    examples = []
    for row in read(args.study/'ENCODING.json')['records']:
        path = Path(row['tensor_path'])
        assert sha(path) == row['tensor_sha256']
        batch = torch.load(path, map_location='cpu', weights_only=True)
        ids = batch['input_ids'][0].tolist()
        starts = [i+len(start) for i in range(len(ids)-len(start)+1) if ids[i:i+len(start)] == start]
        assert len(starts) == 1
        begin = starts[0]
        apos = batch['action_mask'][0].nonzero().flatten().tolist()
        dpos = batch['decision_mask'][0].nonzero().flatten().tolist()
        prefix = tok.decode(ids[begin:apos[0]], skip_special_tokens=False)
        examples.append(dict(id=row['id'], arm=row['arm'], tensor_sha256=sha(path),
            assistant_prefix_before_action=prefix, prefix_tokens=apos[0]-begin,
            prefix_supervised_tokens=sum(int(batch['labels'][0,i] != -100) for i in range(begin, apos[0])),
            action_tokens=len(apos), decision_tokens=len(dpos)))
    dataset = read(args.study/'DATASET.json')
    training_tools = Counter(c['name'] for row in dataset['rows'] for c in row['command'])
    cells = []
    for cell in sorted((args.study/'evaluation').iterdir()):
        if not cell.is_dir():
            continue
        requests = sorted((cell/'recorded').glob('round_*/request.json'))
        rounds = []
        previous = None
        for req in requests:
            messages = read(req)
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            assert text.endswith('<|im_start|>assistant\n')
            raw_path = req.parent/'model/raw_output.txt'
            raw = raw_path.read_text()
            blocks = re.findall(r'<tool_call>(.*?)</tool_call>', raw, re.S)
            call = json.loads(blocks[0]) if len(blocks) == 1 else None
            content = messages[-1]['content']
            feedback_text = next(x['text'] for x in reversed(content) if x.get('type') == 'text' and 'CURRENT MEASURED FEEDBACK' in x.get('text', ''))
            feedback = json.loads(feedback_text.split(':\n', 1)[1])
            rounds.append(dict(round=req.parent.name, request_sha256=sha(req), output_sha256=sha(raw_path),
                call=call, explicit_think='<think>' in raw,
                assistant_history_messages=sum(m['role']=='assistant' for m in messages),
                previous_rejection=feedback.get('previous_rejection'),
                repeats_previous_call=previous is not None and previous == call))
            previous = call
        cells.append(dict(cell=cell.name, rounds=rounds))
    all_rounds = [x for c in cells for x in c['rounds']]
    assert len(examples) == 44 and len(all_rounds) == 144
    groups = {}
    for arm in ('action_only','decision_action'):
        rows = [x for x in examples if x['arm']==arm]
        groups[arm] = dict(examples=len(rows),
            empty_think_prefix_unsupervised=sum(x['assistant_prefix_before_action']=='<think>\n\n</think>\n' and x['prefix_supervised_tokens']==0 for x in rows),
            supervised_decision_prefix=sum(x['decision_tokens']>0 for x in rows))
    result = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status='completed_read_only',
        model_calls=0, training_updates=0, cuda_initialized=torch.cuda.is_initialized(),
        inputs={n:sha(args.study/n) for n in ['ENCODING.json','DATASET.json','ENCODING_PLAN.json']},
        template_sha256=sha(template_file), template_mentions_enable_thinking='enable_thinking' in template,
        training_tools=dict(training_tools), training_prefix_summary=groups, examples=examples,
        inference_summary=dict(calls=len(all_rounds), explicit_think=sum(x['explicit_think'] for x in all_rounds),
            tools=dict(Counter(x['call']['name'] for x in all_rounds if x['call'])),
            exact_consecutive_call_repeats=sum(x['repeats_previous_call'] for x in all_rounds),
            calls_with_previous_rejection=sum(x['previous_rejection'] is not None for x in all_rounds),
            assistant_history_messages=sum(x['assistant_history_messages'] for x in all_rounds),
            all_generation_prefixes_end_at_assistant_start=True), cells=cells,
        interpretation={'established':'Action loss is conditioned on a teacher-forced assistant prefix. Empty think prefixes are present but masked on ordinary examples; deployment begins before this prefix. Explicit decision generation is not demonstrated.',
            'not_established':'Neither prefix mismatch nor limited rejection memory is proven to cause the failed room outcomes. Absence of explicit think does not prove absence of internal reasoning. This audit does not measure fitting or design benefit.'})
    assert not result['cuda_initialized']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','training_tools','training_prefix_summary','inference_summary']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
