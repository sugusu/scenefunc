"""Score fixed one-step functional predictions without rewarding persistence alone.

Predictions are requirement->bool/null maps. This is an ordinary diagnostic
metric, not a new reward or human-preference label. No model is called.
"""
import argparse
import json
from pathlib import Path


def score(rows, predictions):
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != predictions.keys():
        raise ValueError('Predictions must cover the fixed decision set exactly')
    totals = dict(decisions=len(rows), known_after=0, unknown_after=0,
                  covered_known_after=0, correct_after=0,
                  fully_observed_vectors=0, exact_vectors=0)
    events = {k:dict(tp=0,fp=0,fn=0) for k in ('repair','damage')}
    for r in rows:
        pred = predictions[r['id']]
        if pred.keys() != r['after'].keys():
            raise ValueError('Requirement keys changed: '+r['id'])
        if any(v is not None and type(v) is not bool for v in pred.values()):
            raise ValueError('Predictions must be bool or explicit null')
        fully_known = all(v is not None for v in r['after'].values())
        totals['fully_observed_vectors'] += fully_known
        totals['exact_vectors'] += fully_known and pred == r['after']
        for k, target in r['after'].items():
            before = r['before'][k]; guess=pred[k]
            if target is None:
                totals['unknown_after'] += 1
                continue
            totals['known_after'] += 1
            totals['covered_known_after'] += guess is not None
            totals['correct_after'] += guess is target
            if before is None:
                continue
            for event, old, new in [('repair',False,True),('damage',True,False)]:
                actual = before is old and target is new
                predicted = before is old and guess is new
                events[event]['tp'] += actual and predicted
                events[event]['fp'] += not actual and predicted
                events[event]['fn'] += actual and not predicted
    for event in events.values():
        tp,fp,fn=event['tp'],event['fp'],event['fn']
        event['precision']=tp/(tp+fp) if tp+fp else None
        event['recall']=tp/(tp+fn) if tp+fn else None
        event['f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    totals.update(event_metrics=events,
        micro_after_accuracy=totals['correct_after']/totals['known_after'] if totals['known_after'] else None,
        exact_vector_accuracy=totals['exact_vectors']/totals['fully_observed_vectors'] if totals['fully_observed_vectors'] else None,
        scope='Saved measured one-step proxies; unknown excluded from ground truth, abstention is not correctness. No full-room or human-preference claim.')
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transitions',type=Path,required=True)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--predictions',type=Path)
    choice.add_argument('--persistence',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[json.loads(line) for line in a.transitions.read_text().splitlines()]
    pred=({r['id']:r['before'] for r in rows} if a.persistence else json.loads(a.predictions.read_text()))
    result=score(rows,pred)
    result['predictor']='ordinary_before_state_copy' if a.persistence else str(a.predictions)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
