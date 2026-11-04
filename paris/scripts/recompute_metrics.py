"""Recompute reported counts from preserved per-object/pair evidence; CPU stdlib."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'runs/autodl/20261104_paris');p.add_argument('--out',type=Path);a=p.parse_args()
 src=a.results/'paris_local_run_results_20261104';rows=[]
 for arm in ['A','B']:
  before=json.loads((src/f'{arm}_scene_audit.json').read_text());after=json.loads((src/f'{arm}_after_feedback_audit.json').read_text())
  r={'arm':arm,'before_objects':len(before['objects']),'after_objects':len(after['objects']),
     'before_collision_pairs':len(before['furniture_surface_intersection_pairs']),'after_collision_pairs':len(after['furniture_surface_intersection_pairs'])}
  assert r['before_objects']==before['placed_furniture'] and r['after_objects']==after['placed_furniture'];rows.append(r)
 assert [(r['before_objects'],r['after_objects'],r['before_collision_pairs'],r['after_collision_pairs']) for r in rows]==[(9,9,7,7),(10,9,7,5)]
 results=json.loads((src/'results.json').read_text());assert len(results['calls'])==8
 assert sum(r['model_status']=='completed' for r in results['calls'])==6
 retrieval=json.loads((src/'retrieval_replay.json').read_text());assert len(retrieval['rows'])==20
 out={'rows':rows,'model_completed':6,'model_failed':2,'retrieval_requests':20,'scope':'Independent arithmetic from preserved evidence; not a fresh FCL or model execution','full_scene_success':None}
 if a.out:a.out.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
if __name__=='__main__':main()
