"""Portable dispatcher for one bounded stage; all stages refuse overwrite."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['A','B'],default='A');p.add_argument('stage',choices=['preflight','first','correct-retrieval','render-empty','placement','audit-placement','render-placement','feedback','audit-feedback','render-feedback','measure']);a=p.parse_args()
 cfg=json.loads((a.run/'run_config.json').read_text());w=Path(cfg['work']);initial=w/'paris_A_inference_20261104';local=w/'paris_local_run_results_20261104';local.mkdir(exist_ok=True)
 arm=a.arm;geo=cfg['geometry_python'];model=cfg['model_python'];render=cfg['render_python'];s=a.stage
 first=initial/f'gpu_{arm}_first';corrected=local/f'corrected_{arm}_assets';placement=local/f'{arm}_feedback_turn';feedback=local/f'{arm}_mesh_feedback_sdpa_kv'
 if s in ['preflight','first']:cmd=[model,str(initial/'run.py'),'--phase','preflight' if s=='preflight' else 'generate','--arm',arm,'--output',str(initial/(f'preflight_{arm}' if s=='preflight' else f'gpu_{arm}_first'))]
 elif s=='correct-retrieval':cmd=[geo,str(local/'dispatch_corrected.py'),'--calls',str(first/'parsed_tool_calls.json'),'--output',str(corrected)]
 elif s.startswith('render-'):
  source={'render-empty':corrected/'partial_scene.glb','render-placement':placement/'tools/partial_scene.glb','render-feedback':feedback/'tools/partial_scene.glb'}[s]
  dest={'render-empty':local/f'{arm}_empty_views','render-placement':local/f'{arm}_placed_views','render-feedback':local/f'{arm}_after_feedback_views'}[s]
  cmd=[render,str(local/'render_state.py'),'--scene',str(source),'--out',str(dest),'--transient-blend']
 elif s in ['placement','feedback']:
  ck=corrected/'checkpoint.json' if s=='placement' else placement/'tools/checkpoint.json';views=local/f'{arm}_empty_views/receipt.json' if s=='placement' else local/f'{arm}_placed_views/receipt.json'
  cmd=[model,str(local/'continue_one_turn.py'),'--arm',arm,'--checkpoint',str(ck),'--views',str(views),'--out',str(placement if s=='placement' else feedback),'--attention','eager' if s=='placement' else 'sdpa']
  if s=='feedback':cmd+=['--audit',str(local/f'{arm}_scene_audit.json')]
 elif s.startswith('audit-'):
  b=placement if s=='audit-placement' else feedback;dest=local/f'{arm}_scene_audit.json' if s=='audit-placement' else local/f'{arm}_after_feedback_audit.json'
  cmd=[geo,str(local/'audit_scene.py'),'--checkpoint',str(b/'tools/checkpoint.json'),'--out',str(dest)]
 else:cmd=[geo,str(w/'joint_use_witness_20261104/measure_joint_use.py'),'--out',str(w/'joint_use_witness_20261104/replayed_measurements.json')]
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 logs=a.run/'execution';logs.mkdir(exist_ok=True);tag=logs/(arm+'_'+s+'.json')
 if tag.exists():raise SystemExit('Stage receipt exists; create a new run instead of overwriting')
 result=subprocess.run(cmd,env=env)
 tag.write_text(json.dumps({'command':cmd,'exit_code':result.returncode,'parent_commit':cfg['parent_commit'],'stage':s,'arm':arm},indent=2)+'\n')
 raise SystemExit(result.returncode)
if __name__=='__main__':main()
