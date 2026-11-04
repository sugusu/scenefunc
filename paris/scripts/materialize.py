"""Create a new writable machine run from read-only portable snapshots."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--geometry-python',type=Path,required=True);ap.add_argument('--model-python',type=Path,required=True);ap.add_argument('--render-python',type=Path,required=True);a=ap.parse_args()
 run=a.run.resolve();run.mkdir(parents=True,exist_ok=False);work=run/'work';work.mkdir()
 maps={'@WORK@':str(work),'@LEGACY@/envs/hsm_cpu/bin/python':str(a.geometry_python.resolve()),'@LEGACY@/official_assets/hssd-models':str(a.data.resolve()/'hssd-models'),'@GEOMETRY_DEPS@':'','@RENDER_PYTHON@':str(a.render_python.resolve())}
 promptdir=a.data/'upstream/paris_A_inference_20261104/source/prompts'
 template=(promptdir/'furniture_system_prompt').read_text();tools=(promptdir/'furniture_tool').read_text().rstrip()
 for arm in ['A','B']:
  brief=json.loads((ROOT/f'config/static/paris_A_inference_20261104/input_{arm}.json').read_text())['brief']
  prefix=template.replace('{{ room_prompt }}',brief).replace('{{room_prompt}}',brief).replace('{{ tools }}',tools).replace('{{tools}}',tools).rstrip()
  maps[f'@NATIVE_SYSTEM_{arm}@']=json.dumps(prefix,ensure_ascii=False)[1:-1]
 sources=[]
 for base in [ROOT/'src',ROOT/'config/static',a.data/'upstream']:
  for f in sorted(base.rglob('*')):
   if not f.is_file():continue
   target=work/f.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True)
   data=f.read_bytes()
   if f.suffix in ['.py','.json','.txt']:
    text=data.decode()
    for old,new in sorted(maps.items(),key=lambda p:-len(p[0])):text=text.replace(old,new)
    # Original A retry assertion referred to a saved failed attempt. A fresh
    # machine run has no such attempt; when present, retain its exact check.
    text=text.replace("if args.attention=='sdpa' and args.arm=='A' and args.audit:","if args.attention=='sdpa' and args.arm=='A' and args.audit and (P/'A_mesh_feedback/request.json').exists():")
    target.write_text(text)
   else:target.write_bytes(data)
   sources.append({'source':str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else 'downloaded/'+str(f.relative_to(a.data)), 'source_sha256':sha(f),'runtime':str(target.relative_to(run)),'runtime_sha256':sha(target)})
 for src,dst in [(a.data/'weights',work/'critic_public_runtime_20261104/checkpoint'),(a.data/'frames',work/'paris_asset_contract_20261104/official_configs')]:
  dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src.resolve(),target_is_directory=True)
 inp=work/'paris_A_inference_20261104'
 prep={'arms':{arm:{'request_sha256':sha(inp/f'request_{arm}.json')} for arm in ['A','B']},'card':{'raster_sha256':sha(inp/'common_input_card.png')},'frozen_shared_sources':{str(inp/'frozen_initial.json'):sha(inp/'frozen_initial.json')}}
 (inp/'preparation.json').write_text(json.dumps(prep,indent=2))
 try:parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 except subprocess.CalledProcessError:parent='uncommitted'
 config={'parent_commit':parent,'work':str(work),'data':str(a.data.resolve()),'geometry_python':str(a.geometry_python.resolve()),'model_python':str(a.model_python.resolve()),'render_python':str(a.render_python.resolve()),'mapping':maps,'changes':['Absolute path relocation','Fresh-run retry comparison only if prior failed request exists'],'source_records':sources}
 (run/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
 print(json.dumps({'run':str(run),'work':str(work),'parent_commit':parent,'files':len(sources)}))
if __name__=='__main__':main()
