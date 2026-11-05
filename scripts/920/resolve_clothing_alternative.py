"""Resolve only the prospectively selected closed-wardrobe mesh and frame."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    spec=json.loads(a.plan.read_text());aid=spec['asset_id'];assert not a.out.exists()
    print(datetime.now(timezone.utc).isoformat(),'resolve one selected wardrobe; metadata only',flush=True)
    api=HfApi(token=True);files=[]
    for repo,rev,suffix,role in [('hssd/hssd-models','208ea6bc22d863bbc1cbaef1829b3e646600dfb6','.glb','mesh'),('hssd/hssd-hab','4369cb9876214c7fbebcf552eb532380e4d287e4','.object_config.json','coordinate_frame')]:
        path=f'objects/{aid[0]}/{aid}{suffix}'
        try:rows=api.get_paths_info(repo,[path],repo_type='dataset',revision=rev)
        except Exception as exc:
            print('metadata failed',type(exc).__name__,getattr(getattr(exc,'response',None),'status_code',None),flush=True)
            raise SystemExit(1) from None
        assert len(rows)==1 and rows[0].path==path
        x=rows[0];lfs=getattr(x,'lfs',None)
        files.append(dict(repository=repo,revision=rev,path=path,bytes=x.size,asset_id=aid,asset_name=spec['asset_name'],role=role,**({'sha256':lfs.sha256} if lfs else {'git_blob':x.blob_id})))
    assert len(files)==spec['budget']['download_files']==2
    assert sum(x['bytes'] for x in files)<=spec['budget']['maximum_download_bytes']
    r=Path(__file__).resolve().parents[2]
    out=dict(resolved_at=datetime.now(timezone.utc).isoformat(),files=files,cache_directory=str(r.parent/'resources/research_support/decision_asset_cache'),download_script_sha256=hashlib.sha256((r/'scripts/920/fetch_asset_bundle.py').read_bytes()).hexdigest(),budget=dict(files=2,max_bytes=spec['budget']['maximum_download_bytes']),selection_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest())
    a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=2,bytes=sum(x['bytes'] for x in files),mesh_downloaded=False)),flush=True)


if __name__=='__main__':main()
