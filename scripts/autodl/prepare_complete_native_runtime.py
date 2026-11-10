"""Small writable overlay over the immutable, already qualified source tree."""
import hashlib
import json
from pathlib import Path
import shutil


def main():
    repo=Path(__file__).resolve().parents[2]
    previous=repo/'runs/autodl/20261110_remote_native_controller'
    stage=repo/'runs/autodl/20261110_complete_native_runtime';root=stage/'runtime';root.mkdir(exist_ok=False)
    base=Path(json.loads((previous/'materialization.json').read_text())['runtime_root'])
    rows=[]
    def link(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source.resolve(),target_is_directory=source.is_dir())
        rows.append(dict(source=str(source),destination=str(dest),kind='read_only_reference'))
    for name in ('Designer','tools','src_critic'):link(base/name,root/name)
    for source in (base/'src').glob('*.py'):
        target=root/'src'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if source.name=='runtime.py':shutil.copyfile(source,target)
        else:link(source,target)
    for source in (previous/'next_stage_sources').glob('*.py'):link(source,root/'src'/source.name)
    worker=Path('/root/autodl-tmp/agent_method_preparation_20261027/work/native_observation_bridge_20261104/source/scripts/export_scene_worker.py')
    (root/'scripts').mkdir();shutil.copyfile(worker,root/'scripts'/worker.name)
    rows.append(dict(source=str(worker),destination=str(root/'scripts'/worker.name),kind='exact_copy',sha256=hashlib.sha256(worker.read_bytes()).hexdigest()))
    shutil.copytree(base/'prompts',root/'prompts')
    path=root/'prompts/floor_plan_tool';before=path.read_bytes()
    path.write_text(path.read_text().replace('  CLIP retrieval. Use specific descriptions ("light oak wood floor"), not "wood".',
        '  Local default-material lookup only: Wood094 for floor/wood descriptions; Plaster001 otherwise. Arbitrary material retrieval is unavailable.'))
    rows.append(dict(path=str(path),kind='explicit_common_material_capability_notice',before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (stage/'runtime_overlay_manifest.json').write_text(json.dumps(dict(base=str(base),native_root=str(base/'native_source'),
        references=rows,original_source_modified=False,model_calls=0),ensure_ascii=False,indent=2)+'\n')
    print(root)


if __name__=='__main__':main()
