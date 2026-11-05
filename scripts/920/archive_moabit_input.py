"""Archive initial-condition evidence without treating it as a furnished outcome."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from native_project import sha256
from native_checkpoint import canonical_hash


def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    batch=read(t/'BATCH.json');assert batch['status']=='native_inputs_and_nominal_views_complete_pending_review'
    receipt=read(t/'batch_execution/receipt.json');assert receipt['exit_code']==0
    now=datetime.now(timezone.utc).isoformat();copies=[]
    def cp(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha256(src)==sha256(dst);copies.append(dict(path=str(dst.relative_to(p)),sha256=sha256(src)))
    variants={}
    for label,info in batch['variants'].items():
        bundle=Path(info['build']['bundle']);manifest=read(bundle/'manifest.json')
        assert sha256(bundle/'manifest.json')==info['build']['manifest_sha256']
        for rel,ref in manifest['files'].items():assert sha256(bundle/rel)==ref['sha256'],rel
        for name in ['manifest.json','house_state.json','runtime.json','task.json','catalog.json','provenance.json']:
            cp(bundle/name,p/label/'native_metadata'/name)
        cp(bundle/('room_'+manifest['selected_room_id'])/'room.sdf',p/label/'ROOM_GEOMETRY.sdf')
        cp(t/label/'build/RESULT.json',p/label/'BUILD_RESULT.json');cp(t/label/'GEOMETRY.json',p/label/'GEOMETRY.json')
        variants[label]=dict(bundle=str(bundle),manifest_sha256=sha256(bundle/'manifest.json'),
            build_seconds=read(t/(label+'_build_execution')/'receipt.json')['elapsed_seconds'],
            geometry=info['geometry'],bundle_bytes=info['build']['bundle_bytes'])
    obs=read(t/'nominal/observation/observation.json');binding=read(Path(obs['state_binding']))
    assert canonical_hash(binding['native_state'])==binding['state_sha256']
    from PIL import Image
    for q in binding['images']:
        path=Path(q['path']);assert sha256(path)==q['sha256']
        with Image.open(path) as im:im.verify()
        cp(path,p/'nominal/views'/path.name)
    cp(Path(obs['state_binding']),p/'nominal/STATE_BINDING.json')
    for name in ['RESULT.json','observation.json','initial_state.json','current_context.json']:
        cp(t/'nominal/observation'/name,p/'nominal/observation'/name)
    assert read(t/'nominal/observation/RESULT.json')['furniture_instances']==0
    for q in t.glob('*_execution/receipt.json'):
        value=read(q)
        if value['status']=='running':
            assert any(Path(x).name==Path(__file__).name for x in value['command']);continue
        assert value['exit_code']==0
        for name in ['receipt.json','stage.log']:cp(q.parent/name,p/'execution'/q.parent.name/name)
    # Check the exact executed training manifests and two live knowledge pools,
    # rather than inferring isolation from a registry label alone.
    family='berlin_moabit_bagna_coss_2015';isolation=[]
    for rel in ['runs/920/P920-LOSS-v1_20261105T092125Z/PLAN.json','runs/920/P920-CURATE-v1_20261105T212407Z/DATASET.json',
                'runs/920/P920-CURATE-v1_20261105T212407Z/KNOWLEDGE_CONTROL_POOL.json','research/920/decision_packets_v1/SOURCED_DECISIONS.json']:
        q=repo/rel;assert family not in q.read_text() and 'Moabit' not in q.read_text()
        isolation.append(dict(path=rel,sha256=sha256(q),target_family_absent=True))
    exposure=dict(recorded_at=now,family_id=family,checked_inputs=isolation,allocation='test_source_candidate',
        no_policy_output_generated_in_this_batch=True,analyst_saw_source_final_text=True,sealed_test=False,
        limits='Only enumerated local corpora checked; no assertion about base-model pretraining or every external corpus. Source facts and views alone are not test success.')
    write(p/'EXPOSURE_AUDIT.json',exposure)
    write(p/'SUMMARY.json',dict(status='heldout_transfer_inputs_prepared_not_evaluation_ready',run_id=a.run,updated_at=now,
        variants=variants,source_families=1,geometry_variants=3,nominal_views=5,model_calls=0,training_updates=0,
        new_furnished_outputs=0,formal_evaluation_ready=False,batch_receipt=receipt,
        next_step='Freeze feasible full-room requirements and evaluator before any model comparison; keep this family out of training/retrieval.'))
    write(p/'ARCHIVE_CHECKS.json',dict(recorded_at=now,copies=copies,all_three_native_manifests_verified=True,
        nominal_image_hash_decode_checks=5,nominal_state_binding_verified=True,full_meshes_copied_to_git=False))
    (p/'RECOVERY.md').write_text('# 初态恢复\n\n'+ '\n'.join(f"- {k}: `{v['bundle']}`, manifest SHA256 `{v['manifest_sha256']}`。" for k,v in variants.items())+
        '\n\n每个服务器bundle包含完整原生状态、房壳SDF和原目录资产依赖，已在构建后同进程恢复并核全部结构ID/位姿。Git只保存原生元数据、SDF、源资产清单哈希与名义五图。资产来源由catalog.json与manifest追溯到已有固定目录，没有新权重或大型网格上传。原生导入可复用native_project.import_project，不能把这些空壳恢复当成带家具功能验收。窄/宽变体未渲染，任何未来模型执行前须单独获取并核对各自观察。\n')
    cp(Path(__file__),p/'code'/Path(__file__).name)
    print(json.dumps(dict(status='input_evidence_archived',variants=3,views=5,model_calls=0)))


if __name__=='__main__':main()
