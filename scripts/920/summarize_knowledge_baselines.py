"""Summarize completed fixed-base prompt controls without inventing design scores."""
import argparse,json,hashlib,shutil,csv
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);a=ap.parse_args();p=a.public.resolve();t=a.runtime.resolve();now=datetime.now(timezone.utc).isoformat()
    receipt=read(t/'batch_execution/receipt.json');batch=read(t/'BATCH.json');assert receipt['exit_code']==0 and all(x['completed'] for x in batch['arms'])
    rows=[];refs={}
    for arm in ['strong','knowledge']:
        folder=p/arm;rr=t/arm;report=read(folder/'REPORT.json');schema=read(folder/'ACTION_SCHEMA_REVIEW.json');restore=read(folder/'RESTORE_RESULT.json');visual=read(folder/'VISUAL_REVIEW.json')
        assert visual['views_inspected']==5 and restore['receipt_count']==report['tool_calls']==24
        for stage in ['prepare','rollout','audit','export','report','schema_review','restore','archive']:
            r=read(rr/(stage+'_execution')/'receipt.json');assert r['exit_code']==0
        row={k:report[k] for k in ['arm','actual_started_at','actual_ended_at','actual_exit_code','actual_seconds','model_calls','tool_calls','stop_reason','inventory_counts','furniture_count','triangle_contacts','outside_nominal_interior','generation_seconds','output_tokens','input_token_range','peak_allocated_bytes','failed_actions','whole_room_accepted','glb']}
        row.update(schema_mismatch_calls=schema['schema_mismatch_calls'],action_counts=schema['counts'],unchanged_immutable_records=all(report['immutable_records_unchanged'].values()),native_restore_passed=True,
            absent_catalog_roles_by_requirement={x['id']:[k for k,v in x.get('available_catalog_role_counts',{}).items() if v==0] for x in report['requirements']},
            access_warnings={x['id']:x['proxy_results'] for x in report['requirements'] if 'proxy_results' in x},requirement_statuses={x['id']:x['overall_status'] for x in report['requirements']})
        rows.append(row)
        for name in ['REPORT.json','ACTION_SCHEMA_REVIEW.json','RESTORE_RESULT.json','VISUAL_REVIEW.json','INTERPRETATION.md']:refs[arm+'/'+name]=sha(folder/name)
        for f in (rr/'archive_execution').iterdir():
            dest=folder/'execution/archive'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        cp=rr/'rollout/final_checkpoint';exp=read(folder/'EXPORT_RESULT.json')
        (folder/'RECOVERY.md').write_text(f"# 恢复\n\n可编辑原生检查点 `{cp}`；manifest SHA `{sha(cp/'manifest.json')}`。完整资产留服务器，Git checkpoint_metadata 只是索引与小状态，恢复须复制manifest所有资源并校验。\n\nGLB `{exp['glb']['path']}`，{exp['glb']['bytes']}字节，SHA `{exp['glb']['sha256']}`。原生新进程冷加载已通过，对象/资产/24回执保留，恢复后新编辑未测。命令见 execution/restore/receipt.json 和 RESTORE_PLAN.json；不要重跑已完成推理。\n")
        write(folder/'EVIDENCE_MANIFEST.json',{'files':[{'path':str(f.relative_to(folder)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='EVIDENCE_MANIFEST.json']})
    summary={'run_id':p.name,'summarized_at':now,'status':'two_development_baselines_completed_and_reviewed','arms':rows,'batch_started_at':receipt['started_at'],'batch_ended_at':receipt['ended_at'],'batch_exit_code':0,'batch_elapsed_seconds':receipt['elapsed_seconds'],'new_model_calls':48,'new_tool_attempts':48,'new_training_updates':0,'source_sha256':refs,'preparation_review_sha256':sha(p/'PREPARATION_REVIEW.json'),'interpretation':'Fixed complete source memory changed the observed trajectory, but neither condition completed the dwelling. No professional quality improvement, training need, pairing benefit or generalization established.','limits':['One exposed development project, one deterministic attempt per condition; no statistical population claim.','Both new arms use input cap 16384; historical base/adapter arms used 12288 and different prompts.','Knowledge is all ten prior pilot reason texts, including researcher action descriptions; no retrieval/ranking algorithm.','A no-furniture scene has no furniture contacts; fewer contacts cannot offset missing required functions.','Directions/facing are not privacy, articulation, installation or real activity validation.']}
    write(p/'COMPARISON.json',summary)
    with (p/'COMPARISON.csv').open('w',newline='') as f:
        keys=['arm','model_calls','tool_calls','furniture_count','schema_mismatch_calls','actual_seconds','output_tokens'];w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows({k:x[k] for k in keys} for x in rows)
    lines=['# 强提示与同知识开发对照：完整结果','','两组使用同一冻结底座、房壳、目录、五图、greedy和24次动作预算。strong提供完整任务规划提示；knowledge另给原试运行三来源全部10段理由。固定完整小库，不是检索算法。','','| 条件 | 模型/工具 | 末态家具 | 查询schema异常 | 实际秒 |','|---|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['arm']} | {x['model_calls']}/{x['tool_calls']} | {x['furniture_count']} | {x['schema_mismatch_calls']} | {x['actual_seconds']:.3f} |")
    lines+=['','strong全部24调用为查询，未生成家具。knowledge末态仅床与淋浴，缺完整厨房、卫生、收纳、工作与来访用餐；两次炉灶删除后最后一次非目录ID添加被拒。两组均0检测三角接触/越界，但不能当作整屋设计通过。','', 'knowledge的床右侧/淋浴前侧在三个宽度假设下不可达，卫浴内部点在0.5/0.6米失败；露台点通过。实际终态5图每组已审阅，新进程恢复与身份/回执校验通过，仅证明状态可恢复。','', '完整批次2865.325976秒、退出0；两组均预算用尽。详见各组REPORT、INTERPRETATION、ACTION_SCHEMA_REVIEW、RECOVERY与原始回执。','', '**结论范围：** 加入相同历史理由改变了本例的动作轨迹；没有证据表明完成了房间、提高了专业品质或证明训练必要。Casa早已暴露并用于开发，两组各一次，不是跨项目或统计结论。旧base/separate/paired使用不同提示与12288输入上限，本轮共同16384，旧结果仅作描述性背景。','', '新增FAMILYDATA三个家族七知识不在本轮输入中；不得把新资料算成已运行对照的知识。后续要把可执行建设与有来源子决定对应起来，优先保证任务覆盖和对照可解释性，再提出新训练方案。']
    (p/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(t/'BATCH.json',p/'BATCH.json')
    for name in ['receipt.json','stage.log']:
        dest=p/'execution/batch_execution'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(t/'batch_execution'/name,dest)
    status=read(p/'STATUS.json');status.update(status=summary['status'],updated_at=now,actual_exit_code=0,started_at=receipt['started_at'],ended_at=receipt['ended_at'],first_log=receipt['first_log'],completed_planned={'arms':[2,2],'model_calls':[48,48],'tool_attempts':[48,48]},latest_verifiable_artifact=str(p/'COMPARISON.json'),next_step='Publish complete evidence; expand sourced executable construction and matched task controls before requesting any new training.')
    write(p/'STATUS.json',status);write(p.parent/'latest.json',{'run_id':p.name,'stage':summary['status'],'updated_at':now,'run_directory':str(t)})
    print(json.dumps({'status':summary['status'],'furniture':[x['furniture_count'] for x in rows],'schema_mismatch':[x['schema_mismatch_calls'] for x in rows],'batch_exit_code':0}))

if __name__=='__main__':main()
