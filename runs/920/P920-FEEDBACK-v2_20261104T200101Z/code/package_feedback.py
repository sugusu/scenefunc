"""Package the two recorded continuations without editing or scoring scenes."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil

from compact_catalog_context import use_feedback


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--public',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True)
    a=ap.parse_args();p=a.public;t=a.runtime;now=datetime.now(timezone.utc).isoformat()
    protocol=read(p/'RUN_PROTOCOL.json');arms={};commands=[]
    selected={'request.json','next_request.json','initial_request.json','observation.json','RESULT.json',
        'ENCODING.json','raw_output.txt','output_ids.json','parsed_calls.json','tool_reply.json',
        'feedback_audit.json','feedback_binding.json','receipt.json','stage.log'}
    for arm in ('G','U'):
        root=t/arm;out=p/arm;summary=read(root/'SUMMARY.json');audit=read(root/'FINAL_AUDIT.json')
        rollout=read(root/'rollout/RESULT.json');execution=read(root/'rollout_execution/receipt.json')
        if execution['status']=='running':raise ValueError('Policy is still running')
        plan=read(out/'POLICY_PLAN.json')
        if sha(out/'POLICY_PLAN.json')!=protocol['arms'][arm]['policy_plan_sha256']:raise ValueError('Plan changed')
        for src,dst in [('prepare/RESULT.json','PREPARE_RESULT.json'),('rollout/RESULT.json','POLICY_RESULT.json'),
                        ('FINAL_AUDIT.json','FINAL_AUDIT.json'),('SUMMARY.json','SUMMARY.json')]:
            shutil.copy2(root/src,out/dst)
        bundle=Path(summary['last_completed_checkpoint']['directory'])
        # The native endpoint names its directory explicitly; never guess a
        # final checkpoint when a model/encoding stage failed earlier.
        shutil.copy2(bundle/'house_state.json',out/'FINAL_NATIVE_STATE.json')
        observations=sorted((root/'rollout').glob('round_*/observation.json'))
        obs=read(root/'rollout/final_observation.json') if (root/'rollout/final_observation.json').exists() else read(observations[-1] if observations else root/'prepare/observation.json')
        binding=read(Path(obs['state_binding']))
        top=next(x for x in binding['images'] if x['path'].endswith('.topdown.png'))
        if sha(Path(top['path']))!=top['sha256']:raise ValueError('Final view changed')
        shutil.copy2(top['path'],out/'final_topdown.png')
        write(out/'RESUME.json',{'bundle':str(bundle),'manifest_sha256':sha(bundle/'manifest.json'),
            'actual_step_idx':read(bundle/'session_checkpoint.json')['step_idx'],
            'scope':'Saved portable checkpoint, complete receipts, resources and instance identity history. Final file hashes checked; no extra cold restore implied.',
            'prior_tool_calls':plan['prior_tool_calls'],'new_tool_calls':rollout['tool_calls'],
            'remaining_original_lineage_calls':24-plan['prior_tool_calls']-rollout['tool_calls'],
            'continuation':'Do not automatically re-run this completed/failed arm; any further intervention requires a distinct frozen protocol.'})
        for src in sorted(root.rglob('*')):
            relative=src.relative_to(root)
            if not src.is_file() or src.name not in selected or 'house' in relative.parts or 'checkpoint' in relative.parts or 'final_checkpoint' in relative.parts:continue
            if src.stat().st_size>2**20:raise ValueError('Unexpected large log: '+str(src))
            target=out/'recorded'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
            if src.name=='receipt.json':commands.append({'receipt':str(src),'command':read(src)['command'],'exit_code':read(src).get('exit_code')})
        trajectory=[]
        for row in rollout['rounds']:
            i=row['turn'];ad=root/'rollout'/f'round_{i:03d}'/'feedback_binding.json'
            measured=read(ad)['audit'] if ad.exists() else None
            trajectory.append({'turn':i,'calls':row['calls'],'executed':'reply' in row,
                'intersections_after':measured['surface_intersection_pairs'] if measured else None,
                'roles_after':use_feedback(measured)['catalog_roles_present_not_function_pass'] if measured else None})
        roles=use_feedback(audit)['catalog_roles_present_not_function_pass']
        arms[arm]={'summary':summary,'role_coverage':roles,'trajectory':trajectory,'execution':execution,
            'final_view_binding':obs['state_binding'],'final_view_hashes':binding['images']}
        status=read(out/'STATUS.json');status.update(stage='final_independent_audit',status='completed_partial_room_not_accepted',
            updated_at=now,actual_rollout_exit_code=execution['exit_code'],latest_verifiable_artifact=str(out/'SUMMARY.json'),
            next_step='Assess paired whole-room comparison, preserve failures and unknown functions; no automatic retry or training.')
        (out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    actual={arm:read(t/arm/'rollout/round_000/request.json') for arm in arms};normalized=copy.deepcopy(actual);feedback={}
    for arm,request in normalized.items():
        before,raw=request[1]['content'][-1]['text'].split('\n\n## Independent current-state feedback\n',1)
        request[1]['content'][-1]['text']=before;feedback[arm]=json.loads(raw)
    actual_pair={'first_actual_common_messages_identical':normalized['G']==normalized['U'],
        'same_first_actual_geometry_feedback':feedback['G']['geometry']==feedback['U']['geometry'],
        'actual_first_request_hashes':{arm:sha(t/arm/'rollout/round_000/request.json') for arm in arms}}
    if not actual_pair['first_actual_common_messages_identical'] or not actual_pair['same_first_actual_geometry_feedback']:raise ValueError('Actual first inputs were not matched')
    initial_audit=read(Path(read(p/'G/POLICY_PLAN.json')['feedback_source']['path']))
    audits={'initial':initial_audit,**{arm:read(t/arm/'FINAL_AUDIT.json') for arm in arms}}
    target_lifecycle=[]
    for i,scenario in enumerate(initial_audit['access']['scenarios']):
        targets=sorted(set().union(*(set(v['access']['scenarios'][i]['targets']) for v in audits.values())))
        for target in targets:
            states={}
            for name,audit in audits.items():
                value=audit['access']['scenarios'][i]['targets'].get(target)
                states[name]='absent_object_no_approach_point' if value is None else ('reachable_under_assumptions' if value['reachable_from_entry'] else 'unreachable_under_assumptions')
            target_lifecycle.append({'square_body_width_m':scenario['square_body_width_m'],'target':target,'states':states})
    result={'run_id':p.name,'recorded_at':now,'question':'Ordinary G versus G+U feedback on the same partial development room within the nine remaining actions.',
        'arms':arms,'actual_initial_pairing':actual_pair,'shared_prior_model_calls':15,'shared_prior_tool_calls':15,
        'initial':{'audit':initial_audit,'role_coverage':use_feedback(initial_audit)['catalog_roles_present_not_function_pass']},
        'conditional_target_lifecycle':target_lifecycle,
        'new_model_calls':sum(v['summary']['actual_model_calls'] for v in arms.values()),
        'new_tool_calls':sum(v['summary']['actual_tool_calls'] for v in arms.values()),
        'training_updates':0,'qualified_professional_decisions_added':0,'independent_designer_review':False,
        'whole_room_accepted':False,'limits':['One development family and one deterministic continuation per condition, no independent sample replication.',
            'U adds information/input tokens and cost; no equal-FLOP claim. Both arms use the same compact representation.',
            'Program feedback is not professional preference. Conditional routes, inventory and empty area are separate measures.',
            'Privacy, installation, articulation, comfort and complete usable dwelling remain unverified. Ordinary feedback gains are not training or generalization gains.'],
        'package_script_sha256':sha(Path(__file__))}
    write(p/'COMPARISON.json',result)
    write(p/'COMMANDS.json',commands)
    lines=['# 同一部分房间的几何／用途反馈对照','','两组从 WHOLE-v2 的同一 step15 检查点出发，各最多再用 9 次动作；此前 15 次动作是共享成本。G 接收几何检查，U 另接收条件通行和功能清单反馈。两组均未训练。','',
        '| 项目 | G | U |','| --- | --- | --- |']
    metrics=[('新增模型调用',lambda x:x['summary']['actual_model_calls']),('新增实际工具调用',lambda x:x['summary']['actual_tool_calls']),
        ('最终家具数',lambda x:x['summary']['furniture_count']),('实际视觉网格相交对数',lambda x:len(x['summary']['intersection_pairs'])),
        ('越界家具数',lambda x:x['summary']['outside_room_count']),('原始结构记录保持',lambda x:x['summary']['immutable_structure_unchanged']),
        ('仍缺的目录功能指示项',lambda x:', '.join(k for k,v in x['role_coverage'].items() if not v) or '无；不代表功能已验收'),
        ('实际退出码',lambda x:x['execution']['exit_code']),('生成 token',lambda x:x['summary']['output_tokens'])]
    for title,fn in metrics:lines.append('| '+title+' | '+str(fn(arms['G']))+' | '+str(fn(arms['U']))+' |')
    lines+=['','完整住宅尚未验收。家具数量、朝向、空地面积或工具成功不能替代用途与隐私检查；删除所需功能造成的几何改善不记为设计收益。','',
        '共享初态有 6 件家具、2 对实际视觉网格相交。条件通行在 0.4／0.5／0.6 米方形身体假设下分别为 5/7、3/7、2/7 个接近点可达；它们不是活动成功率。', '',
        '[完整比较与逐步动作](COMPARISON.json) · [冻结协议](RUN_PROTOCOL.json) · [初始配对核验](PAIR_INPUT_AUDIT.json) · [G 结果](G/SUMMARY.json) · [U 结果](U/SUMMARY.json)','',
        'COMPARISON 的 conditional_target_lifecycle 将初态和两终态的接近点取并集逐项展示。被删除物体的接近点记为缺失，不从共同表中消失；新增目标与原目标分开识别。各终态的可达点比例因对象增删而分母不同，不能直接当作可比的使用成功率。','',
        '两组实际首请求的共同内容再次逐字核对。U 增加输入信息和 token，因此不宣称计算量完全相同；此单开发房间对照不提供统计优越性、学习或跨项目泛化证据。','',
        '原生四元数冷恢复的极小舍入逐项记录；原五视图按已有 ≤1e-12 分量差规则复用，其他状态字段完全相同。先前 v1 的精确哈希拒绝与 0 次模型结果完整保留。','',
        '原始状态、全图、网格和检查点留在服务器；路径与哈希见 SERVER_ARTIFACTS.json。实际命令及退出码见 COMMANDS.json，恢复位置见两组 RESUME.json。','',
        '![G 终态](G/final_topdown.png)','', '![U 终态](U/final_topdown.png)','']
    (p/'README.md').write_text('\n'.join(lines))
    files=[{'path':str(x),'bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(t.rglob('*')) if x.is_file()]
    write(p/'SERVER_ARTIFACTS.json',{'runtime':str(t),'files':files,'total_bytes':sum(x['bytes'] for x in files),
        'recovery':'Portable checkpoint resources on 920, pinned upstream revisions and weights in POLICY_PLAN; all actual small command receipts archived. No large assets in Git.'})
    print(json.dumps({'status':'packaged','new_model_calls':result['new_model_calls'],'new_tool_calls':result['new_tool_calls'],'server_files':len(files),'server_bytes':sum(x['bytes'] for x in files)}),flush=True)


if __name__=='__main__':main()
