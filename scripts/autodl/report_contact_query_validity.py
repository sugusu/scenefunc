"""Consolidate executed native diagnostics without new engine/model calls."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main(repo, stage):
    def read(name):
        return json.loads((stage/name).read_text())
    pair = read('two_body_v1/result.json')
    point = read('same_room_control_v1/result.json')
    guarded = read('same_room_veto_v1/result.json')
    versions = read('asset_version_store_v2/result.json')
    history = read('historical_asset_versions_v2/result.json')
    bounds = read('bound_certificates_v1/result.json')
    wall_order = read('asset_version_store_v2/wall_order_diagnosis.json')
    receipts = read('same_room_veto_v1/backend_receipts.json')
    if isinstance(receipts, dict):
        raise ValueError('Inspect changed receipt schema before summarizing')
    assert sum(v['veto'] for v in receipts) == 35
    assert guarded['original_reported_pairs'] == 3 and guarded['alternative_reported_pairs'] == 1
    assert guarded['alternative_report'][0] == guarded['original_report'][-1]
    assert history['known_scale_version_conflicts'] == 15
    assert versions['native_scene_registry_roundtrip_all_exact'] is False
    assert all(v['state_equal_after_indexing_walls_by_ID'] for v in wall_order['results'])
    previous = json.loads((repo/'runs/autodl/20261111_reasoning_identifiability/source_before_after.json').read_text())
    checks = []
    for row in previous['original_inputs']:
        actual = sha(Path(row['path']))
        checks.append(dict(path=row['path'], before=row['after'], after=actual, same=actual == row['after']))
    assert len(checks) == 579 and all(x['same'] for x in checks)
    dump(stage/'source_before_after.json', dict(original_inputs=checks))
    theory = '''本轮的结论和边界

一、固定真实几何揭示了哪一类错误
固定先前证书中的第一个实际床部件，保持原始网格和姿态来源，在两物体诊断副本中检查五个明确的竖直偏移。
Mesh和Convex两种表示均复现：精确接触时signed distance约-0.965米；向上或向下0.1毫米时返回相应间距或穿透。
这是所安装Drake1.47.0查询路径上的零接触边界异常。没有审查数值求解器内核，不宣称已定位FCL内部根因或所有Drake查询均错误。
这些偏移仅发生在显式诊断副本，原始整场姿态和旧得分均未改变。

二、为什么没有直接更换为point penetration
同一原始整场中，point查询使原生报告3对变1对，但剩余北墙—床深度从约0.046米变成约0.455米。
原生API的近接触过滤和精度有其限定。不能把“少报两对”当作数值等价，所以不把该替换接入奖励基线。

三、共同基线采用何种有限修正
设真实形状被世界坐标AABB A=[a-,a+]、B=[b-,b+]包围。取
U=max(0,min_i{a+_i-b-_i,b+_i-a-_i})。
沿选中的有向坐标轴平移U加任意正的小量，可以使包围盒分离，因而也使其中形状分离。
故U是最小平移穿透量的上界，而不是碰撞下界；U>0不能证明形状相交。包含关系不能误用“交集宽度”。
实际实现逐次浮点乘加均向外舍入，对已表示的有限顶点和变换构造包围盒；无法构造的类型保留原报告并标未知。
仅在上界严格小于原有1毫米报告阈值时剔除候选。等于阈值保留，原有地板5厘米和支撑2厘米容差未修改。
同一整场53条原始候选中剔除35条，最终报告3对变1对；剩余墙—床记录和原深度逐字段一致。
固定二物体的真实2毫米穿透保留。该修正不计算精确距离、不保证其余报告正确，也不证明整场任务成功。

四、为什么历史JSON不能直接用于分支学习
10个历史家具状态共36个对象引用，其中15个逻辑缩放与最终存档文件版本冲突。
初版把没有显式mesh scale的衣柜误算为冲突，得到24；该初版保留并明确作废，正确版本为15。
早期SDF字节没有保存就不能反推成“真实历史文件”，不能用最终SDF替代历史状态来重算奖励或制造偏好。
新版本存储只是普通内容寻址文件记录。两个实际原生缩放调用后，中间和父版本的SDF/注册表字节、所测资产逻辑缩放能恢复。
原生反序列化会按objects字典重建walls列表：按唯一wall ID比较后内容相同，但原始JSON列表顺序不同。
重复读取同一父版本一致。仍不宣称所有顺序消费者、渲染缓存、工作进程或随机状态均已恢复。

五、对学习机制的影响
这些属于所有方法共享的执行与奖励基线，不是append-only新算法，不验证记忆胜过状态表。
下一次生成必须使用前瞻文件版本记录，不能把旧不完整状态包装成可重放的反事实分支。
现有A/AUX/J/P及普通反馈F、完整轨迹T的比较仍需要真实同条件完整生成；本轮没有新增整场正样本、模型前向、梯度或训练。
决策监督的作用不能从流畅理由、teacher-forced损失或工程修正后的碰撞数推断。没有证据宣布CVPR级创新成立。
'''
    (stage/'algorithm_and_limits.txt').write_text(theory)
    reading = [
        dict(source='Installed Drake1.47.0 QueryObject.ComputePointPairPenetration documentation',
             URL='https://github.com/RobotLocomotion/drake/blob/v1.47.0/geometry/query_object.h',
             scope='Full installed method docstring including support tables; version and documentation hash recorded. Numerical solver kernel not read.',
             consequence='Do not equate point penetration and signed distance feedback without numerical qualification.'),
        dict(source='Pinned SceneSmith native source', revision='e888297a8b0a2f3acbb8634edf9def59cceab836',
             URL='https://github.com/nepfaff/scenesmith/tree/e888297a8b0a2f3acbb8634edf9def59cceab836',
             scope='compute_scene_collisions body and query/filter/dedup path; rescale_object_common and registry/file side effects; RoomScene serialization/restoration and SDF directive-loading paths. This is targeted source reading, not a new full-source claim.',
             consequence='Keep source and thresholds unchanged; ordinary runtime dependency injection and file versions are baseline engineering.'),
        dict(source='Scene-Designer', URL='https://arxiv.org/html/2610.23121v1',
             scope='This continuation reconfirmed main method3.1/3.2 and training setup; earlier full reading recorded separately. No new author model reproduction.',
             consequence='Reason-action supervision, branch ranking and parent-child critique already exist. Reliable replay is required for our control; improving it is not a distinct learned contribution.'),
        dict(source='ThinkBLOX', URL='https://arxiv.org/html/2607.13539v1',
             scope='This continuation reconfirmed III-A/III-B: progressive CoT/action, consistency rewards and tier-normalized advantages. No author training or original dataset acquired.',
             consequence='Reason-then-place and reason-action consistency cannot serve as new-method claims by themselves.')]
    dump(stage/'reading_ledger.json', reading)
    rows = [
        ('two_body_engine_queries',20,'controlled copies, fixed actual component'),
        ('two_body_seconds',pair['seconds'],'CPU diagnostic, not generation speed'),
        ('point_replacement_seconds',point['seconds'],'replacement NOT admitted as equivalent reward'),
        ('guarded_report_seconds',guarded['seconds'],'same original whole-room state'),
        ('guarded_raw_candidates_removed',35,'original53; unchanged1mm threshold'),
        ('whole_room_reported_pairs_before',3,'original native reporter'),
        ('whole_room_reported_pairs_after',1,'retained original wall-bed record exactly'),
        ('historical_snapshots',10,'no historical replay or reward recomputation'),
        ('historical_asset_refs',36,'15knownscale-versionconflicts'),
        ('historical_known_scale_conflicts',15,'v1count24superseded; absentexplicitScaleisnotconflict'),
        ('actual_version_helper_calls',4,'twoinitialcalls+twoboundeddiagnosticcalls; no automatic retry'),
        ('v2_stored_asset_versions',3,'SDF/registry bytes + native SceneJSON subset'),
        ('native_JSON_bitwise_roundtrip',False,'wall order changes; unique-ID contents agree'),
        ('protected_source_hashes_unchanged',len(checks),'integrity, not scene quality'),
        ('new_model_forwards',0,'no loaded policy/model'),
        ('training_updates',0,'none'),('paid_API_calls',0,'none'),
        ('new_verified_full_rooms',0,'partial bedroom remains incomplete')]
    with (stage/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','value','qualification']);w.writerows(rows)
    commands = '''All outputs must use fresh directories; do not run into original result paths.
Native Python and PYTHONPATH must come from runs/autodl/20261110_complete_native_runtime/ENVIRONMENT.json.
No entrypoint below loads a model, runs training, downloads assets or calls a paid API.

python scripts/autodl/probe_actual_contact_queries.py --help
python scripts/autodl/compare_native_contact_feedback.py --help
python scripts/autodl/check_contact_bound_certificates.py --help
python scripts/autodl/audit_archived_asset_versions.py --help
python scripts/autodl/check_native_asset_versions.py --help

Executed contact report controls used --backend point and --backend conservative respectively.
The version storage v2 evidence continuation used --out-name asset_version_store_v2 with the same two0.9 rescalings.
The first failed evidence run remains asset_version_store_v1/result.json; do not erase or present it as success.
Report consolidation (no simulator/model call):
python scripts/autodl/report_contact_query_validity.py --repo REPOSITORY --stage STAGE

Stored version records include the original owned scratch root. Direct cross-root restoration is intentionally refused.
To reproduce runtime versions in a new root, restore original solid evidence first and rerun the qualified helper probe, not blindly overwrite paths.
'''
    (stage/'reproduce.txt').write_text(commands)
    links=['algorithm_and_limits.txt','reading_ledger.json','summary.csv',
        'two_body_v1/result.json','two_body_v1/query_records.json',
        'same_room_control_v1/result.json','point_backend_depth_check.json',
        'same_room_veto_v1/result.json','same_room_veto_v1/backend_receipts.json',
        'bound_certificates_v1/result.json','historical_asset_versions_v1/STATUS.json',
        'historical_asset_versions_v2/result.json','asset_version_store_v1/result.json',
        'asset_version_store_v2/result.json','asset_version_store_v2/wall_order_diagnosis.json',
        'source_before_after.json','reproduce.txt','evidence.zip']
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)
    (stage/'stage_report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生反馈与分支证据资格</title><style>body{max-width:1050px;margin:40px auto;padding:20px;font:16px/1.7 system-ui}pre{white-space:pre-wrap}td,th{border:1px solid #aaa;padding:7px}table{border-collapse:collapse}a{overflow-wrap:anywhere}</style><h1>真实工具反馈与历史版本：已执行结果</h1><p>保守共同修正排除已证明达不到原阈值的候选，保留墙—床真实待处理报告。新文件版本能恢复所测字节，但不是完整模拟器快照。本轮没有新增完整生成或学习效果，旧结果均保留。</p><table><tr><th>指标</th><th>值</th><th>适用范围</th></tr>'+table+'</table><h2>推导与研究影响</h2><pre>'+html.escape(theory)+'</pre><h2>证据和复现</h2><ul>'+''.join('<li><a href="'+x+'">'+x+'</a></li>' for x in links)+'</ul></html>')
    scripts=['probe_actual_contact_queries.py','native_penetration_query_adapter.py',
             'compare_native_contact_feedback.py','conservative_contact_query.py',
             'check_contact_bound_certificates.py','audit_archived_asset_versions.py',
             'native_asset_version_store.py','check_native_asset_versions.py',
             'report_contact_query_validity.py']
    files=[p for p in sorted(stage.rglob('*')) if p.is_file() and p.name not in ('evidence.zip','delivery_verification.json')]
    bundle=stage/'evidence.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(stage)))
        for name in scripts:z.write(repo/'scripts/autodl'/name,'code/'+name)
    dump(stage/'delivery_verification.json',dict(
        files={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
        scripts={name:sha(repo/'scripts/autodl'/name) for name in scripts},
        evidence_zip=dict(bytes=bundle.stat().st_size,sha256=sha(bundle)),
        source_inputs_unchanged=len(checks),data_free_GiB=shutil.disk_usage(stage).free/2**30,
        goal_achieved=False,model_forwards=0,training_updates=0,paid_API_calls=0))
    print(json.dumps(dict(stage=str(stage),report=str(stage/'stage_report.html'),
                         protected_inputs=len(checks),evidence_bytes=bundle.stat().st_size)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();main(a.repo.resolve(),a.stage.resolve())
