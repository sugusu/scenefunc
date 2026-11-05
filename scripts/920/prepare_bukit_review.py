"""Derive a non-professional, tool-grounded review record from Bukit v2 outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from decision_state import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--run',required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();rt=root/'runtime'/a.run
    load=lambda path:json.loads(path.read_text())
    ref=lambda path:{'path':str(path.resolve().relative_to(root)),'sha256':sha(path)}
    approaches=load(rt/'APPROACHES.json');candidates=[];measurements={};evidence=[]
    for cid in ('initial','after'):
        obs=load(rt/'replay'/f'{cid}_observation.json')
        candidates.append({'candidate_id':cid,'state':ref(rt/'replay'/f'{cid}_state.json'),
                           'observation':ref(rt/'replay'/f'{cid}_observation.json'),
                           'context':ref(rt/'replay'/f'{cid}_context.json'),
                           'binding':ref(Path(obs['state_binding']))})
        m=load(rt/'replay'/f'{cid}_measurement.json')
        measurements[cid]={'reserved_floor':m['reserved_floor'],
                          'fixed_targets':[{'body_width_m':s['square_body_width_m'],
                              'reachable':[k for k,v in s['targets'].items() if v['reachable_from_entry']],
                              'unreachable':[k for k,v in s['targets'].items() if not v['reachable_from_entry']]}
                              for s in m['scenarios']],
                          'object_approaches':[{'body_width_m':s['body_width_m'],
                              'groups':{k:{'reachable':v['reachable_samples'],'sampled':v['total_samples']}
                                        for k,v in s['groups'].items()}}
                              for s in approaches['states']['v2_'+cid]]}
        evidence.append({'evidence_id':cid+'_measurement','artifact':ref(rt/'replay'/f'{cid}_measurement.json')})
    evidence.append({'evidence_id':'approaches','artifact':ref(rt/'APPROACHES.json')})
    before=measurements['initial']['reserved_floor']['unoccupied_AABB_m2'];after=measurements['after']['reserved_floor']['unoccupied_AABB_m2']
    # These expected comparisons are grounded in the actual saved tool results.
    for i in range(3):
        b=measurements['initial']['object_approaches'][i]['groups'];e=measurements['after']['object_approaches'][i]['groups']
        assert b['table']['reachable']==8 and e['table']['reachable']==16
        assert b['sofa_front']['reachable']==e['sofa_front']['reachable']==8
        assert e['bookcase']['reachable']==e['window_right']['reachable']==0
        assert b['bookcase']['reachable']==[2,1,0][i] and b['window_right']['reachable']==[8,8,6][i]
        assert b['window_left']['reachable']==e['window_left']['reachable']==8
    assert [len(x['reachable']) for x in measurements['initial']['fixed_targets']]==[6,5,3]
    assert [len(x['reachable']) for x in measurements['after']['fixed_targets']]==[4,3,3]
    claims=[
        {'candidate_id':'after','activity':'为后续活动预留地面', 'object_ids':['recon_sofa_0','recon_folded_bed_cabinet_0'],
         'assessment':f'在声明的包围盒模型下，预留区未占面积由 {before:.6f} 增至 {after:.6f} 平方米，增加 {after-before:.6f} 平方米。没有实际展开床，不能据此判断睡眠可用。',
         'evidence_ids':['initial_measurement','after_measurement']},
        {'candidate_id':'after','activity':'接近桌边与沙发声明正面', 'object_ids':['recon_table_0','recon_sofa_0'],
         'assessment':'三种方形身体宽度下，桌边可达采样位置均由 8/32 增至 16/32；按声明的局部 +Y 正面，沙发前方仍为 8/8。这些是路径代理量，不是可用座位数、舒适度或经实测确认的家具正面。',
         'evidence_ids':['approaches']},
        {'candidate_id':'after','activity':'接近书架和窗前', 'object_ids':['recon_bookcase_0','recon_north_window_left','recon_north_window_right','recon_sofa_0'],
         'assessment':'书架在 0.4/0.5 米宽度下由 2/8、1/8 降至 0/8，0.6 米原本也是 0/8；右窗由 8/8、8/8、6/8 降至全部 0/8。左窗仍为 8/8。不能将这些局部损失写成所有窗户不可达，也不能把右窗损失略去。',
         'evidence_ids':['approaches']},
        {'candidate_id':'after','activity':'按原固定诊断点检查通行', 'object_ids':['recon_sofa_0','recon_bookcase_0','recon_table_0'],
         'assessment':'原先冻结的八个固定点可达数从 6/5/3 变为 4/3/3；这些坐标不随家具移动。新区域指标必须与其并列，不能覆盖旧指标后宣称整体功能改善。',
         'evidence_ids':['initial_measurement','after_measurement']}
    ]
    record={'review_id':a.run+'_objective_consequences','created_at':datetime.now(timezone.utc).isoformat(),
            'provenance':{'kind':'objective_geometric_review_of_source_guided_reconstruction',
                          'family_id':'singapore_bukit_batok_interim_spaces','allocation':'development',
                          'unique_review_unit':1,'professional_review':False,'training_eligible':False,
                          'qualification':'objective-only; training requires separate review and authorization'},
            'brief':'研究重建任务：在声明的 5.2×3.0 米主房间中移动沙发与两张凳子，暂时释放西侧预留空间；保留对象实例，报告对其他活动的影响。此任务及实例约束由研究者设定，并非住户原话或保留旧家具的原始要求。',
            'source_context':{'url':'https://www.nevertoosmall.com/post/flexible-home-for-a-singapore-social-worker',
                              'supported':'轻型沙发和可移动家具支持活动变化。具体初态、尺寸、替代资产、移动坐标及实例约束均为重建。'},
            'question':'结合各候选的五视图与实际量测回执，指出释放了什么、损失了什么，并判断现有证据是否足以给出整屋专业偏好。按 initial/after 稳定 ID 引用候选，不把图像顺序当标签。',
            'assumptions':['这里只重建主房间，厨房、卫浴和完整住宅不在范围内。',
                           '0.02 米路径网格；方形身体宽度 0.4/0.5/0.6 米，高度 1.8 米；门扇假定全开。',
                           '障碍使用包围盒，接近区域按对象声明方向采样；沙发局部 +Y 是量测假设，并非实测朝向验收。',
                           '三座沙发替代来源中的轻型双人沙发，重量、人的搬运过程、床折叠、柜门操作与座位舒适度未知。',
                           '可达采样位置不是参与者、座位数或独立试验。数值来自工具，不要求从图像估计。'],
            'candidates':candidates,'measurement_input':measurements,
            'judgment':{'verdict':'insufficient_evidence','ranking_groups':[],'claims':claims,'evidence':evidence,
                        'unknowns':['无独立专业人员对本候选的整体评价。','没有实测原房尺寸、床展开与搬运验证，不能确认完整日常活动可用。',
                                    '局部空间增加伴随接近损失，现有工具数据不能决定住户对这些取舍的偏好。']}}
    with a.out.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'record':str(a.out),'sha256':sha(a.out),'unique_objective_reviews':1,'professional_reviews':0}),flush=True)


if __name__=='__main__':main()
