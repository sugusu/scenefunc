"""Exact finite-pool two-rectangle disjointness, an ordinary strong control.

For axis-aligned rectangles, interior disjointness is equivalent to one of
four endpoint separations. For each work rectangle it suffices to test the
four dining rectangles extreme in those four endpoints. This is exhaustive
over the SAME finite pool, not a proof over all continuous poses.
"""
from pathlib import Path
import json
import time
import resource
from shapely.geometry import shape
from measure_joint_use import candidates, serial, TASKS, digest, ROOT


def strong_solve(work, dining):
    if not work or not dining:
        return {'status':'not_found','finite_pool_complete':True,'comparisons':0,
                'reason':'one activity has no candidate in the fixed finite pool'}
    extremes = [min(dining, key=lambda d:d['polygon'].bounds[2]),
                max(dining, key=lambda d:d['polygon'].bounds[0]),
                min(dining, key=lambda d:d['polygon'].bounds[3]),
                max(dining, key=lambda d:d['polygon'].bounds[1])]
    count = 0
    for w in work:
        a,b,c,d = w['polygon'].bounds
        for i, other in enumerate(extremes):
            count += 1
            e,f,g,h = other['polygon'].bounds
            separated = [g <= a, e >= c, h <= b, f >= d][i]
            if separated:
                # Strict geometric validation of the actual returned witness.
                assert w['polygon'].intersection(other['polygon']).area == 0
                return {'status':'witness','finite_pool_complete':True,'comparisons':count,
                        'witnesses':{'work':serial(w),'dining':serial(other)}}
    return {'status':'not_found','finite_pool_complete':True,'comparisons':count,
            'reason':'no interior-disjoint pair in the complete finite candidate pool; continuous feasibility unknown'}


def main():
    target = ROOT/'strong_control.json'
    if target.exists():
        raise FileExistsError(target)
    start = time.perf_counter()
    raw = json.loads((ROOT/'measurements.json').read_text())
    output = []
    for row in raw['rows']:
        for mode in ['asset_surface','scene_section_free']:
            pools = {t:[] for t in TASKS}
            for surf in row['surfaces']:
                if mode == 'scene_section_free' and surf['uncertain_section_objects']:
                    continue
                polygon = shape(surf['surface' if mode=='asset_surface' else 'free_section'])
                for t, dims in TASKS.items():
                    pool = candidates(polygon,dims,surf['object_id'],surf['z_m'])
                    assert len(pool) == surf['candidate_counts'][mode][t]
                    pools[t].extend(pool)
            result = strong_solve(pools['work'],pools['dining'])
            output.append({'arm':row['arm'],'stage':row['stage'],'mode':mode,
                           'pool_sizes':{t:len(p) for t,p in pools.items()},'joint':result,
                           'bounded_enumeration_status':row['results'][mode]['simultaneous_use']['status']})
    target.write_text(json.dumps({'source_sha256':digest(ROOT/'measurements.json'),'rows':output,
        'seconds':time.perf_counter()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'ordinary exact finite-pool control; no new model or scene'},indent=2)+'\n')
    for row in output:
        print(row['arm'],row['stage'],row['mode'],row['joint']['status'],row['joint']['comparisons'])


if __name__=='__main__':
    main()
