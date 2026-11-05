"""Propose leakage groups from locked Swiss geometry; never allocate a dataset split.

Compare the arrangement of residential area polygons, preserving holes. Furniture,
openings, elevations, professional reasons and simulation labels are not assessed.
Equal shapes suggest a conservative grouping, not identity of physical projects.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(ref):
    path = Path(ref['path'])
    if path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']:
        raise ValueError('Locked input changed: ' + str(path))
    return path


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def canonical_ring(points, grid):
    points = np.rint(np.asarray(points)[:, :2] / grid).astype(np.int64).tolist()
    ring = []
    for point in points:
        value = tuple(point)
        if not ring or value != ring[-1]:
            ring.append(value)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    # Collinear vertices added during digitization should not change a polygon key.
    while len(ring) >= 3:
        keep = []
        for i, b in enumerate(ring):
            a, c = ring[i-1], ring[(i+1) % len(ring)]
            cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            forward = (b[0]-a[0])*(c[0]-b[0]) + (b[1]-a[1])*(c[1]-b[1])
            if cross != 0 or forward < 0:
                keep.append(b)
        if len(keep) == len(ring):
            break
        ring = keep
    if len(ring) < 3:
        raise ValueError('Area ring degenerates at fingerprint precision')
    candidates = []
    for sequence in (ring, list(reversed(ring))):
        minimum = min(sequence)
        candidates.extend(tuple(sequence[i:] + sequence[:i])
                          for i, point in enumerate(sequence) if point == minimum)
    return min(candidates)


def polygon_key(poly, grid):
    return (canonical_ring(poly.exterior.coords, grid),
            tuple(sorted(canonical_ring(ring.coords, grid) for ring in poly.interiors)))


def frame(areas):
    """Normalize translation and arbitrary rotation, retaining all eight symmetries."""
    if not areas:
        raise ValueError('No area polygons')
    for _, poly in areas:
        if poly.geom_type != 'Polygon' or poly.is_empty or not poly.is_valid or poly.has_z:
            raise ValueError('Unsupported, invalid, empty or nonplanar area polygon')
        if not np.isfinite(shapely.get_coordinates(poly)).all() or poly.area <= 0:
            raise ValueError('Nonfinite or zero-area geometry')
    union = shapely.union_all([poly for _, poly in areas])
    rectangle = union.minimum_rotated_rectangle
    if rectangle.geom_type != 'Polygon' or not union.is_valid or union.area <= 0:
        raise ValueError('Invalid unit union or bounding rectangle')
    points = list(rectangle.exterior.coords)
    dx, dy = points[1][0]-points[0][0], points[1][1]-points[0][1]
    angle = -math.degrees(math.atan2(dy, dx))
    origin = (union.centroid.x, union.centroid.y)
    normalized = []
    for label, poly in areas:
        rotated = affinity.rotate(poly, angle, origin=origin)
        normalized.append((label, affinity.translate(rotated, -origin[0], -origin[1])))
    normalized_union = shapely.union_all([p for _, p in normalized])
    edges = [math.dist(points[i], points[i+1]) for i in range(4)]
    return normalized, normalized_union, sorted(edges)[:1] + sorted(edges)[-1:]


def symmetries(areas):
    for mirror in (1, -1):
        for angle in (0, 90, 180, 270):
            yield [(label, affinity.rotate(affinity.scale(poly, xfact=mirror, yfact=1,
                                                         origin=(0, 0)), angle, origin=(0, 0)))
                   for label, poly in areas]


def fingerprints(normalized, grid):
    untyped, typed = [], []
    for variant in symmetries(normalized):
        records = [(label, polygon_key(poly, grid)) for label, poly in variant]
        for target, payload in ((untyped, sorted(key for _, key in records)),
                                (typed, sorted(records))):
            encoded = json.dumps(payload, separators=(',', ':')).encode()
            target.append(hashlib.sha256(encoded).hexdigest())
    return {'geometry_sha256': min(untyped), 'typed_geometry_sha256': min(typed)}


def compare_units(first, second, criteria):
    """Require agreement of both outer coverage and every matched room polygon."""
    if len(first['areas']) != len(second['areas']):
        return None
    a_union = first['union']
    for variant in symmetries(second['areas']):
        b_union = shapely.union_all([p for _, p in variant])
        intersection = a_union.intersection(b_union).area
        union_area = a_union.area + b_union.area - intersection
        outer_iou = intersection / union_area if union_area > 0 else 0.0
        if outer_iou < criteria['union_iou_min']:
            continue
        outer_distance = a_union.hausdorff_distance(b_union)
        if outer_distance > criteria['union_hausdorff_max_m']:
            continue
        matrix = np.zeros((len(variant), len(variant)))
        for i, (_, a) in enumerate(first['areas']):
            for j, (_, b) in enumerate(variant):
                overlap = a.intersection(b).area
                matrix[i, j] = overlap / (a.area + b.area - overlap)
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        minimum_iou = float(matrix[rows, columns].min())
        if minimum_iou < criteria['matched_room_iou_min']:
            continue
        maximum_distance = max(first['areas'][i][1].hausdorff_distance(variant[j][1])
                               for i, j in zip(rows, columns))
        if maximum_distance > criteria['matched_room_hausdorff_max_m']:
            continue
        return {'union_iou': outer_iou, 'union_hausdorff_m': outer_distance,
                'minimum_matched_room_iou': minimum_iou,
                'maximum_matched_room_hausdorff_m': maximum_distance,
                'matched_semantic_labels_equal': all(first['areas'][i][0] == variant[j][0]
                                                     for i, j in zip(rows, columns))}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    result = {'started_at': now(), 'status': 'running', 'rows_scanned': 0,
              'processed_units': 0, 'model_calls': 0, 'training_updates': 0,
              'allocated_projects': 0, 'plan_sha256': sha(args.plan)}

    def progress(stage):
        if time.monotonic() - started > plan['budget']['seconds']:
            raise TimeoutError('Frozen geometry audit deadline exceeded')
        result['stage'] = stage
        dump(args.out/'RESULT.json', result)
        print(now(), stage, 'rows', result['rows_scanned'], 'units', result['processed_units'],
              'near_checks', result.get('near_pairs_checked', 0), flush=True)

    progress('verify_inputs')
    try:
        if sha(Path(__file__)) != plan['script_sha256']:
            raise ValueError('Audit script differs from frozen plan')
        archive = checked(plan['archive'])
        index = checked(plan['plan_floor_index'])
        selected = {}
        all_sites = set()
        with index.open() as stream:
            for row in csv.DictReader(stream):
                key = (row['site_id'], row['building_id'], row['plan_id'])
                assert key not in selected
                selected[key] = min(row['floor_ids'].split('|'), key=int)
                all_sites.add(row['site_id'])
        units, skipped_rows = defaultdict(list), Counter()
        unit_metadata, bad_units = {}, {}
        csv.field_size_limit(4*1024**2)
        with zipfile.ZipFile(archive) as zipped:
            member = plan['geometry_member']
            assert zipped.getinfo(member).file_size <= plan['budget']['max_uncompressed_bytes']
            with zipped.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
                for count, row in enumerate(reader, 1):
                    if count > plan['budget']['max_rows'] or None in row or any(v is None for v in row.values()):
                        raise ValueError('Malformed CSV or frozen row limit exceeded')
                    result['rows_scanned'] = count
                    if count % 100000 == 0:
                        progress('scan_area_polygons')
                    if row['entity_type'] != 'area':
                        continue
                    parent = (row['site_id'], row['building_id'], row['plan_id'])
                    if row['floor_id'] != selected[parent]:
                        skipped_rows['repeated_plan_floor'] += 1
                        continue
                    if row['unit_usage'] != 'RESIDENTIAL':
                        skipped_rows['not_residential'] += 1
                        continue
                    if not row['unit_id'] or not row['apartment_id']:
                        skipped_rows['missing_unit_or_apartment_id'] += 1
                        continue
                    key = (*parent, row['floor_id'], row['unit_id'])
                    if key not in unit_metadata:
                        unit_metadata[key] = row['apartment_id']
                    elif unit_metadata[key] != row['apartment_id']:
                        bad_units[key] = 'Conflicting apartment IDs within unit'
                    units.setdefault(key, [])
                    try:
                        poly = shapely.from_wkt(row['geometry'])
                    except Exception as error:
                        bad_units[key] = 'WKT parse failure: ' + type(error).__name__
                        continue
                    units[key].append((row['entity_subtype'], poly))
                    if len(units) > plan['budget']['max_units']:
                        raise ValueError('Frozen unit limit exceeded')
        result.update(geometry_csv_crc_checked=True, selected_plan_count=len(selected),
                      raw_residential_units=len(units), skipped_area_rows=dict(skipped_rows))
        progress('canonicalize_units')
        groups = defaultdict(list)
        unit_features = {}
        exclusions = []
        fields = ['site_id', 'building_id', 'plan_id', 'floor_id', 'unit_id', 'apartment_id',
                  'area_polygon_count', 'union_area_m2', 'union_perimeter_m', 'short_extent_m',
                  'long_extent_m', 'geometry_sha256', 'typed_geometry_sha256']
        with (args.out/'UNIT_FINGERPRINTS.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for key, areas in sorted(units.items()):
                result['processed_units'] += 1
                if result['processed_units'] % 100 == 0:
                    progress('canonicalize_units')
                try:
                    if key in bad_units:
                        raise ValueError(bad_units[key])
                    if len(areas) < plan['minimum_area_polygons']:
                        raise ValueError('Too few area polygons for informative grouping')
                    normalized, union, extents = frame(areas)
                    hashes = fingerprints(normalized, plan['fingerprint_grid_m'])
                except (ValueError, shapely.errors.GEOSException) as error:
                    exclusions.append({'unit_key': list(key), 'reason': str(error)})
                    continue
                feature = {'areas': normalized, 'union': union, 'extents': extents,
                           'key': key, **hashes}
                unit_features[key] = feature
                groups[hashes['geometry_sha256']].append(key)
                writer.writerow([*key, unit_metadata[key], len(areas), union.area, union.length,
                                 *extents, hashes['geometry_sha256'], hashes['typed_geometry_sha256']])
        del units
        dump(args.out/'EXCLUDED_UNITS.json', exclusions)
        parent = {site: site for site in all_sites}

        def root(site):
            while parent[site] != site:
                parent[site] = parent[parent[site]]
                site = parent[site]
            return site

        def link(sites):
            roots = sorted({root(site) for site in sites})
            for other in roots[1:]:
                parent[other] = roots[0]

        exact_groups, representatives = [], []
        for digest, members in sorted(groups.items()):
            sites = sorted({key[0] for key in members})
            typed = {unit_features[key]['typed_geometry_sha256'] for key in members}
            if len(sites) > 1:
                link(sites)
                exact_groups.append({'geometry_sha256': digest, 'sites': sites,
                                     'unit_keys': [list(key) for key in members],
                                     'typed_geometry_variants': len(typed)})
            representative = dict(unit_features[min(members)])
            representative['sites'] = sites
            representatives.append(representative)
        dump(args.out/'CROSS_SITE_QUANTIZED_GROUPS.json', exact_groups)
        result.update(valid_units=len(unit_features), excluded_units=len(exclusions),
                      unique_geometry_fingerprints=len(groups),
                      cross_site_quantized_groups=len(exact_groups), near_pairs_proposed=0,
                      near_pairs_checked=0, near_matches=0)
        progress('near_geometry_candidates')
        buckets = defaultdict(list)
        for record in representatives:
            buckets[len(record['areas'])].append(record)
        matches = []
        for room_count, records in sorted(buckets.items()):
            # Five-percent log-distance gates retrieve candidates, not an equality proof.
            features = np.asarray([[math.log(x['union'].area), math.log(x['union'].length),
                                    math.log(x['extents'][0]), math.log(x['extents'][1])]
                                   for x in records])
            tree = cKDTree(features)
            radius = plan['near_criteria']['descriptor_log_radius']
            pair_count = (int(tree.count_neighbors(tree, radius, p=np.inf))-len(records))//2
            if result['near_pairs_proposed'] + pair_count > plan['budget']['max_candidate_pairs']:
                raise ValueError('Frozen candidate-pair limit exceeded; no partial split is admitted')
            pairs = tree.query_pairs(radius, p=np.inf, output_type='ndarray')
            result['near_pairs_proposed'] += len(pairs)
            if len(pairs) != pair_count:
                raise ValueError('Candidate-count inconsistency')
            for i, j in pairs:
                a, b = records[int(i)], records[int(j)]
                sites = sorted(set(a['sites']) | set(b['sites']))
                if len(sites) < 2:
                    continue
                result['near_pairs_checked'] += 1
                if result['near_pairs_checked'] % 100 == 0:
                    progress('verify_cross_site_near_geometry')
                score = compare_units(a, b, plan['near_criteria'])
                if score is not None:
                    link(sites)
                    matches.append({'first_geometry_sha256': a['geometry_sha256'],
                                    'second_geometry_sha256': b['geometry_sha256'],
                                    'first_representative_unit': list(a['key']),
                                    'second_representative_unit': list(b['key']),
                                    'sites': sites, **score})
                    result['near_matches'] = len(matches)
        dump(args.out/'CROSS_SITE_NEAR_MATCHES.json', matches)
        components = defaultdict(list)
        for site in sorted(all_sites):
            components[root(site)].append(site)
        proposed = []
        for sites in sorted(components.values(), key=lambda x: x[0]):
            digest = hashlib.sha256('|'.join(sites).encode()).hexdigest()[:16]
            proposed.append({'geometry_family_id': 'swiss_v3_shape_'+digest, 'site_ids': sites,
                             'allocation': 'quarantine_unallocated',
                             'contains_exposed_example_site127': '127' in sites})
        dump(args.out/'SITE_GROUP_PROPOSALS.json', proposed)
        result.update(status='geometry_group_proposals_complete_no_split', exit_code=0,
                      sites=len(all_sites), proposed_geometry_families=len(proposed),
                      multi_site_families=sum(len(x['site_ids']) > 1 for x in proposed),
                      largest_family_sites=max(len(x['site_ids']) for x in proposed),
                      output_bytes=sum(p.stat().st_size for p in args.out.iterdir() if p.is_file()),
                      source_simulation_values_read=False, native_scenes_derived=0,
                      limitations=plan['limitations'])
        if result['output_bytes'] > plan['budget']['max_output_bytes']:
            raise ValueError('Frozen output limit exceeded')
    except BaseException as error:
        result.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result.update(ended_at=now(), elapsed_seconds=time.monotonic()-started)
        dump(args.out/'RESULT.json', result)
        print(now(), result['status'], 'exit', result.get('exit_code'), flush=True)


if __name__ == '__main__':
    main()
