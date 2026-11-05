"""Geometric falsification cases for leakage grouping; no corpus or model access."""
import json
import hashlib
from pathlib import Path
import sys

from shapely import affinity
import shapely
from shapely.geometry import Polygon, box

from group_swiss_geometry import compare_units, fingerprints, frame


CRITERIA = {'union_iou_min': .995, 'union_hausdorff_max_m': .02,
            'matched_room_iou_min': .98, 'matched_room_hausdorff_max_m': .05}


def feature(areas):
    normalized, union, extents = frame(areas)
    return {'areas': normalized, 'union': union, 'extents': extents}


def main():
    checks = []
    # Asymmetric outline with an inner courtyard hole, two internal rooms.
    first = [('ROOM', Polygon([(0, 0), (4, 0), (4, 3), (0, 3)],
                              [[(1, 1), (2, 1), (2, 2), (1, 2)]])),
             ('KITCHEN', box(4, 0, 7, 2))]
    a = feature(first)
    original = fingerprints(a['areas'], .001)
    for angle, scale in [(0, 1), (37, 1), (90, 1), (137, -1)]:
        transformed = [(label, affinity.translate(affinity.rotate(
            affinity.scale(poly, xfact=scale, yfact=1, origin=(0, 0)), angle,
            origin=(0, 0)), 103.25, -47.5)) for label, poly in reversed(first)]
        b = feature(transformed)
        assert fingerprints(b['areas'], .001) == original
        assert compare_units(a, b, CRITERIA) is not None
        checks.append('rigid_transform_and_mirror_'+str(angle)+'_'+str(scale))
    relabel = [('BATHROOM' if label == 'KITCHEN' else label, poly) for label, poly in first]
    other = fingerprints(feature(relabel)['areas'], .001)
    assert other['geometry_sha256'] == original['geometry_sha256']
    assert other['typed_geometry_sha256'] != original['typed_geometry_sha256']
    assert compare_units(a, feature(relabel), CRITERIA)['matched_semantic_labels_equal'] is False
    checks.append('semantic_difference_recorded_separately')
    filled = [('ROOM', box(0, 0, 4, 3)), first[1]]
    assert fingerprints(feature(filled)['areas'], .001)['geometry_sha256'] != original['geometry_sha256']
    assert compare_units(a, feature(filled), CRITERIA) is None
    checks.append('hole_is_not_discarded')
    split_left = [('ROOM', box(0, 0, 3, 4)), ('ROOM', box(3, 0, 7, 4))]
    split_wrong = [('ROOM', box(0, 0, 1, 4)), ('ROOM', box(1, 0, 7, 4))]
    assert compare_units(feature(split_left), feature(split_wrong), CRITERIA) is None
    checks.append('same_outer_outline_different_internal_partition_rejected')
    split_close = [('ROOM', box(0, 0, 3.005, 4)), ('ROOM', box(3.005, 0, 7, 4))]
    score = compare_units(feature(split_left), feature(split_close), CRITERIA)
    assert score is not None and score['maximum_matched_room_hausdorff_m'] <= .005001
    checks.append('five_millimetre_internal_change_is_bounded_near_match')
    bigger = [(label, affinity.scale(poly, xfact=1.1, yfact=1.1, origin=(0, 0)))
              for label, poly in split_left]
    assert compare_units(feature(split_left), feature(bigger), CRITERIA) is None
    checks.append('different_scale_rejected')
    redundant = [('ROOM', Polygon([(0, 0), (1, 0), (3, 0), (3, 2), (3, 4), (0, 4)])),
                 ('ROOM', box(3, 0, 7, 4))]
    assert fingerprints(feature(redundant)['areas'], .001) == fingerprints(feature(split_left)['areas'], .001)
    checks.append('collinear_digitization_points_ignored')
    for name, invalid in [('bow_tie', Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])),
                          ('empty', Polygon())]:
        try:
            frame([('ROOM', invalid)])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid geometry accepted')
        checks.append(name+'_rejected')
    source_case = None
    if len(sys.argv) > 2:
        case_path = Path(sys.argv[2])
        case = json.loads(case_path.read_text())
        actual = [(row['entity_subtype'], shapely.from_wkt(row['geometry']))
                  for row in case['area_rows']]
        source_union = shapely.union_all([poly for _, poly in actual])
        normalized = feature(actual)
        assert all(value > 0 for value in normalized['extents'])
        assert abs(normalized['union'].area-source_union.area) < 1e-8
        assert normalized['extents'][0]*normalized['extents'][1] >= source_union.area
        translated = [(label, affinity.translate(poly, 231.25, -132.75)) for label, poly in actual]
        moved = feature(translated)
        assert fingerprints(normalized['areas'], .001) == fingerprints(moved['areas'], .001)
        assert compare_units(normalized, moved, CRITERIA) is not None
        checks.extend(['real_source_positive_extent_and_area_preserved',
                       'real_source_translation_invariance'])
        source_case = {'path': str(case_path),
                       'sha256': hashlib.sha256(case_path.read_bytes()).hexdigest(),
                       'unit_key': case['unit_key']}
    report = {'status': 'passed', 'checks': checks, 'count': len(checks),
              'criteria': CRITERIA, 'model_calls': 0, 'training_updates': 0,
              'source_case': source_case,
              'limits': 'Synthetic geometry and optional source regression checks; no corpus recall or physical-project identity established.'}
    if len(sys.argv) > 1:
        with Path(sys.argv[1]).open('x') as stream:
            stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
