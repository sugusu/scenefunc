"""Explicit prompt compaction; current facts remain losslessly recoverable.

Historical query object details are summarized as IDs and a full-result hash.
Full native receipts/checkpoints remain unchanged on disk. No coordinates or
design action are suggested by this representation change.
"""
import copy
import hashlib
import json
from collections import Counter


OBJECT_FIELDS=('object_id','object_type','name','transform','bbox_min','bbox_max','immutable')
NOTICE=(
    'Current object facts, asset identities and numeric values are unchanged. '
    'Repeated catalog descriptions/limitations use explicit dictionary references. '
    'Past successful state-query object details are summarized by object IDs and a full-result hash; '
    'the complete current typed object table is above, and exact historical results remain in the checkpoint. '
    'Other tool results and all attempted arguments are retained verbatim. '
    'Native get_current_scene_state currently counts structural/opening objects as furniture; '
    'use derived_object_counts and object_type for counts. Its input arguments are {}. '
    'check_facing_tool measures orientation, not occlusion, privacy or activity suitability.'
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def pack_context(source):
    context=copy.deepcopy(source)
    context['objects']={oid:{key:obj[key] for key in OBJECT_FIELDS} for oid,obj in context['objects'].items()}
    receipts=context.pop('prior_tool_receipts')
    descriptions={};limitations={}
    def ref(table,value,prefix):
        for key,existing in table.items():
            if existing==value:return key
        key=prefix+str(len(table));table[key]=copy.deepcopy(value);return key
    for entry in context['asset_catalog']:
        entry['description_ref']=ref(descriptions,entry.pop('description'),'D')
        entry['limitation_refs']=[ref(limitations,item,'L') for item in entry.pop('limitations')]
    context['catalog_descriptions']=descriptions;context['catalog_limitations']=limitations
    context['derived_object_counts']=dict(Counter(obj['object_type'] for obj in context['objects'].values()))
    compact_receipts=[]
    for receipt in receipts:
        row={key:copy.deepcopy(receipt[key]) for key in ('step_idx','tool_calls','results')}
        for reply in row['results']:
            result=reply.get('output')
            if reply['name']=='get_current_scene_state' and isinstance(result,dict) and result.get('success') is True and isinstance(result.get('objects'),list):
                full=copy.deepcopy(result)
                result['queried_object_ids']=[x['object_id'] for x in result.pop('objects')]
                result['full_query_result_sha256']=canonical_hash(full)
                result['historical_details_omitted']=True
        compact_receipts.append(row)
    return context,{'recent_actual_batches':compact_receipts,'history_policy':NOTICE}


def restore_catalog(context):
    entries=copy.deepcopy(context['asset_catalog'])
    for entry in entries:
        entry['description']=copy.deepcopy(context['catalog_descriptions'][entry.pop('description_ref')])
        entry['limitations']=[copy.deepcopy(context['catalog_limitations'][r]) for r in entry.pop('limitation_refs')]
    return entries


def geometry_feedback(audit):
    return {'kind':'independent_program_geometry_feedback','measured_state_manifest_sha256':audit['input_manifest_sha256'],
        'visual_triangle_intersection_pairs':[x['objects'] for x in audit['positive_aabb_pairs'] if x['triangle_surface_intersection_detected']],
        'outside_nominal_room_ids':[x['object_id'] for x in audit['bounds'] if x['outside_nominal_room_xy']],
        'immutable_structure_records_unchanged':all(audit['immutable_records_unchanged'].values()),
        'limits':['Surface intersections are not penetration-depth or stability tests.','No intersection does not rule out containment.','These measurements do not suggest target coordinates or a reference layout.']}


def use_feedback(audit):
    # Observable catalog coverage is a warning, never a proof of functionality.
    names={x['name'] for x in audit['inventory']}
    present={
        'living_seating':bool(names & {'sofa','dining_chair_1'}),
        'sleeping':bool(names & {'queen_bed'}),
        'cooking_appliance':bool(names & {'cooker'}),
        'food_preparation_surface':bool(names & {'kitchen_counter','dining_table'}),
        'kitchen_sink':bool(names & {'kitchen_sink'}),
        'cold_food_storage':bool(names & {'refrigerator'}),
        'bathing':bool(names & {'shower'}),
        'toileting':bool(names & {'toilet'}),
        'washbasin':bool(names & {'washroom_vanity'}),
        'storage':bool(names & {'bookshelf','storage_bin'}),
    }
    return {'kind':'conditional_activity_and_inventory_feedback_not_professional_preference',
        'catalog_roles_present_not_function_pass':present,
        'coverage_rule':'The brief requires living, sleeping, cooking, bathing/toilet and storage. These catalog roles are researcher operational indicators, not requirements to use one exact asset or layout. Multipurpose alternatives may be proposed but require evidence.',
        'access_scenarios':[{'assumed_square_body_width_m':s['square_body_width_m'],
            'entry_cell_clear':s['entry_cell_clear'],
            'unreachable_hypothetical_approach_points':[k for k,v in s['targets'].items() if not v['reachable_from_entry']]}
            for s in audit['access']['scenarios']],
        'access_assumptions':{'grid_m':audit['access']['settings']['grid_m'],
            'body_height_m':audit['access']['settings']['assumed_body_height_m'],
            'target_rule':'0.35m beyond canonical front envelope; both side midpoints for bed; hypothesized approach points, not measured human ergonomics.',
            'door':'assumed open without leaf'},
        'warnings':['Object presence and facing-away relations do not verify use or privacy.',
            'A larger empty rectangle obtained by deleting a required function is not a dwelling improvement.',
            'A conservative blocked route is a warning, not proof no person could pass.',
            'Sanitary enclosure/privacy, articulation, installation, heat and comfort remain unverified.']}
