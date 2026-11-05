"""Declared geometric consequences of concentrating three proxy storage units."""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from inspect_native_mesh_contacts import world_transform
from measure_room_access import measure, obstacles
from measure_free_floor import largest_floor_rectangle, footprint

def storage_measures(state,room,root,plan):
 settings=plan['access_settings'];ids=plan['storage_ids'];region=box(*plan['living_region_xyxy_m']);ob=obstacles(state,room,root,settings['assumed_body_height_m']);hits=[(x['id'],region.intersection(box(*x['lo'][:2],*x['hi'][:2]))) for x in ob];occupied=unary_union([s for _,s in hits]).area
 approach=[]
 for width in settings['square_body_widths_m']:
  points={}
  for oid in ids:
   obj=state['objects'][oid];lo=np.asarray(obj['bbox_min']);hi=np.asarray(obj['bbox_max']);matrix=world_transform(obj);pad=width/2+0.05
   for i,u in enumerate(np.linspace(lo[0]+pad,hi[0]-pad,5)):
    xy=(matrix@np.array([u,hi[1]+pad,0,1]))[:2];points[f'{oid}:{i}']=xy.tolist()
  scenario=dict(settings,square_body_widths_m=[width],approach_points_xy_m=points);approach.append(measure(state,room,root,scenario)['scenarios'][0])
 return {'fixed_access':measure(state,room,root,settings),'floor_rectangle':largest_floor_rectangle(state,room,root,settings),'living_region':{'bounds_xyxy_m':plan['living_region_xyxy_m'],'area_m2':region.area,'occupied_envelope_m2':occupied,'unoccupied_envelope_m2':region.area-occupied,'overlaps':[{'object_id':oid,'area_m2':shape.area} for oid,shape in hits if shape.area>1e-9],'scope':'Researcher-defined living-floor region; not source area ratio, light, comfort or activity success.'},'storage_front_approaches':approach,'storage':{'object_ids':ids,'sum_footprint_m2':sum(footprint(state['objects'][oid]).area for oid in ids),'external_box_volume_m3':sum(float(np.prod(np.array(state['objects'][oid]['bbox_max'])-state['objects'][oid]['bbox_min'])) for oid in ids),'capacity_known':False,'front_definition':'Researcher-declared local +Y, five sampled points per unit/body width. Static boxes have no doors, shelves or usable storage model.'}}
