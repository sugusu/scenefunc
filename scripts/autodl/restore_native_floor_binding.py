"""Restore a floor state without invalidating native tool/vision references.

Only serialized use on the original HouseLayout instance. No new room, tool
answer, opening or geometry is invented. This is ordinary resume correctness.
"""
from scenesmith.agent_utils.house import HouseLayout


def restore_floor_in_place(agent,data,*,house_dir):
    original=agent.layout
    if type(original) is not HouseLayout:raise TypeError('Exact native HouseLayout expected')
    restored=HouseLayout.from_dict(data,house_dir=house_dir)
    if not hasattr(original,'__dict__') or not hasattr(restored,'__dict__'):
        raise TypeError('Native layout storage contract changed')
    vision=getattr(agent,'_vision_tools',None)
    if vision is not None and vision.layout is not original:
        raise ValueError('Existing native vision tools already reference another layout')
    original.__dict__.clear();original.__dict__.update(restored.__dict__)
    assert agent.layout is original
    if vision is not None:assert vision.layout is agent.layout
    return dict(identity_preserved=True,layout_object_id=id(original),
        restored_rooms=len(original.room_specs),restored_placed_rooms=len(original.placed_rooms or []),
        existing_vision_reference_preserved=vision is not None)
