# Dataset preparation

NaLA trains on preprocessed scenes from **3D-FRONT**, **Imaginarium**, and optionally **M3DLayout**. Obtain each dataset under its own license, then convert raw assets/scenes into the layout below.

## Expected layout under `NALA_DATA_ROOT` (default: `data/`)

```
data/
  3Dfront_cleaned/
    converted_scenes_file_list.json
    centralized_scene/
    3D-FUTURE-model/
    empty_scene_glbs/
    category_to_model_ids.json
    model_ids_to_category.json
    sampled_orders.json
    usage_table.json
  Imaginarium/
    converted_scenes_file_list.json
    centralized_scene/
    imaginarium_assets_obj/
    empty_scene_glbs/
    category_to_model_ids.json
    model_ids_to_category.json
    sampled_orders.json
    usage_table.json
  M3DLayout/scene_dataset/infinigen_converted/
    converted_scenes_file_list.json
    centralized_scene/
    assets/
    empty_scene_glbs/
    category_to_model_ids.json
    model_ids_to_category.json
    sampled_orders.json
    usage_table.json
```

## Normalization requirements

### Coordinate frames

**3D-FRONT / Imaginarium:** convert room layouts from Z-up to Y-up when loading. Keep asset meshes in the same frame after centering.

**M3DLayout:** set `scene_is_m3dlayout=True`. Use metric units and skip Z→Y remapping; apply any recorded `center_offset` when composing rooms.

### Asset point clouds

Center each mesh by its AABB, then scale so the longest edge is 1. Sample and voxel-downsample surface points (~4096, voxel ≈ 0.02) and cache as `raw_model_points.pt` next to the mesh (`tools/preprocess_pc.py`). Poses are applied later in world space during training.

### Empty-room (scene) point clouds

Sample floor, walls, and ceiling of the empty room. Collate to sparse voxels (`voxel_size=0.05`) for the SPFormer scene branch.

### Placement order

Prefer large → small by volume. Optional support-aware orders can be stored in `sampled_orders.json` and passed via `sampled_orders_path`.

## Tools and core APIs

| Entry | Purpose |
|-------|---------|
| `tools/preprocess_pc.py` | Build `raw_model_points.pt` for an asset library |
| `tools/build_support_graph.py` | Sample placement orders into `sampled_orders.json` |
| `tools/build_asset_usage_table.py` | Build asset usage / replaceability tables |
| `ThreeDFrontDataset` | Main dataset over scene ids + layout / asset / empty-room dirs |
| `LayoutLLMDataCollator` / `LayoutLLMEvalDataCollator` | Build training / inference batches |

```python
ThreeDFrontDataset(
    scene_list,
    layout_data_dir=...,
    asset_lib_dir=...,
    empty_scene_dir=...,
    asset_swap_dict_dir=...,      # category_to_model_ids.json
    asset_category_list=...,      # model_ids_to_category.json
    sampled_orders_path=...,      # optional
    asset_usage_table_path=...,   # optional
    scene_is_m3dlayout=False,
)
```

Scene lists come from each dataset’s `converted_scenes_file_list.json`.

## Training mix

`--dataset_to_use mixed_all` samples **M3DLayout : 3D-FRONT : Imaginarium ≈ 2 : 2 : 1**. Single-dataset modes: `3D_FRONT`, `Imaginarium`, `M3DLayout`.

## License note

3D-FRONT, Imaginarium, and M3DLayout retain their original terms of use. Obtain the data independently and comply with each license.
