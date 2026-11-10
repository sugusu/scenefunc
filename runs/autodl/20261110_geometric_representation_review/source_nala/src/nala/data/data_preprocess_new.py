import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
import os
from nala.models.point_encoder.PointEncoder import SceneQueryExtractor
from torch.utils.data import DataLoader, Dataset
import trimesh
import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None
    # from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import random
from scipy.spatial.transform import Rotation as R_scipy
import trimesh.transformations as tra
import json
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def scatter_mean(src, index, dim=0, dim_size=None):
    """"""
    if dim != 0:
        raise NotImplementedError("This simple version assumes dim=0")

    if dim_size is None:
        dim_size = int(index.max()) + 1

    out = torch.zeros((dim_size,) + src.shape[1:], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)

    count = torch.bincount(index, minlength=dim_size).clamp(min=1)
    count = count.view(-1, *([1] * (src.dim() - 1)))

    return out / count


def scale_matrix(sx, sy, sz):
    S = np.eye(4)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz
    return S


def numpy_voxel_downsample(points, colors, voxel_size):
    """"""
    if len(points) == 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    view = np.ascontiguousarray(voxel_indices).view(
        np.dtype((np.void, voxel_indices.dtype.itemsize * voxel_indices.shape[1]))
    )

    _, unique_idx, inverse_indices = np.unique(
        view, return_index=True, return_inverse=True
    )

    down_points = points[unique_idx]
    if colors is not None:
        down_colors = colors[unique_idx]
    else:
        down_colors = None

    return down_points, down_colors


def translation_matrix(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def quantize_pose(pos_vector, scene_size, num_bins=20):
    """"""
    x, y, z, scale, rot = pos_vector.tolist()
    sx, sy, sz = scene_size.tolist()

    def get_bin_and_res_dynamic(val, size_dim, steps):
        half_size = size_dim / 2.0
        min_v = -half_size
        max_v = half_size

        norm = (val - min_v) / (max_v - min_v + 1e-6)
        norm = max(0.0, min(1.0, norm))

        idx = int(norm * steps)
        if idx == steps:
            idx -= 1

        bin_center_norm = (idx + 0.5) / steps
        bin_center_val = bin_center_norm * (max_v - min_v) + min_v
        res = val - bin_center_val
        return idx, res

    idx_x, res_x = get_bin_and_res_dynamic(x, sx, num_bins)
    idx_y, res_y = get_bin_and_res_dynamic(y, sy, num_bins)
    idx_z, res_z = get_bin_and_res_dynamic(z, sz, num_bins)

    PI = math.pi
    TWO_PI = 2 * PI
    HALF_PI = PI / 2
    QUARTER_PI = PI / 4

    rot = rot % TWO_PI

    # Bin 0 (Front): Center 0      Range: [7PI/4, 2PI) U [0, PI/4)
    # Bin 1 (Left) : Center PI/2   Range: [PI/4, 3PI/4)
    # Bin 2 (Back) : Center PI     Range: [3PI/4, 5PI/4)
    # Bin 3 (Right): Center 3PI/2  Range: [5PI/4, 7PI/4)

    if QUARTER_PI <= rot < 3 * QUARTER_PI:
        idx_rot = 1
        center_rot = HALF_PI
    elif 3 * QUARTER_PI <= rot < 5 * QUARTER_PI:
        idx_rot = 2
        center_rot = PI
    elif 5 * QUARTER_PI <= rot < 7 * QUARTER_PI:
        idx_rot = 3
        center_rot = 3 * HALF_PI
    else:
        idx_rot = 0

        center_rot = 0.0 if rot < PI else TWO_PI

    res_rot = rot - center_rot

    if res_rot > PI:
        res_rot -= TWO_PI
    if res_rot < -PI:
        res_rot += TWO_PI

    bin_indices = [idx_x, idx_y, idx_z, idx_rot]
    residuals = [res_x, res_y, res_z, scale, res_rot]

    return bin_indices, residuals


def collate_point_clouds_to_sparse_input(list_of_pc_dicts, voxel_size=0.05):
    """"""
    if not list_of_pc_dicts:
        return {}

    batch_indices = []
    all_points = []
    all_colors = []

    for i, data in enumerate(list_of_pc_dicts):
        pts = data["points"]
        cols = data["colors"]
        num_points = pts.shape[0]

        batch_indices.append(torch.full((num_points, 1), i, dtype=torch.int))
        all_points.append(pts)
        all_colors.append(cols)

    batch_indices = torch.cat(batch_indices, dim=0)
    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)

    # print("all_colors: ",all_colors[0,1:6])

    coords = all_points - all_points.min(0)[0]
    feats = torch.cat([all_colors, all_points], dim=1)  # (N_total, 6)

    voxel_coords_int = torch.div(coords, voxel_size, rounding_mode="trunc").long()
    voxel_coords_with_batch = torch.cat([batch_indices, voxel_coords_int], dim=1)

    voxel_coords_unique, p2v_map = torch.unique(
        voxel_coords_with_batch, dim=0, return_inverse=True
    )
    superpoints = p2v_map

    _, v2p_map = torch.sort(p2v_map)

    # 4. Spatial Shape
    spatial_shape = (voxel_coords_unique[:, 1:].max(0)[0] + 1).numpy().tolist()

    num_scenes = len(list_of_pc_dicts)
    num_superpoints_per_scene = []

    unique_batch_ids = voxel_coords_unique[:, 0]
    for i in range(num_scenes):
        count = (unique_batch_ids == i).sum().item()
        num_superpoints_per_scene.append(count)

    batch_offsets = torch.cumsum(
        torch.tensor([0] + num_superpoints_per_scene), dim=0
    ).long()

    superpoints_centers = scatter_mean(all_points, superpoints, dim=0)

    superpoints_scene_sizes = []
    for i in range(num_scenes):
        num_sps = num_superpoints_per_scene[i]
        if num_sps > 0:
            sz = list_of_pc_dicts[i]["scene_size"]  # (3,)
            superpoints_scene_sizes.append(sz.unsqueeze(0).repeat(num_sps, 1))

    if superpoints_scene_sizes:
        superpoints_scene_sizes = torch.cat(superpoints_scene_sizes, dim=0)
    else:
        superpoints_scene_sizes = torch.empty(0, 3, dtype=torch.float32)

    return {
        "voxel_coords": voxel_coords_unique,  # (N_sp, 4) [b, x, y, z]
        "p2v_map": p2v_map.long(),
        "v2p_map": v2p_map.long(),
        "spatial_shape": spatial_shape,
        "feats": feats.float(),
        "superpoints": superpoints.long(),
        "batch_offsets": batch_offsets,
        "superpoints_centers": superpoints_centers.float(),
        "superpoints_scene_sizes": superpoints_scene_sizes.float(),
    }


def empty_point_cloud(num_points: int = 1):
    return {
        "points": torch.zeros((num_points, 3), dtype=torch.float32),
        "colors": torch.zeros((num_points, 3), dtype=torch.float32),
        "scene_size": torch.zeros((3,), dtype=torch.float32),
    }


def load_point_cloud_from_glb(
    glb_path: str,
    voxel_size: float = 0.05,
    num_points: int = 4096,
    use_vertices=False,
    centralized=True,
) -> Optional[Dict]:
    """"""
    mesh = None

    # ----------------------------------------------------

    # ----------------------------------------------------

    file_dir = os.path.dirname(glb_path)
    cached_pt_path = os.path.join(file_dir, "raw_model_points.pt")

    if os.path.exists(cached_pt_path):
        try:

            data = torch.load(cached_pt_path, map_location="cpu", weights_only=True)

            if data["points"].shape[0] != num_points:

                curr_n = data["points"].shape[0]
                if curr_n > num_points:
                    idx = torch.randperm(curr_n)[:num_points]
                    data["points"] = data["points"][idx]
                    data["colors"] = data["colors"][idx]
                elif curr_n < num_points:
                    idx = torch.randint(0, curr_n, (num_points,))
                    data["points"] = data["points"][idx]
                    data["colors"] = data["colors"][idx]

            return data

        except Exception as e:

            print(
                f"[Warning] Failed to load cached pt {cached_pt_path}: {e}. Fallback to online processing."
            )

            # ==========================================

            # ==========================================
    try:

        mesh = trimesh.load(glb_path, force="mesh")
    except Exception as e_tex:

        # print(f"[Info] Standard load failed for {glb_path}: {e_tex}. Retrying without materials...")
        try:

            mesh = trimesh.load(glb_path, force="mesh", skip_materials=True)
        except Exception as e_geo:

            print(f"[Error] Failed to load geometry for {glb_path}: {e_geo}")
            return empty_point_cloud(num_points)

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return empty_point_cloud(num_points)
        mesh = mesh.dump(concatenate=True)

        # ==========================================
        # 2. Geometry Processing (Modified for Scaling)
        # ==========================================
    try:
        # compute center (AABB center)

        original_bbox_min, original_bbox_max = mesh.bounds
        center = (original_bbox_min + original_bbox_max) / 2

        mesh.vertices -= center

        max_extent = np.max(mesh.extents)
        if max_extent < 1e-6:
            max_extent = 1.0
        norm_scale = 1.0 / max_extent

        mesh.apply_scale(norm_scale)

        points = np.array([])

        if not use_vertices:
            try:

                points, _ = trimesh.sample.sample_surface(mesh, count=100000)
                points = points.astype(np.float32)
            except Exception as e:

                # print(f"[Warning] Sampling failed, falling back to vertices.")
                points = mesh.vertices.astype(np.float32)
        else:
            points = mesh.vertices.astype(np.float32)

        if points.shape[0] == 0:
            return empty_point_cloud(num_points)

        # ==========================================

        # ==========================================
        colors = None
        try:

            visual = mesh.visual

            if hasattr(visual, "to_color"):

                try:
                    visual = visual.to_color()
                except:

                    visual = None

            raw_colors = getattr(visual, "vertex_colors", None)

            if raw_colors is None:

                colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)
            elif raw_colors.ndim == 1 and raw_colors.shape[0] == 4:

                single_color = raw_colors[:3].astype(np.float32) / 255.0
                colors = np.tile(single_color, (points.shape[0], 1))
            else:

                if raw_colors.shape[1] >= 3:
                    colors = raw_colors[:, :3].astype(np.float32) / 255.0
                else:
                    colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)
        except Exception:

            colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)

        if colors is None or len(colors) != len(points):
            colors = np.full((points.shape[0], 3), 0.5, dtype=np.float32)

            # ==========================================
            # 4. Downsampling (Normalized Space)
            # ==========================================

        norm_voxel_size = 0.02

        down_points, down_colors = numpy_voxel_downsample(
            points, colors, norm_voxel_size
        )

        if down_points.shape[0] == 0:
            return empty_point_cloud(num_points)

        if down_points.shape[0] > num_points:
            choice_idx = np.random.choice(
                down_points.shape[0], num_points, replace=False
            )
            down_points = down_points[choice_idx]
            down_colors = down_colors[choice_idx]
        elif down_points.shape[0] < num_points:

            choice_idx = np.random.choice(
                down_points.shape[0], num_points, replace=True
            )
            down_points = down_points[choice_idx]
            down_colors = down_colors[choice_idx]

        down_points = down_points / norm_scale

        if not centralized:
            down_points += center

            # 5. Pack Output
        points_tensor = torch.from_numpy(down_points)
        colors_tensor = torch.from_numpy(down_colors)

        if points_tensor.shape[0] == 0:
            return empty_point_cloud(num_points)

        scene_min = points_tensor.min(0)[0]
        scene_max = points_tensor.max(0)[0]
        scene_size = scene_max - scene_min

        return {
            "points": points_tensor,  # (N, 3)
            "colors": colors_tensor,  # (N, 3)
            "scene_size": scene_size,  # (3,)
        }

    except Exception as e:

        print(f"[Error] Unexpected error processing {glb_path}: {e}")
        return empty_point_cloud(num_points)


GLOBAL_MAX_RANGE = 10.0


class ThreeDFrontDataset(Dataset):

    def __init__(
        self,
        scene_list,
        layout_data_dir="data/3Dfront_cleaned/centralized_scene",
        asset_lib_dir="data/3Dfront_cleaned/3D-FUTURE-model",
        empty_scene_dir="data/3Dfront_cleaned/empty_scene_glbs",
        num_asset_points=4096,
        use_asset_swap=False,
        use_asset_desc=True,
        use_scene_desc=True,
        augment_scene_rot=False,
        augment_asset_rot=False,
        asset_swap_dict_dir="data/3Dfront_cleaned/category_to_model_ids.json",
        asset_category_list="data/3Dfront_cleaned/model_ids_to_category.json",
        pristine_prob=0.1,
        sampled_orders_path=None,
        asset_usage_table_path=None,
        scene_is_m3dlayout=False,
    ):
        self.scene_list = scene_list
        self.layout_data_dir = layout_data_dir
        self.asset_lib_dir = asset_lib_dir
        self.empty_scene_dir = empty_scene_dir
        self.num_asset_points = num_asset_points
        self.use_desc = use_asset_desc
        self.use_scene_desc = use_scene_desc
        self.use_asset_swap = use_asset_swap
        self.pristine_prob = pristine_prob
        self.scene_is_m3dlayout = scene_is_m3dlayout

        self.augment_scene_rot = augment_scene_rot
        self.augment_asset_rot = augment_asset_rot

        with open(asset_swap_dict_dir, "r") as f:
            asset_name_list = json.load(f)
            self.asset_name_list = asset_name_list

        with open(asset_category_list, "r") as f:
            model_to_cate = json.load(f)
            self.asset_category_list = model_to_cate

        self.sampled_orders = None
        if sampled_orders_path and os.path.exists(sampled_orders_path):
            with open(sampled_orders_path, "r", encoding="utf-8") as f:
                self.sampled_orders = json.load(f)
            print(
                f"[ThreeDFrontDataset] Loaded sampled orders from {sampled_orders_path} "
                f"({len(self.sampled_orders)} scenes)"
            )

        self.usage_table = None
        self.jid_to_cat = None
        if asset_usage_table_path and os.path.exists(asset_usage_table_path):
            with open(asset_usage_table_path, "r", encoding="utf-8") as f:
                self.usage_table = json.load(f)
                # Build jid_to_cat from usage_table
            self.jid_to_cat = {}
            for jid, info in self.usage_table.items():
                if info.get("category"):
                    self.jid_to_cat[jid] = info["category"]
                    # Also add from asset_category_list (model_to_cate)
            for jid, cat in self.asset_category_list.items():
                if jid not in self.jid_to_cat:
                    self.jid_to_cat[jid] = cat
            print(
                f"[ThreeDFrontDataset] Loaded usage table from {asset_usage_table_path} "
                f"({len(self.usage_table)} jids, subtree-aware swap enabled)"
            )

    def _get_sampled_paths(self, scene_id):
        """Look up pre-computed topological orderings for this scene."""
        if self.sampled_orders is None:
            return None
        entry = self.sampled_orders.get(scene_id, None)
        if entry is None:
            return None
        return entry.get("sampled_paths", None)

    def __len__(self):
        return len(self.scene_list)

    def get_asset_path(self, asset_name):
        path = os.path.join(self.asset_lib_dir, asset_name, "raw_model.obj")
        if os.path.exists(path):
            return path

        if "_" in asset_name:
            scene_id = asset_name.split("_")[0]
            alt = os.path.join(
                self.asset_lib_dir, scene_id, asset_name, "raw_model.glb"
            )
            if os.path.exists(alt):
                return alt
        return path

    def __getitem__(self, idx):
        scene_id = self.scene_list[idx]

        is_pristine = random.random() < self.pristine_prob

        do_swap = self.use_asset_swap and (not is_pristine)
        do_scene_rot = self.augment_scene_rot and (not is_pristine)
        do_asset_rot = self.augment_asset_rot and (not is_pristine)

        # print("self.layout_data_dir ",type(self.layout_data_dir))
        # print("scene_id ",scene_id)
        # print("scene_id",len(scene_id))

        layout_json = os.path.join(self.layout_data_dir, scene_id + ".json")
        with open(layout_json, "r", encoding="utf-8") as f:
            layout_items = json.load(f)

            # print(layout_json)
            # print(layout_items)

        if self.use_scene_desc:
            try:
                scene_text = layout_items["scene_type"]
            except:
                scene_text = "A room that is functional, physically comfortable, and aesthetically pleasing."
        else:
            scene_text = "A room that is functional, physically comfortable, and aesthetically pleasing."

        online_topo_paths = None
        if do_swap:
            if self.usage_table is not None:
                try:
                    from nala.data.subtree_swap import subtree_swap_scene

                    new_objects, new_graph, online_topo_paths = subtree_swap_scene(
                        layout_items["objects"],
                        self.usage_table,
                        self.jid_to_cat,
                        self.asset_name_list,
                        max_depth=3,
                    )
                    layout_items["objects"] = new_objects
                except Exception as e:
                    import traceback

                    print(f"[WARN] subtree_swap_scene failed for {scene_id}: {e}")
                    traceback.print_exc()
                    online_topo_paths = None
            else:
                layout_items_swap = []
                for j in layout_items["objects"]:
                    try:
                        category = self.asset_category_list[j["jid"]]
                        asset_name_list = self.asset_name_list
                        swap_asset_list = asset_name_list[category]
                        j["jid"] = random.choice(swap_asset_list)
                    except:
                        pass
                    layout_items_swap.append(j)
                layout_items["objects"] = layout_items_swap

        layout_items_id = [
            {"asset_uid": q["jid"], "asset_des": q.get("short_desc", "")}
            for q in layout_items["objects"]
        ]

        scene_rot_rad = 0.0
        R_scene = np.eye(3)
        if do_scene_rot:

            scene_rot_deg = np.random.choice([0, 90, 180, 270])

            scene_rot_rad = np.deg2rad(scene_rot_deg)
            # Y-Up Rotation Matrix
            c, s = np.cos(scene_rot_rad), np.sin(scene_rot_rad)
            R_scene = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        if layout_items.get("empty_scene_path", None):
            raw_scene_data = load_point_cloud_from_glb(
                layout_items["empty_scene_path"]
            )  # Expected: {'points': tensor, 'colors': tensor}
        elif os.path.exists(
            os.path.join(self.empty_scene_dir, layout_items["scene_id"] + ".glb")
        ):
            raw_scene_data = load_point_cloud_from_glb(
                os.path.join(self.empty_scene_dir, layout_items["scene_id"] + ".glb")
            )  # Expected: {'points': tensor, 'colors': tensor}
        else:
            raw_scene_data = load_point_cloud_from_glb(
                os.path.join(self.empty_scene_dir, scene_id + ".glb")
            )

            # Handle Scene Points Rotation
        scene_pts = raw_scene_data["points"]  # Tensor or Numpy
        if isinstance(scene_pts, torch.Tensor):
            scene_pts = scene_pts.numpy()
        scene_pts_aug = scene_pts @ R_scene.T
        raw_scene_data["points"] = torch.from_numpy(scene_pts_aug).float()
        s_min = scene_pts_aug.min(axis=0)
        s_max = scene_pts_aug.max(axis=0)
        curr_scene_size = torch.Tensor(s_max - s_min)  # [Lx, Ly, Lz]
        raw_scene_data["scene_size"] = curr_scene_size

        raw_asset_data_list = []
        raw_asset_dense_list = []
        bboxes = []
        descriptions = []
        asset_rot_rad_list = []
        zy_preprocesses = []

        if self.scene_is_m3dlayout:
            center_offset_arr = np.array(layout_items.get("center_offset", [0, 0, 0]))

        for item in layout_items["objects"]:
            # 1. Load Raw Labels
            size = item["size"]  # [x, y, z, lx, ly, lz, ro]
            pos = item["pos"]
            euler_rot = item["euler"]
            euler_rot = [i * np.pi / 180 for i in euler_rot]
            model_uid = item["jid"]
            short_desc = item.get("short_desc", "")

            if self.scene_is_m3dlayout:
                pos = (np.array(pos) + center_offset_arr).tolist()

                # ==========================================
                # 3. Label Augmentation Calculation
                # ==========================================
                # A. Position: Apply Scene Rotation

            center_pos = np.array(pos)

            center_pos_aug = R_scene @ center_pos

            # -> Apply Scene Rotation (+ Delta)

            if self.scene_is_m3dlayout:
                # M3DLayout: Z-up yaw
                final_zaw = euler_rot[2] + scene_rot_rad
            else:
                final_zaw = euler_rot[-1] + scene_rot_rad

                # ==========================================
                # 4. Asset Input Augmentation (Asset Rotation)
                # ==========================================
            asset_path = self.get_asset_path(model_uid)

            pc_data_dict = load_point_cloud_from_glb(
                asset_path, num_points=self.num_asset_points
            )

            if self.scene_is_m3dlayout:

                points_norm = pc_data_dict["points"]
            else:
                transform_matrix = np.array(
                    [
                        [1, 0, 0],  # x' = x
                        [0, 0, -1],  # y' = -z
                        [0, 1, 0],  # z' = y
                    ],
                    dtype=np.float32,
                )
                points_norm = pc_data_dict["points"] @ transform_matrix.T
            if isinstance(points_norm, np.ndarray):
                points_norm = torch.from_numpy(points_norm)

            raw_aabb_min = points_norm.min(dim=0).values
            raw_aabb_max = points_norm.max(dim=0).values
            current_size = raw_aabb_max - raw_aabb_min

            current_size = torch.clamp(current_size, min=1e-4)

            if self.scene_is_m3dlayout:
                target_size = torch.tensor(size, dtype=torch.float32)
            else:
                target_size = torch.tensor(size, dtype=torch.float32)
            scale_factors = target_size / current_size
            # print("scale_factors:", scale_factors)

            scale_factors = torch.clamp(scale_factors, max=1000.0)

            S = scale_matrix(scale_factors[0], scale_factors[1], scale_factors[2])[
                :3, :3
            ]
            points = points_norm @ S.T

            if self.scene_is_m3dlayout:

                R1 = trimesh.transformations.euler_matrix(
                    euler_rot[0], euler_rot[1], 0, axes="rxyz"
                )[:3, :3]
            else:
                R1 = trimesh.transformations.euler_matrix(
                    euler_rot[0], euler_rot[1], 0, axes="rxyz"
                )[:3, :3]

            points_canon = points @ R1.T

            dist_sq = points_canon.pow(2).sum(dim=-1)
            # max_dist = sqrt(max(dist_sq))
            rot_scale = dist_sq.max().sqrt()

            if rot_scale < 1e-6:
                # print(f"[Warning] Degenerate mesh detected for {model_uid}, scale is 0.")
                rot_scale = torch.tensor(1.0)
                points_canon = torch.zeros_like(points_canon)
            elif torch.isinf(rot_scale) or torch.isnan(rot_scale):
                # print(f"[Warning] Inf/NaN scale detected for {model_uid}.")
                rot_scale = torch.tensor(1.0)
                points_canon = torch.clamp(points_canon, min=-10.0, max=10.0)

            points_canon = points_canon / (rot_scale + 1e-8)

            zy_preprocesses.append(
                [
                    euler_rot[0],
                    euler_rot[1],
                    scale_factors[0],
                    scale_factors[1],
                    scale_factors[2],
                    size[0],
                    size[1],
                    size[2],
                    rot_scale,
                ]
            )

            # print(type(points_canon))

            # [Step 2] Asset Input Rotation (Data Augmentation)
            if do_asset_rot:

                asset_aug_deg = np.random.choice([0, 90, 180, 270])
                asset_aug_rad = np.deg2rad(asset_aug_deg)

                # Construct Rotation Matrix (Y-Up)
                ca, sa = np.cos(asset_aug_rad), np.sin(asset_aug_rad)
                R_asset_aug = torch.tensor(
                    [[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]], dtype=torch.float32
                )

                # Apply to Input PC
                points_input = points_canon.float() @ R_asset_aug.T

                # -> Update Label Yaw (- Delta)

                final_zaw = final_zaw - asset_aug_rad

            else:

                points_input = points_canon
                asset_aug_rad = 0.0

            asset_rot_rad_list.append(float(asset_aug_rad))

            # Normalize Yaw to [-PI, PI]
            final_zaw = (final_zaw + np.pi) % (2 * np.pi) - np.pi

            # ==========================================
            # 5. Store Processed Data
            # ==========================================
            pc_data_dict["points"] = points_input.float()

            # Dense Sampling
            colors = pc_data_dict["colors"]
            points_dense = torch.concat([pc_data_dict["points"], colors], dim=-1)

            curr_n = points_dense.shape[0]
            if curr_n >= self.num_asset_points:
                indices = torch.randperm(curr_n)[: self.num_asset_points]
                sampled_points = points_dense[indices]
            else:
                padding = torch.zeros(self.num_asset_points - curr_n, 6)
                sampled_points = torch.cat([points_dense, padding], dim=0)

            raw_asset_data_list.append(pc_data_dict)
            raw_asset_dense_list.append(sampled_points)

            # Store New BBox
            new_bbox = list()
            new_bbox += center_pos_aug.tolist()
            new_bbox.append(rot_scale)
            new_bbox.append(float(final_zaw))

            bboxes.append(new_bbox)

            if self.use_desc:
                descriptions.append(short_desc)
            else:
                descriptions.append(self.asset_category_list.get(model_uid, "others"))

                # 6. Final Batch construction
        Ni = len(layout_items["objects"])
        req_id = torch.arange(Ni, dtype=torch.long)
        req_mask = torch.ones(Ni, dtype=torch.long)

        if Ni > 0:
            req_pos = torch.tensor(bboxes, dtype=torch.float32)
        else:
            req_pos = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "scene_id": scene_id,
            "scene_text": scene_text,
            "assets_list": layout_items_id,
            "descriptions": descriptions,
            "num_items": Ni,
            "raw_asset_data_list": raw_asset_data_list,
            "raw_asset_dense_list": raw_asset_dense_list,
            "raw_scene_data": raw_scene_data,
            "zy_preprocess": zy_preprocesses,
            "required_item_id": req_id,
            "required_item_mask": req_mask,
            "required_item_position": req_pos,
            # [New] Augmented Scene Size for dynamic binning
            "scene_size": curr_scene_size,
            "scene_rot_rad": float(scene_rot_rad),
            "asset_rot_rad_list": asset_rot_rad_list,
            "is_pristine": is_pristine,
            "sampled_topo_paths": (
                online_topo_paths
                if online_topo_paths
                else self._get_sampled_paths(scene_id)
            ),
        }


class LayoutLLMDataCollator:
    """Unified collator for both training and evaluation.

    - mode='train': stochastic sampling / type-token permutation / asset-desc dropout
                    are honored exactly as before; full text contains the answer
                    so the LM can compute supervised loss.
    - mode='eval' : all augmentation is force-disabled; full text contains only
                    the prefix + 'Sure, here are ...' (no answer), matching the
                    previous LayoutLLMEvalDataCollator behavior so generation
                    happens at inference time.

    The previous LayoutLLMEvalDataCollator class is kept as a thin
    backward-compatible subclass below.
    """

    def __init__(
        self,
        tokenizer,
        special_token_map,
        max_length=2048,
        mode="train",
        shuffle_items=False,
        max_items=50,
        sampling_prob=0.8,
        anchor_threshold=0.5,
        asset_desc_dropout=True,
        permute_type_tokens=True,
        fixed_order=None,
    ):
        # If the legacy `fixed_order` kwarg is supplied, treat as eval mode for
        # exact backward compatibility with old call sites.
        if fixed_order is not None and mode == "train":
            mode = "eval"

        assert mode in ("train", "eval"), f"unknown mode: {mode}"
        self.mode = mode

        self.tokenizer = tokenizer
        self.special_token_map = special_token_map
        self.max_length = max_length
        self.shuffle_items = shuffle_items
        self.max_items = max_items
        self.asset_desc_dropout = asset_desc_dropout
        self.permute_type_tokens = permute_type_tokens
        # New Configs
        self.max_type_tokens = 50
        self.sampling_prob = sampling_prob
        self.anchor_threshold = anchor_threshold
        self.permute_type_tokens = permute_type_tokens
        self.fixed_order = (
            bool(fixed_order) if fixed_order is not None else (mode == "eval")
        )

        # --- Eval-mode invariants (mirror the old LayoutLLMEvalDataCollator) ---
        # Eval never shuffles items, never permutes type tokens, and never drops
        # asset descriptions. We force these regardless of how the caller set them
        # so behavior matches the previous separate eval collator exactly.
        if self.mode == "eval":
            self.shuffle_items = False
            self.permute_type_tokens = False
            self.asset_desc_dropout = False

        self.setup_tokens()

    def setup_tokens(self):

        # print("self.special_token_map:\n",self.special_token_map)
        self.ITEM_START = self.get_tok("ITEM_START")
        self.ITEM_END = self.get_tok("ITEM_END")
        self.PC_START = self.get_tok("PC_START")
        self.PC_END = self.get_tok("PC_END")
        self.PC_FEAT = self.get_tok("PC_FEAT")
        self.LOC_TOKEN = self.get_tok("POS_TOKEN")
        self.SCENE_START = self.get_tok("SCENE_START")
        self.SCENE_END = self.get_tok("SCENE_END")
        self.SCENE_FEATURE = self.get_tok("SCENE_FEATURE")
        self.G = 16
        self.pc_feats_placeholder = "".join([self.PC_FEAT] * self.G)
        self.scene_G = 64
        self.scene_feats_placeholder = "".join([self.SCENE_FEATURE] * self.scene_G)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def get_tok(self, name):
        return self.special_token_map.get(name, name)

    def get_type_tok(self, idx):
        return self.special_token_map.get(f"<TYPE_TOKEN_{idx}>", f"<TYPE_TOKEN_{idx}>")

    def __call__(self, batch_list):
        # Dispatch to the appropriate, behavior-identical implementation.
        if self.mode == "eval":
            return self._call_eval(batch_list)
        return self._call_train(batch_list)

    def _call_train(self, batch_list):

        batch_list = [x for x in batch_list if x is not None]

        K = self.max_items

        processed_batch_list = []

        for sample in batch_list:

            is_pristine = sample.get("is_pristine", False)

            do_shuffle = self.shuffle_items and (not is_pristine)
            do_permute = self.permute_type_tokens and (not is_pristine)
            do_dropout = self.asset_desc_dropout and (not is_pristine)

            # --------------------------------

            raw_pos = sample["required_item_position"]
            raw_mask = sample["required_item_mask"]
            raw_assets = sample["raw_asset_data_list"]
            raw_descs = sample["descriptions"]
            asset_id_list = sample["assets_list"]
            zy_preproceses = sample["zy_preprocess"]
            asset_rot_rad_list = sample["asset_rot_rad_list"]

            valid_indices = [i for i, m in enumerate(raw_mask) if m == 1]
            valid_set = set(valid_indices)

            def layout_priority_key(idx):
                zy = zy_preproceses[idx]
                volume = zy[5] * zy[6] * zy[7]
                return -volume

            topo_paths = sample.get("sampled_topo_paths", None)

            if topo_paths and len(topo_paths) > 0:
                chosen_path = random.choice(topo_paths)
                sorted_all_indices = [i for i in chosen_path if i in valid_set]
                path_set = set(sorted_all_indices)
                missing = [i for i in valid_indices if i not in path_set]
                if missing:
                    missing.sort(key=layout_priority_key)
                    sorted_all_indices.extend(missing)
            else:
                sorted_all_indices = sorted(valid_indices, key=layout_priority_key)

            num_total = len(sorted_all_indices)
            if num_total <= K:
                selected_indices = sorted_all_indices
            else:
                selected_indices = sorted_all_indices[:K]

            num_valid_items = len(selected_indices)

            if num_valid_items == 0:
                continue

            local_indices = list(range(num_valid_items))
            if do_shuffle:
                random.shuffle(local_indices)
                prefix_perm = local_indices
            else:
                prefix_perm = local_indices

            logical_idx_to_mapped_id = {}
            for prefix_pos, ideal_k in enumerate(prefix_perm):
                logical_idx_to_mapped_id[ideal_k] = prefix_pos

            final_assets_list = []
            final_descriptions = []
            final_id_asset_list = []
            final_dense_list = []
            final_zy_preprocess = []
            final_asset_rot_rad_list = []
            final_mapped_type_ids_for_prefix = []

            ideal_k_to_prefix_id = {}

            for prefix_pos, ideal_k in enumerate(prefix_perm):

                raw_idx = selected_indices[ideal_k]

                final_assets_list.append(raw_assets[raw_idx])
                final_descriptions.append(raw_descs[raw_idx])
                final_id_asset_list.append(asset_id_list[raw_idx])
                final_dense_list.append(sample["raw_asset_dense_list"][raw_idx])
                final_zy_preprocess.append(zy_preproceses[raw_idx])
                final_asset_rot_rad_list.append(asset_rot_rad_list[raw_idx])

                mapped_tid = logical_idx_to_mapped_id[ideal_k]
                final_mapped_type_ids_for_prefix.append(mapped_tid)

                ideal_k_to_prefix_id[ideal_k] = prefix_pos

            final_req_target_prefix_indices = []
            final_req_pos = []
            final_req_mapped_type_ids = []

            for k in range(num_valid_items):

                target_prefix_idx = ideal_k_to_prefix_id[k]
                final_req_target_prefix_indices.append(target_prefix_idx)

                final_req_mapped_type_ids.append(logical_idx_to_mapped_id[k])

                # 3. Position
                raw_idx = selected_indices[k]
                final_req_pos.append(raw_pos[raw_idx])

            new_sample = sample.copy()
            new_sample["raw_asset_data_list"] = final_assets_list
            new_sample["raw_asset_dense_list"] = final_dense_list
            new_sample["descriptions"] = final_descriptions
            new_sample["num_items"] = num_valid_items
            new_sample["zy_preprocess"] = final_zy_preprocess
            new_sample["asset_rot_rad_list"] = final_asset_rot_rad_list

            new_sample["_prefix_type_ids"] = final_mapped_type_ids_for_prefix

            # Answer:

            new_sample["required_item_id"] = torch.tensor(
                final_req_mapped_type_ids, dtype=torch.long
            )
            new_sample["required_item_position"] = torch.stack(final_req_pos, dim=0)
            new_sample["required_item_mask"] = torch.ones(
                num_valid_items, dtype=torch.long
            )
            new_sample["assets_list"] = final_id_asset_list
            new_sample["_do_dropout"] = do_dropout

            processed_batch_list.append(new_sample)

        batch_list = processed_batch_list

        # =========================================================================

        # =========================================================================

        prefix_texts = []
        full_texts = []
        req_pos_residuals_list = []
        scene_sizes_list = []

        for sample in batch_list:
            desc_list = sample["descriptions"]
            num_items = sample["num_items"]
            scene_text = sample["scene_text"]

            prefix_type_ids = sample["_prefix_type_ids"]

            enable_dropout = sample.get("_do_dropout", False)

            items_strs = []
            for i in range(num_items):

                mapped_tid = prefix_type_ids[i]
                type_tok = self.get_type_tok(mapped_tid)

                desc = desc_list[i]
                if enable_dropout:
                    select_list = [f"Description {desc}", ""]
                    selected_desc = random.choice(select_list)
                    s = f"{self.ITEM_START} ID {type_tok} Feature {self.PC_START}{self.pc_feats_placeholder}{self.PC_END}{selected_desc}{self.ITEM_END}"
                else:
                    s = f"{self.ITEM_START} ID {type_tok} Feature {self.PC_START}{self.pc_feats_placeholder}{self.PC_END} Description {desc}{self.ITEM_END}"
                items_strs.append(s)

            scene_req = f"The description of the scene is {scene_text}. And the 3D feature of the empty scene is {self.SCENE_START}{self.scene_feats_placeholder}{self.SCENE_END}.\n"
            prefix = f"You are a reliable layout designer. I want create a scene using given assets. \n{scene_req} Here are some items you could use: {''.join(items_strs)}"
            prefix_texts.append(prefix)

            # Scene Size
            current_scene_size = sample["raw_scene_data"]["scene_size"]
            scene_sizes_list.append(current_scene_size)

            req_ids = sample["required_item_id"]
            raw_positions = sample["required_item_position"]

            ans_parts = ["Sure, here are the required assets and their poses:"]
            sample_residuals = []

            for i, rid in enumerate(req_ids):

                type_tok = self.get_type_tok(int(rid.item()))

                curr_pos = raw_positions[i]
                bin_idxs, residuals = quantize_pose(
                    curr_pos, current_scene_size, num_bins=20
                )
                sample_residuals.append(torch.tensor(residuals, dtype=torch.float32))

                bin_x_str = self.special_token_map[f"<POS_BIN_{bin_idxs[0]}>"]
                bin_y_str = self.special_token_map[f"<POS_BIN_{bin_idxs[1]}>"]
                bin_z_str = self.special_token_map[f"<POS_BIN_{bin_idxs[2]}>"]
                bin_rot_str = self.special_token_map[f"<ROT_BIN_{bin_idxs[3]}>"]

                ans_parts.append(
                    f"{self.ITEM_START} ID {type_tok} Position {bin_x_str}{bin_y_str}{bin_z_str}{bin_rot_str}{self.LOC_TOKEN}{self.ITEM_END}"
                )

            req_pos_residuals_list.append(torch.stack(sample_residuals))

            full_text = prefix + "".join(ans_parts) + self.tokenizer.eos_token
            full_texts.append(full_text)

        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        prefix_enc = self.tokenizer(
            prefix_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_lens = prefix_enc["attention_mask"].sum(dim=1)

        labels = torch.full_like(input_ids, -100)
        for b in range(len(batch_list)):
            pl = prefix_lens[b]
            sl = attention_mask[b].sum()
            if pl < sl:
                labels[b, pl:sl] = input_ids[b, pl:sl]

        all_asset_dicts_flat = []
        counts_per_sample = []
        all_asset_dense_flat = []  # New

        for sample in batch_list:
            assets_list = sample["raw_asset_data_list"]
            assets_dense = sample["raw_asset_dense_list"]
            counts_per_sample.append(len(assets_list))
            all_asset_dicts_flat.extend(assets_list)
            all_asset_dense_flat.extend(assets_dense)

        batched_asset_sparse_input = collate_point_clouds_to_sparse_input(
            all_asset_dicts_flat, voxel_size=0.05
        )

        if len(all_asset_dense_flat) > 0:
            batched_asset_dense_input = torch.stack(all_asset_dense_flat, dim=0)
        else:
            batched_asset_dense_input = torch.zeros(0, 4096, 3)

        scene_dicts = [
            s["raw_scene_data"] for s in batch_list if s["raw_scene_data"] is not None
        ]
        batched_scene_sparse_input = collate_point_clouds_to_sparse_input(
            scene_dicts, voxel_size=0.05
        )

        num_items = torch.tensor([s["num_items"] for s in batch_list], dtype=torch.long)

        req_id_list = [s["required_item_id"] for s in batch_list]
        req_mask_list = [s["required_item_mask"] for s in batch_list]
        req_pos_list = [s["required_item_position"] for s in batch_list]
        asset_id_list = [s["assets_list"] for s in batch_list]
        scene_id_list = [s["scene_id"] for s in batch_list]
        zy_preprocess_list = [s["zy_preprocess"] for s in batch_list]
        asset_rot_rad_list_all = [s["asset_rot_rad_list"] for s in batch_list]
        scene_rot_rad_list = [s["scene_rot_rad"] for s in batch_list]

        req_id_padded = torch.nn.utils.rnn.pad_sequence(
            req_id_list, batch_first=True, padding_value=0
        )
        req_mask_padded = torch.nn.utils.rnn.pad_sequence(
            req_mask_list, batch_first=True, padding_value=0
        )
        req_pos_padded = torch.nn.utils.rnn.pad_sequence(
            req_pos_list, batch_first=True, padding_value=0
        )
        req_res_pos_padded = torch.nn.utils.rnn.pad_sequence(
            req_pos_residuals_list, batch_first=True, padding_value=0
        )
        is_pristine_list = [s.get("is_pristine", False) for s in batch_list]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_lens": prefix_lens,
            "batch_processed": {
                "scene_id_list": scene_id_list,
                "scene_sizes": scene_sizes_list,
                "scenes": batched_scene_sparse_input,
                "assets": batched_asset_sparse_input,
                "assets_dense": batched_asset_dense_input,
                "asset_counts": counts_per_sample,
                "asset_id_list": asset_id_list,
                "zy_preprocess": zy_preprocess_list,
                "asset_rot_rad_list": asset_rot_rad_list_all,
                "scene_rot_rad_list": scene_rot_rad_list,
                "num_items": num_items,
                "required_item_id": req_id_padded,
                "required_item_mask": req_mask_padded,
                "required_item_position": req_pos_padded,
                "required_item_res_pos": req_res_pos_padded,
                "required_item_num": req_mask_padded.sum(dim=1),
                "_prefix_type_ids": [s["_prefix_type_ids"] for s in batch_list],
                "is_pristine": is_pristine_list,
            },
        }

    def _call_eval(self, batch_list):
        batch_list = [x for x in batch_list if x is not None]
        if len(batch_list) == 0:
            return None

        K = self.max_items
        processed_batch_list = []

        for sample in batch_list:
            raw_pos = sample["required_item_position"]
            raw_mask = sample["required_item_mask"]
            raw_assets = sample["raw_asset_data_list"]
            raw_descs = sample["descriptions"]
            asset_id_list = sample["assets_list"]
            zy_preproceses = sample["zy_preprocess"]
            asset_rot_rad_list = sample["asset_rot_rad_list"]

            valid_indices = [i for i, m in enumerate(raw_mask) if m == 1]
            valid_set = set(valid_indices)

            def layout_priority_key(idx):
                zy = zy_preproceses[idx]
                volume = zy[5] * zy[6] * zy[7]
                return -volume

            topo_paths = sample.get("sampled_topo_paths", None)

            if topo_paths and len(topo_paths) > 0:
                chosen_path = topo_paths[0]
                sorted_all_indices = [i for i in chosen_path if i in valid_set]
                path_set = set(sorted_all_indices)
                missing = [i for i in valid_indices if i not in path_set]
                if missing:
                    missing.sort(key=layout_priority_key)
                    sorted_all_indices.extend(missing)
            else:
                sorted_all_indices = sorted(valid_indices, key=layout_priority_key)

            if len(sorted_all_indices) > K:
                selected_indices = sorted_all_indices[:K]
            else:
                selected_indices = sorted_all_indices

            num_valid_items = len(selected_indices)

            if num_valid_items == 0:
                continue

            prefix_perm = list(range(num_valid_items))

            # --- [Step 4: Type Mapping (Deterministic)] ---

            logical_idx_to_mapped_id = {i: i for i in range(num_valid_items)}

            final_assets_list = []
            final_descriptions = []
            final_id_asset_list = []
            final_dense_list = []
            final_zy_preprocess = []
            final_asset_rot_rad_list = []

            final_mapped_type_ids_for_prefix = []

            ideal_k_to_prefix_id = {}

            for prefix_pos, ideal_k in enumerate(prefix_perm):
                raw_idx = selected_indices[ideal_k]

                final_assets_list.append(raw_assets[raw_idx])
                final_descriptions.append(raw_descs[raw_idx])
                final_id_asset_list.append(asset_id_list[raw_idx])
                final_dense_list.append(sample["raw_asset_dense_list"][raw_idx])
                final_zy_preprocess.append(zy_preproceses[raw_idx])
                final_asset_rot_rad_list.append(asset_rot_rad_list[raw_idx])

                # Mapping
                mapped_tid = logical_idx_to_mapped_id[ideal_k]
                final_mapped_type_ids_for_prefix.append(mapped_tid)

                ideal_k_to_prefix_id[ideal_k] = prefix_pos

            final_req_target_prefix_indices = []
            final_req_pos = []
            final_req_mapped_type_ids = []

            for k in range(num_valid_items):
                target_prefix_idx = ideal_k_to_prefix_id[k]
                final_req_target_prefix_indices.append(target_prefix_idx)

                final_req_mapped_type_ids.append(logical_idx_to_mapped_id[k])

                raw_idx = selected_indices[k]
                final_req_pos.append(raw_pos[raw_idx])

            new_sample = sample.copy()
            new_sample["raw_asset_data_list"] = final_assets_list
            new_sample["raw_asset_dense_list"] = final_dense_list
            new_sample["descriptions"] = final_descriptions
            new_sample["num_items"] = num_valid_items
            new_sample["zy_preprocess"] = final_zy_preprocess
            new_sample["asset_rot_rad_list"] = final_asset_rot_rad_list

            new_sample["_prefix_type_ids"] = final_mapped_type_ids_for_prefix

            new_sample["required_item_id"] = torch.tensor(
                final_req_mapped_type_ids, dtype=torch.long
            )
            new_sample["required_item_position"] = torch.stack(final_req_pos, dim=0)
            new_sample["required_item_mask"] = torch.ones(
                num_valid_items, dtype=torch.long
            )
            new_sample["assets_list"] = final_id_asset_list

            processed_batch_list.append(new_sample)

        batch_list = processed_batch_list

        prefix_texts = []
        full_texts = []
        req_pos_residuals_list = []
        scene_sizes_list = []

        for sample in batch_list:
            desc_list = sample["descriptions"]
            num_items = sample["num_items"]
            scene_text = sample["scene_text"]
            prefix_type_ids = sample["_prefix_type_ids"]

            items_strs = []
            for i in range(num_items):
                mapped_tid = prefix_type_ids[i]
                type_tok = self.get_type_tok(mapped_tid)
                desc = desc_list[i]

                s = f"{self.ITEM_START} ID {type_tok} Feature {self.PC_START}{self.pc_feats_placeholder}{self.PC_END} Description {desc}{self.ITEM_END}"
                items_strs.append(s)

            scene_req = f"The description of the scene is {scene_text}. And the 3D feature of the empty scene is {self.SCENE_START}{self.scene_feats_placeholder}{self.SCENE_END}.\n"

            prefix = f"You are a reliable layout designer. I want Screate a scene using given assets. \n{scene_req} Here are some items you could use: {''.join(items_strs)}"

            current_scene_size = sample["raw_scene_data"]["scene_size"]
            scene_sizes_list.append(current_scene_size)

            req_ids = sample["required_item_id"]
            raw_positions = sample["required_item_position"]

            ans_parts = ["Sure, here are the required assets and their poses:"]
            sample_residuals = []

            for i, rid in enumerate(req_ids):
                type_tok = self.get_type_tok(int(rid.item()))

                curr_pos = raw_positions[i]
                bin_idxs, residuals = quantize_pose(
                    curr_pos, current_scene_size, num_bins=20
                )
                sample_residuals.append(torch.tensor(residuals, dtype=torch.float32))

                bin_x_str = self.special_token_map[f"<POS_BIN_{bin_idxs[0]}>"]
                bin_y_str = self.special_token_map[f"<POS_BIN_{bin_idxs[1]}>"]
                bin_z_str = self.special_token_map[f"<POS_BIN_{bin_idxs[2]}>"]
                bin_rot_str = self.special_token_map[f"<ROT_BIN_{bin_idxs[3]}>"]

                ans_parts.append(
                    f"{self.ITEM_START} ID {type_tok} Position {bin_x_str}{bin_y_str}{bin_z_str}{bin_rot_str}{self.LOC_TOKEN}{self.ITEM_END}"
                )

            req_pos_residuals_list.append(torch.stack(sample_residuals))

            full_text = prefix + "Sure, here are the required assets and their poses:"
            # + "".join(ans_parts) + self.tokenizer.eos_token
            full_texts.append(full_text)
            prefix_texts.append(prefix)

        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        prefix_enc = self.tokenizer(
            prefix_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_lens = prefix_enc["attention_mask"].sum(dim=1)

        # Labels! (Important for Loss)
        labels = torch.full_like(input_ids, -100)
        for b in range(len(batch_list)):
            pl = prefix_lens[b]
            sl = attention_mask[b].sum()
            if pl < sl:
                labels[b, pl:sl] = input_ids[b, pl:sl]

        all_asset_dicts_flat = []
        counts_per_sample = []
        all_asset_dense_flat = []
        for sample in batch_list:
            all_asset_dicts_flat.extend(sample["raw_asset_data_list"])
            all_asset_dense_flat.extend(sample["raw_asset_dense_list"])
            counts_per_sample.append(len(sample["raw_asset_data_list"]))

        batched_asset_sparse_input = (
            collate_point_clouds_to_sparse_input(all_asset_dicts_flat, voxel_size=0.05)
            if all_asset_dicts_flat
            else None
        )
        batched_asset_dense_input = (
            torch.stack(all_asset_dense_flat, dim=0)
            if all_asset_dense_flat
            else torch.zeros(0, 4096, 6)
        )

        scene_dicts = [s["raw_scene_data"] for s in batch_list if s["raw_scene_data"]]
        batched_scene_sparse_input = collate_point_clouds_to_sparse_input(
            scene_dicts, voxel_size=0.05
        )

        # Meta wrap
        num_items = torch.tensor([s["num_items"] for s in batch_list], dtype=torch.long)
        req_id_padded = torch.nn.utils.rnn.pad_sequence(
            [s["required_item_id"] for s in batch_list],
            batch_first=True,
            padding_value=0,
        )
        req_mask_padded = torch.nn.utils.rnn.pad_sequence(
            [s["required_item_mask"] for s in batch_list],
            batch_first=True,
            padding_value=0,
        )
        req_pos_padded = torch.nn.utils.rnn.pad_sequence(
            [s["required_item_position"] for s in batch_list],
            batch_first=True,
            padding_value=0,
        )
        req_res_pos_padded = torch.nn.utils.rnn.pad_sequence(
            req_pos_residuals_list, batch_first=True, padding_value=0
        )
        is_pristine_list = [True] * len(batch_list)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_lens": prefix_lens,
            "batch_processed": {
                "scene_id_list": [s["scene_id"] for s in batch_list],
                "scene_sizes": scene_sizes_list,
                "scenes": batched_scene_sparse_input,
                "assets": batched_asset_sparse_input,
                "assets_dense": batched_asset_dense_input,
                "asset_counts": counts_per_sample,
                "asset_id_list": [s["assets_list"] for s in batch_list],
                "zy_preprocess": [s["zy_preprocess"] for s in batch_list],
                "asset_rot_rad_list": [s["asset_rot_rad_list"] for s in batch_list],
                "scene_rot_rad_list": [s["scene_rot_rad"] for s in batch_list],
                "num_items": num_items,
                "required_item_id": req_id_padded,
                "required_item_mask": req_mask_padded,
                "required_item_position": req_pos_padded,
                "required_item_res_pos": req_res_pos_padded,
                "required_item_num": req_mask_padded.sum(dim=1),
                "_prefix_type_ids": [s["_prefix_type_ids"] for s in batch_list],
                "is_pristine": is_pristine_list,  # [Added for compatibility]
            },
        }


class LayoutLLMEvalDataCollator(LayoutLLMDataCollator):
    """Backward-compat thin wrapper over LayoutLLMDataCollator(mode='eval').

    Existing call sites that do
        LayoutLLMEvalDataCollator(tokenizer, special_token_map, max_length=...,
                                  fixed_order=True, max_items=30,
                                  asset_desc_dropout=False)
    keep working with identical behavior.  New code is encouraged to use
        LayoutLLMDataCollator(..., mode='eval')
    directly.
    """

    def __init__(
        self,
        tokenizer,
        special_token_map,
        max_length=2048,
        fixed_order=True,
        max_items=30,
        asset_desc_dropout=False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            special_token_map=special_token_map,
            max_length=max_length,
            mode="eval",
            shuffle_items=False,
            max_items=max_items,
            asset_desc_dropout=asset_desc_dropout,
            permute_type_tokens=False,
            fixed_order=fixed_order,
        )
