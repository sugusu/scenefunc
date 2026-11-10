# from pointbert.point_encoder import PointTransformer
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
import os
from pathlib import Path
from nala.models.point_encoder.PointEncoder import SceneQueryExtractor
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
import trimesh
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy
import trimesh.transformations as tra

# import flash_attn
import random
import json

import os
import json
import numpy as np
import pybullet as p
import trimesh
import math
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


SIMULATION_STEPS = 500
GRAVITY = -9.8
WALL_HEIGHT_PADDING = 0.5


def create_boundary_walls(boundary_points, scene_height, thickness=0.1):
    """"""
    if len(boundary_points) < 3:
        return
    half_height = scene_height / 2.0 + WALL_HEIGHT_PADDING

    for i in range(len(boundary_points)):
        p1 = boundary_points[i]
        p2 = boundary_points[(i + 1) % len(boundary_points)]
        vp1 = np.array(p1)
        vp2 = np.array(p2)
        center_2d = (vp1 + vp2) / 2.0
        vec = vp2 - vp1
        length = np.linalg.norm(vec)
        if length < 1e-4:
            continue
        angle = math.atan2(vec[1], vec[0])
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, thickness / 2.0, half_height]
        )
        p.createMultiBody(
            0,
            col_shape,
            basePosition=[center_2d[0], center_2d[1], 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle]),
        )


def create_floor_mesh(boundary_points, scene_height):
    """"""
    try:
        from shapely.geometry import Polygon

        poly = Polygon(boundary_points)
        mesh = trimesh.creation.triangulate_polygon(poly)

        if isinstance(mesh, tuple):
            mesh = trimesh.Trimesh(vertices=mesh[0], faces=mesh[1])

        if mesh.vertices.shape[1] == 2:
            vertices_3d = np.hstack(
                [mesh.vertices, np.zeros((mesh.vertices.shape[0], 1))]
            )
            mesh.vertices = vertices_3d

        floor_z = -scene_height / 2.0
        matrix = trimesh.transformations.translation_matrix([0, 0, floor_z])
        mesh.apply_transform(matrix)

        mesh.visual.face_colors = [200, 200, 200, 255]
        return mesh

    except Exception as e:
        print(f"[Warn] Failed to create floor mesh: {e}")
        return None


def process_scene_physics(scene_dir):
    json_path = os.path.join(scene_dir, "layout_physics_info.json")
    glb_path = os.path.join(scene_dir, "layout_wt_scene.glb")
    output_glb_path = os.path.join(scene_dir, "layout_fixed.glb")
    output_json_path = os.path.join(scene_dir, "layout_adjusted.json")

    if not os.path.exists(json_path) or not os.path.exists(glb_path):
        return

    print(f"🔧 Processing: {scene_dir}")
    with open(json_path, "r") as f:
        data = json.load(f)

    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, GRAVITY)

    scene_h = data.get("scene_height", 2.8)
    if isinstance(scene_h, list):
        scene_h = scene_h[0]

    floor_z = -scene_h / 2.0
    p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[100, 100, 0.1]),
        basePosition=[0, 0, floor_z - 0.1],
    )
    p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[100, 100, 0.1]),
        basePosition=[0, 0, scene_h / 2.0 + 0.1],
    )

    boundary = data.get("boundary", [])
    if boundary:
        create_boundary_walls(boundary, scene_h)

    body_map = {}
    for obj in data["objects"]:
        extent = obj["extents"]
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[e / 2.0 for e in extent]
        )

        init_pos = list(obj["initial_pos"])
        init_pos[2] += 0.02

        body_id = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=col_shape,
            basePosition=init_pos,
            baseOrientation=obj["initial_quat"],
        )

        p.changeDynamics(
            body_id,
            -1,
            linearDamping=0.9,
            angularDamping=1.0,
            restitution=0.0,
            localInertiaDiagonal=[1e20, 1e20, 1e20],
        )
        body_map[body_id] = obj

    for _ in range(SIMULATION_STEPS):
        p.stepSimulation()

    trimesh_scene = trimesh.load(glb_path, force="scene")

    adjusted_record = []

    for body_id, obj_info in body_map.items():
        new_pos, _ = p.getBasePositionAndOrientation(body_id)

        c = np.array(obj_info["initial_pos"])
        t_target = np.array(new_pos)
        translation = t_target - c
        final_transform_delta = trimesh.transformations.translation_matrix(translation)

        node_name = obj_info["node_name"]

        if node_name in trimesh_scene.geometry:
            geom = trimesh_scene.geometry[node_name]

            geom.apply_transform(final_transform_delta)

            adjusted_record.append({"node_name": node_name, "final_pos": new_pos})

    p.disconnect()

    if boundary:
        floor_mesh = create_floor_mesh(boundary, scene_h)
        if floor_mesh:
            trimesh_scene.add_geometry(floor_mesh, geom_name="generated_floor")

    matrix_y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

    # trimesh_scene.apply_transform(matrix_y_up)

    trimesh_scene.apply_transform(matrix_y_up)

    trimesh_scene.export(output_glb_path)
    print(f"✅ Saved Y-up scene with floor: {output_glb_path}")

    with open(output_json_path, "w") as f:
        json.dump(adjusted_record, f, indent=4)

        # os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"


def scale_matrix(sx, sy, sz):
    S = np.eye(4)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz
    return S


def translation_matrix(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def get_unified_data(rot):

    raw_rz = rot[0]
    raw_rx = rot[1]
    raw_ry = rot[2]

    # Axis_Vec = Rz @ [1,0,0]
    axis_vec = tra.euler_matrix(raw_rz, 0, 0, axes="rzxy")[:3, :3] @ np.array([1, 0, 0])

    if np.abs(axis_vec[0]) > 0.9:
        sign = np.sign(axis_vec[0])

        target_mode = "x_to_y"
        target = raw_rx * sign

    elif np.abs(axis_vec[1]) > 0.9:
        sign = np.sign(axis_vec[1])
        target_mode = "y_to_y"
        target = raw_rx * sign
    else:
        return None

    R_pred_ideal = tra.euler_matrix(0, 0, target, axes="rzxy")[:3, :3]

    R_total = tra.euler_matrix(raw_rz, raw_rx, raw_ry, axes="rzxy")[:3, :3]

    R_fix = R_pred_ideal.T @ R_total

    return R_fix, target

    # necessary parameters for llm


class model_config:
    def __init__(
        self,
        device: str,
        item_position_dim: int,
        pos_decoder_layer: int,
        pos_decoder_dim: List[int],
        token_3D_start_end: bool,
        token_item_start_end: bool,
        hidden_size: int,
        special_tokens: Optional[Dict[str, str]] = None,
        loss_weight: torch.FloatTensor = None,
        load_model: bool = None,
        asset_encoder_type: str = "pointbert",
        asset_encoder_ckpt_path: Optional[str] = None,
    ):
        self.device = device
        self.token_3D_start_end = token_3D_start_end
        self.token_item_start_end = token_item_start_end
        self.item_position_dim = item_position_dim

        self.hidden_size = hidden_size

        self.pos_decoder_layer = pos_decoder_layer
        self.pos_decoder_dim = pos_decoder_dim
        self.load_model = load_model
        self.asset_encoder_type = asset_encoder_type
        self.asset_encoder_ckpt_path = asset_encoder_ckpt_path

        if special_tokens is None:
            self.special_tokens = {
                "ITEM_START": "ITEM_START",
                "PC_START": "PC_START",
                "PC_END": "PC_END",
                "ITEM_END": "ITEM_END",
                "PC_FEAT": "PC_FEAT",
                "POS_TOKEN": "POS_TOKEN",
                "TYPE_TOKEN": "TYPE_TOKEN",
                "SCENE_START": "SCENE_START",
                "SCENE_END": "SCENE_END",
                "SCENE_FEATURE": "SCENE_FEATURE",
                # "SCENE_FEAT":"<SCENE_FEAT>"
            }
        else:
            self.special_tokens = special_tokens
        self.loss_weight = loss_weight


class point_backbone_config:
    def __init__(
        self,
        model_name: str,
        load_model: bool,
        model_dir: str,
        trans_dim: int,
        depth: int,
        drop_path_rate: float,
        cls_dim: int,
        num_heads: int,
        group_size: int,
        num_group: int,
        point_dims: int,
        encoder_dims: int,
        scene_dim: int,
        box_dim: int,
        box_hidden_layer: int,
        box_hidden_dim: List[int],
        projection_hidden_layer: int,
        use_max_pool: bool,
        projection_hidden_dim: List[int],
        use_color: bool,
    ):
        if projection_hidden_layer != len(projection_hidden_dim):
            raise ValueError(
                "The length of projection_hidden_dim should be the same as projection_hidden_layer."
            )

        self.model_name = model_name
        self.load_model = load_model
        self.model_dir = model_dir

        # parameter for point bert
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.point_dims = point_dims if not use_color else point_dims + 3
        self.encoder_dims = encoder_dims
        self.use_max_pool = use_max_pool

        self.scene_dim = scene_dim

        # parameter after box mlp
        self.box_dim = box_dim
        self.box_hidden_layer = box_hidden_layer
        self.box_hidden_dim = box_hidden_dim

        # parameter after point bert
        self.backbone_output_dim = trans_dim if not use_max_pool else trans_dim * 2
        self.point_token_len = num_group + 1 if not use_max_pool else 1

        self.projection_hidden_layer = projection_hidden_layer
        self.projection_hidden_dim = projection_hidden_dim


class AdapterLMHead(nn.Module):
    """
    Wrapper lm_head that composes a base lm_head and a small adapter
    that only models logits for a small set of new token ids.

    base_lm_head: existing Linear(embed_dim, vocab_size) or any callable module(samples->logits)
    new_token_ids: list[int]  (target vocab ids to augment)
    new_emb: nn.Parameter of shape (K, E)  (or None to create inside)
    new_bias: nn.Parameter of shape (K,)
    """

    def __init__(
        self,
        base_lm_head: nn.Module,
        new_token_ids: list,
        new_emb: torch.nn.Parameter,
        new_bias: torch.nn.Parameter,
    ):
        super().__init__()
        self.base_lm_head = base_lm_head
        self.new_token_ids = list(new_token_ids)
        self.K = len(self.new_token_ids)
        # print("\n----------K----------\n",self.K,"-----------------------\n")
        self.new_lm_output_emb = new_emb  # shape (K, E)  (nn.Parameter)
        self.new_lm_output_bias = new_bias  # shape (K,)  (nn.Parameter)

        # ensure deterministic ordering and unique ids
        assert len(set(self.new_token_ids)) == len(
            self.new_token_ids
        ), "new_token_ids must be unique"

        # If base_lm_head has weight (Linear), freeze it by default (you can change later)
        try:
            if hasattr(self.base_lm_head, "weight"):
                self.base_lm_head.weight.requires_grad = False
        except Exception:
            pass

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, E)
        return logits: (B, L, V)
        """
        # 1) compute base logits (may be huge)
        logits = self.base_lm_head(hidden_states)  # (B, L, V)

        # 2) if no adapter, just return base logits
        if self.K == 0 or self.new_lm_output_emb is None:
            return logits

            # ensure new_emb/bias on same device/dtype as logits
        dev = logits.device
        dt = logits.dtype

        # cast adapter params
        new_emb = self.new_lm_output_emb.to(device=dev, dtype=dt)  # (K, E)
        new_bias = self.new_lm_output_bias.to(device=dev, dtype=dt)  # (K,)

        # compute adapter logits: (B,L,E) @ (E,K) -> (B,L,K)
        # cast hidden_states to dt
        hs = hidden_states.to(device=dev, dtype=dt)
        adapter_logits = torch.matmul(hs, new_emb.t())  # (B,L,K)
        adapter_logits = adapter_logits + new_bias.view(1, 1, -1)

        # add adapter logits into logits at new_token_ids
        # vectorized add: use index_add_ along vocab dim (dim=2)
        idxs = torch.tensor(self.new_token_ids, device=dev, dtype=torch.long)  # (K,)
        # index_add_ expects src shape (B, L, K) and will add along dim=2 into logits
        # but index_add_ modifies in-place
        logits.index_add_(2, idxs, adapter_logits)

        return logits

    def freeze_base(self, freeze=True):
        """Freeze/unfreeze base lm_head weights (if present)"""
        if hasattr(self.base_lm_head, "weight"):
            self.base_lm_head.weight.requires_grad = not freeze

    def get_adapter_params(self):
        """Return adapter parameters (new_emb and new_bias) for optimizer"""
        params = []
        if self.new_lm_output_emb is not None:
            params.append(self.new_lm_output_emb)
        if self.new_lm_output_bias is not None:
            params.append(self.new_lm_output_bias)
        return params

    def state_dict_adapter(self):
        """Return dict only for adapter params for saving"""
        sd = {}
        if self.new_lm_output_emb is not None:
            sd["new_lm_output_emb"] = self.new_lm_output_emb.detach().cpu()
        if self.new_lm_output_bias is not None:
            sd["new_lm_output_bias"] = self.new_lm_output_bias.detach().cpu()
        sd["new_token_ids"] = torch.tensor(self.new_token_ids, dtype=torch.long)
        return sd

    def load_adapter_state_dict(self, sd):
        """Load adapter state dict saved by state_dict_adapter"""
        if "new_lm_output_emb" in sd and self.new_lm_output_emb is not None:
            self.new_lm_output_emb.data.copy_(
                sd["new_lm_output_emb"].to(self.new_lm_output_emb.device)
            )
        if "new_lm_output_bias" in sd and self.new_lm_output_bias is not None:
            self.new_lm_output_bias.data.copy_(
                sd["new_lm_output_bias"].to(self.new_lm_output_bias.device)
            )
        if "new_token_ids" in sd:
            self.new_token_ids = sd["new_token_ids"].tolist()
            self.K = len(self.new_token_ids)


class MultiLayerPositionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.llm_embed_dim = config.hidden_size  # e.g. 4096
        self.decoder_dim = 512
        self.num_layers_to_use = 4
        self.n_heads = 8
        self.output_dim = config.item_position_dim

        self.layer_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.llm_embed_dim),
                    nn.Linear(self.llm_embed_dim, self.decoder_dim),
                    nn.GELU(),
                )
                for _ in range(self.num_layers_to_use)
            ]
        )

        self.aggregator = nn.MultiheadAttention(
            embed_dim=self.decoder_dim,
            num_heads=self.n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.agg_norm = nn.LayerNorm(self.decoder_dim)

        self.regressor = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.GELU(),
            nn.Linear(self.decoder_dim, self.output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, hidden_states_tuple):
        """
        Args:
            hidden_states_tuple: ...
        """

        projected_feats = []
        for i, feat in enumerate(hidden_states_tuple):

            adapter = self.layer_adapters[i]

            try:
                target_dtype = next(adapter.parameters()).dtype
            except StopIteration:
                target_dtype = torch.float32

            if feat.dtype != target_dtype:
                feat = feat.to(target_dtype)

            proj = adapter(feat)  # -> (N_pos, 512)

            projected_feats.append(proj.unsqueeze(1))

        query = projected_feats[-1]  # (N_pos, 1, 512)

        main_dtype = query.dtype
        context_list = []
        for p in projected_feats:
            if p.dtype != main_dtype:
                context_list.append(p.to(main_dtype))
            else:
                context_list.append(p)

        context = torch.cat(context_list, dim=1)

        # 3. Cross Attention

        agg_dtype = next(self.aggregator.parameters()).dtype
        if query.dtype != agg_dtype:
            query = query.to(agg_dtype)
        if context.dtype != agg_dtype:
            context = context.to(agg_dtype)

            # attn_output: (N_pos, 1, 512)
        attn_output, _ = self.aggregator(query, context, context)

        # Residual + Norm

        res = query + attn_output

        norm_dtype = self.agg_norm.weight.dtype
        if res.dtype != norm_dtype:
            res = res.to(norm_dtype)

        x = self.agg_norm(res).squeeze(1)  # (N_pos, 512)

        # 4. Regression

        reg_dtype = next(self.regressor.parameters()).dtype
        if x.dtype != reg_dtype:
            x = x.to(reg_dtype)

        out = self.regressor(x)  # (N_pos, 9)
        out = out.to(torch.bfloat16)

        return out


class LayoutLLM(nn.Module):
    def __init__(
        self,
        base_model: Optional[Any] = None,
        base_model_name_or_path: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        config: model_config = None,
        point_backbone_config: point_backbone_config = None,
        model_load_kwargs: Optional[Dict] = None,
    ):
        super().__init__()

        # Set target device according to config
        self.device = torch.device(config.device)

        self.point_backbone_config = point_backbone_config
        self.config = config

        # Allow overriding the base model path and tokenizer externally
        if base_model_name_or_path is None:
            base_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

        cache_dir = os.environ.get(
            "HF_HOME",
            str(Path(__file__).resolve().parents[2].parent / ".cache" / "hf"),
        )
        os.makedirs(cache_dir, exist_ok=True)

        # Loading tokenizer: priority is external input, otherwise load from local path/cache
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path,
                cache_dir=cache_dir,
                use_fast=True,
                trust_remote_code=False,
                padding_side="right",
            )
        else:
            self.tokenizer = tokenizer

            # Ensure a pad token is always available; default to EOS if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load language model: priority is external input, otherwise from given path
        if base_model is None:
            self.base = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
                use_cache=False,
            )
        else:
            self.base = base_model

            # Qwen2.5 always has an EOS token, so use that as pad_token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base = self.base.to(self.device)

        # Optionally move model to desired device unless sharded (device_map="auto")
        target_device = getattr(self.config, "device", "cpu")
        try:
            if not hasattr(self.base, "is_loaded_in_8bit") or not getattr(
                self.base, "is_loaded_in_8bit"
            ):
                self.base.to(target_device)
        except Exception:
            # Fallback for sharded or special device_map behavior
            pass

            # Feature extractor for querying scene and assets
        asset_enc_type = getattr(config, "asset_encoder_type", "pointbert")
        asset_enc_ckpt = getattr(config, "asset_encoder_ckpt_path", None)
        self.extractor = SceneQueryExtractor(
            num_scene_query_token=64,
            num_asset_query_token=16,
            asset_encoder_type=asset_enc_type,
            asset_encoder_ckpt_path=asset_enc_ckpt,
            device=self.device,
        ).to(self.device)

        # The following lines show where Qformer weights can be loaded if necessary:
        # self.extractor.load_qformer_weights("/path/to/qformer_weights.pth")
        # self.extractor.add_lora_to_qformers()

        backbone_output_dim = 128

        # Define a projection head for point-based feature representation,
        # with optional hidden layers (as specified in the backbone config)
        if point_backbone_config.projection_hidden_layer > 0:
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_backbone_config.projection_hidden_layer):
                projection_layers.append(
                    nn.Linear(last_dim, point_backbone_config.projection_hidden_dim[i])
                )
                projection_layers.append(nn.GELU())
                last_dim = point_backbone_config.projection_hidden_dim[i]
            projection_layers.append(nn.Linear(last_dim, config.hidden_size))
            self.point_proj = nn.Sequential(*projection_layers)
        else:
            self.point_proj = nn.Linear(backbone_output_dim, config.hidden_size)
        self.point_proj.to(self.device)

        # A similar projection head for scene-level point features
        if point_backbone_config.projection_hidden_layer > 0:
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_backbone_config.projection_hidden_layer):
                projection_layers.append(
                    nn.Linear(last_dim, point_backbone_config.projection_hidden_dim[i])
                )
                projection_layers.append(nn.GELU())
                last_dim = point_backbone_config.projection_hidden_dim[i]
            projection_layers.append(nn.Linear(last_dim, config.hidden_size))
            self.scene_point_proj = nn.Sequential(*projection_layers)
        else:
            self.scene_point_proj = nn.Linear(backbone_output_dim, config.hidden_size)
        self.scene_point_proj.to(self.device)

        self.embed_dim = config.hidden_size
        self.G = 16  # Number of groups/queries for point backbone, default to 16
        self.scene_G = 64  # Number of groups/queries for scene-level queries

        # --- Start: Token Registration ---
        # Maximum number of asset type tokens supported
        self.max_type_tokens = 50
        # Number of bins/types for positional encoding (XYZ share a total of 20 bins)
        self.num_pos_bins = 20
        # Number of bins for rotation (0, 90, 180, 270 degrees)
        self.num_rot_bins = 4

        if self.config.special_tokens is None:
            self.config.special_tokens = {}

            # (1) Asset Type Tokens: <TYPE_TOKEN_0> ... <TYPE_TOKEN_49>
        for i in range(self.max_type_tokens):
            key = f"TYPE_TOKEN_{i}"
            if key not in self.config.special_tokens:
                self.config.special_tokens[key] = f"<{key}>"

                # (2) Position Bin Tokens: <POS_BIN_{i}> for position discretization
        for i in range(self.num_pos_bins):
            key = f"POS_BIN_{i}"
            if key not in self.config.special_tokens:
                self.config.special_tokens[key] = f"<{key}>"

                # (3) Rotation Bin Tokens: <ROT_BIN_{i}> for rotational positions
        for i in range(self.num_rot_bins):
            key = f"ROT_BIN_{i}"
            if key not in self.config.special_tokens:
                self.config.special_tokens[key] = f"<{key}>"

        self.token_bias = None
        self.special_token_ids = {}

        # Register all defined special tokens with the tokenizer and LM
        self.register_special_tokens(self.config.special_tokens)

        self.special_tokens = self.config.special_tokens

        self.eos_tok = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id

        # MultiLayerPositionDecoder is a positional decoder module
        self.pos_decoder = MultiLayerPositionDecoder(config)
        # Ensure decoder runs in bfloat16 for optimal performance
        self.pos_decoder.to(self.device).to(torch.bfloat16)

        ######################
        # LM head construction: ensure all new special tokens are included for training
        self.new_special_token_ids = []

        # Collect IDs for base special tokens (e.g., "ITEM_START", "ITEM_END", "POS_TOKEN")
        for name in ["ITEM_START", "ITEM_END", "POS_TOKEN"]:
            if name in self.special_token_ids:
                self.new_special_token_ids.append(int(self.special_token_ids[name]))

                # Collect all asset type token IDs
        for i in range(self.max_type_tokens):
            key = f"TYPE_TOKEN_{i}"
            if key in self.special_token_ids:
                self.new_special_token_ids.append(int(self.special_token_ids[key]))

                # Collect all position bin token IDs
        for i in range(self.num_pos_bins):
            key = f"POS_BIN_{i}"
            if key in self.special_token_ids:
                self.new_special_token_ids.append(int(self.special_token_ids[key]))

                # Collect all rotation bin token IDs
        for i in range(self.num_rot_bins):
            key = f"ROT_BIN_{i}"
            if key in self.special_token_ids:
                self.new_special_token_ids.append(int(self.special_token_ids[key]))

                # Remove duplicates to ensure each token appears only once
        self.new_special_token_ids = list(dict.fromkeys(self.new_special_token_ids))

        K = len(
            self.new_special_token_ids
        )  # Number of special tokens for the adapter LM head
        print(f"K (Total special tokens in AdapterHead): {K}")
        E = self.embed_dim
        ######################

        if K > 0:
            # Create a new output "adapter head": transforms last_hidden_state (B,L,E) -> (B,L,K)
            # Number of parameters: K*E for weights, K for bias
            self.new_lm_output_emb = nn.Parameter(
                torch.zeros(K, E, device=self.device, dtype=torch.float32)
            )
            self.new_lm_output_bias = nn.Parameter(
                torch.zeros(K, device=self.device, dtype=torch.float32)
            )
            # Weight initialization: try to copy from existing lm_head if possible,
            # Otherwise, fallback to small random init
            try:
                with torch.no_grad():
                    wb = self.base.lm_head.weight  # Shape: (vocab_size, E)
                    # For each new token, copy its current embedding from the language model head
                    for i, tokid in enumerate(self.new_special_token_ids):
                        if 0 <= tokid < wb.size(0):
                            self.new_lm_output_emb[i].data.copy_(
                                wb[tokid].detach().clone()
                            )
            except Exception:
                # If copying fails, use normal initialization
                nn.init.normal_(self.new_lm_output_emb, mean=0.0, std=0.02)
        else:
            self.new_lm_output_emb = None
            self.new_lm_output_bias = None

            # If any special token embeddings exist, construct an adapter LM head for them
        if self.new_lm_output_emb is not None:
            self.lm_head = AdapterLMHead(
                base_lm_head=self.base.lm_head,
                new_token_ids=self.new_special_token_ids,
                new_emb=self.new_lm_output_emb,
                new_bias=self.new_lm_output_bias,
            )
            # Ensure only adapter parameters require gradients by freezing the base
            self.lm_head.freeze_base(True)

            # Build map for efficiently looking up token bias values by input ID
        self._build_fast_bias_map()

    def _build_fast_bias_map(self):
        """
        Create a lookup table that maps input token IDs to indexes in the Token Bias vector.
        Optimizes bias addition by enabling vectorized lookup during inference/training.
        """
        # 1. Collect all token bias keys and parameter tensors
        self.bias_keys = sorted(
            list(self.token_bias.keys())
        )  # Ensure bias sequence order is deterministic
        self.bias_tensors = [self.token_bias[k] for k in self.bias_keys]

        # If there are no token bias tensors, there is nothing to map
        if not self.bias_tensors:
            self.fast_bias_map = None
            return

            # 2. Construct a large lookup table for the entire vocabulary.
            # Each entry maps a vocab token_id => index in the bias tensor list.
        vocab_size = len(self.tokenizer)
        # Use int32/long for memory efficiency
        self.register_buffer(
            "fast_bias_map_idx", torch.full((vocab_size,), -1, dtype=torch.long)
        )

        print(
            f"[Optimization] Building fast bias map for {len(self.bias_keys)} tokens..."
        )

        count = 0
        for idx, key in enumerate(self.bias_keys):
            tid = self.special_token_ids.get(key)
            if tid is not None and tid < vocab_size:
                self.fast_bias_map_idx[tid] = idx
                count += 1

        print(f"[Optimization] Mapped {count} special tokens for vectorization.")

    def register_special_tokens(
        self, special_tokens: Dict[str, str], init_strategy: str = "zeros"
    ):
        """"""
        print(f"[Info] Registering {len(special_tokens)} special tokens...")

        tokens_to_add = []
        for logical_name, token_str in special_tokens.items():
            if token_str not in self.tokenizer.get_vocab():
                tokens_to_add.append(token_str)
            else:

                pass

        if len(tokens_to_add) > 0:

            tokens_to_add = sorted(list(set(tokens_to_add)))
            # rint(tokens_to_add)
            # exit()
            num_added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens_to_add}
            )
            print(f"[Info] Added {num_added} new tokens to tokenizer vocabulary.")

            self.base.resize_token_embeddings(len(self.tokenizer))

        self.special_token_ids = {}

        self.special_token_map = {}

        for logical_name, token_str in special_tokens.items():
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            self.special_token_ids[logical_name] = tid
            self.special_token_map[token_str] = token_str

        emb_layer = self.base.get_input_embeddings()
        emb_weight = emb_layer.weight
        model_embed_dim = emb_weight.shape[1]
        self.embed_dim = model_embed_dim

        self.token_bias = nn.ParameterDict()

        for key, token_str in special_tokens.items():

            if init_strategy == "zeros":
                init_vec = torch.zeros(
                    model_embed_dim, device=emb_weight.device, dtype=emb_weight.dtype
                )
            else:
                init_vec = torch.zeros(
                    model_embed_dim, device=emb_weight.device, dtype=emb_weight.dtype
                )

            self.token_bias[key] = nn.Parameter(init_vec, requires_grad=True)

        print(
            f"[Info] Token registration & bias initialization complete. New Vocab Size: {len(self.tokenizer)}"
        )

    def add_lora(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        target_modules: Optional[list] = None,
        bias: str = "none",
    ):
        """
        Add LoRA adapters to self.base using PEFT.
        - target_modules: list of module name substrings that should receive LoRA (e.g. ['q_proj','v_proj'] or ['c_attn'])
                        If None, we will attempt a reasonable default based on model type.
        - After calling this, self.base becomes a PEFT-wrapped model (get_peft_model result).
        """
        # Defaults: try common patterns. You can override target_modules explicitly.
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            # create config
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # get peft model wrapper
        self.base = get_peft_model(self.base, lora_config)
        # ensure model moved to device
        self.base.to(self.device)

        # helper to inspect params
        try:
            self.base.print_trainable_parameters()  # prints how many params are trainable (LoRA)
        except Exception:
            pass

    def set_trainable_modules_for_finetune(
        self,
        stage: int = 1,
        trainable_top_modules: Optional[list] = None,
        keep_base_trainable: bool = False,
        debug_mode=False,
    ):
        """
        Set trainable parameters based on training stage.

        Args:
            stage (str):
                - "stage1": Freeze LLM (including LoRA), ONLY train Extractor, Adapters, and Decoders.
                            Goal: Feature Alignment.
                - "stage2": Train LLM (LoRA/Full) AND Adapters.
                            Goal: Instruction Tuning / Reasoning.
            trainable_top_modules: List of module names to ALWAYS keep trainable.
            keep_base_trainable: If True, unfreeze full base model (rarely used if using LoRA).
        """

        if debug_mode:
            trainable_top_modules = ["pos_decoder"]
            for n, p in self.named_parameters():
                p.requires_grad = False

            for name in trainable_top_modules:
                module = getattr(self, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = True
                elif hasattr(self, name):

                    attr = getattr(self, name)
                    if isinstance(attr, torch.nn.ParameterDict):
                        for k, param in attr.items():
                            param.requires_grad = True

            trainable = [
                (n, p.numel()) for n, p in self.named_parameters() if p.requires_grad
            ]
            total = sum([p.numel() for _, p in self.named_parameters()])
            trainable_total = sum([n for _, n in trainable])

            print(
                f"[STAGE {stage}] Total params: {total:,}, Trainable: {trainable_total:,} ({100. * trainable_total / total:.3f}%)"
            )

            print("Sample trainable params:", [n for n, _ in trainable[:5]])

            return {
                "total_params": total,
                "trainable_params": trainable_total,
                "trainable_list": [n for n, _ in trainable],
            }

        if trainable_top_modules is None:

            trainable_top_modules = [
                "pos_decoder",
                "point_proj",
                "scene_point_proj",
                "token_bias",
            ]

        print(f"setting trainable modules for [{stage}]...")

        for n, p in self.named_parameters():
            p.requires_grad = False

            # =========================================================

            # =========================================================

        for name in trainable_top_modules:
            module = getattr(self, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
            elif hasattr(self, name):

                attr = getattr(self, name)
                if isinstance(attr, torch.nn.ParameterDict):
                    for k, param in attr.items():
                        param.requires_grad = True

        extractor_modules = [
            self.extractor.sp_xyz_encoder,
            self.extractor.scene_size_encoder,
            self.extractor.scene_feature_fusion,
            self.extractor.scene_pc_adapter,
            self.extractor.scene_Qformer,
            # self.extractor.asset_encoder,
            self.extractor.asset_pc_adapter,
            self.extractor.asset_Qformer,
            self.extractor.asset_pos_embedder,
        ]

        for module in extractor_modules:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

                    # C. Query Tokens (Parameters)
                    # self.extractor.scene_query_tokens.requires_grad = True
                    # self.extractor.asset_query_tokens.requires_grad = True

        if hasattr(self, "lm_head"):

            for n, p in self.lm_head.named_parameters(recurse=True):
                if any(
                    x in n for x in ["new_lm_output", "adapter", "small_token_head"]
                ):
                    p.requires_grad = True

        for attr_name in ["new_lm_output_emb", "new_lm_output_bias"]:
            if hasattr(self, attr_name):
                p = getattr(self, attr_name)
                if isinstance(p, torch.nn.Parameter):
                    p.requires_grad = True

                    # =========================================================

                    # =========================================================
        if stage == 2:
            print("🟢 Stage 2: Unfreezing LLM (LoRA/Base)...")

            if hasattr(self, "base"):
                if keep_base_trainable:

                    for n, p in self.base.named_parameters():
                        p.requires_grad = True
                else:

                    for n, p in self.base.named_parameters():

                        if any(
                            x in n.lower()
                            for x in ["lora", "adapter", "adaption", "prefix"]
                        ):
                            p.requires_grad = True

            for n, p in self.named_parameters():
                if any(x in n.lower() for x in ["lora", "adapter", "adaption"]):
                    p.requires_grad = True

        elif stage == 1:
            print("🔵 Stage 1: LLM is Frozen. Training Encoders & Adapters only.")

            pass

        trainable = [
            (n, p.numel()) for n, p in self.named_parameters() if p.requires_grad
        ]
        total = sum([p.numel() for _, p in self.named_parameters()])
        trainable_total = sum([n for _, n in trainable])

        print(
            f"[STAGE {stage}] Total params: {total:,}, Trainable: {trainable_total:,} ({100. * trainable_total / total:.3f}%)"
        )

        print("Sample trainable params:", [n for n, _ in trainable[:5]])

        return {
            "total_params": total,
            "trainable_params": trainable_total,
            "trainable_list": [n for n, _ in trainable],
        }

    def tok_str(self, name):
        # name here is logical token literal like "<PC_FEAT>"
        return self.special_token_map.get(
            name, name
        )  # fallback to original if not mapped

    def forward(
        self,
        input_ids,  # (B, L) directly from DataLoader
        attention_mask,  # (B, L)
        batch_processed,
        prefix_len,
        labels=None,
        debug_mode=False,
    ):

        B, L = input_ids.shape
        inputs_embeds = self.base.get_input_embeddings()(input_ids)  # (B, L, E)

        if self.training:
            inputs_embeds.requires_grad_(True)

        G = self.G

        flat_point_embs, asset_mask = self.extractor.forward_assets_batch(
            batch_processed, batch_processed["num_items"].tolist()
        )  # (B, M, G, Q_dim)

        if flat_point_embs.dtype != torch.bfloat16:
            flat_point_embs = flat_point_embs.to(torch.bfloat16)

        flat_point_embs = self.point_proj(flat_point_embs)
        flat_point_embs = flat_point_embs.view(B, -1, self.embed_dim).to(
            self.device, dtype=torch.bfloat16
        )  # (B,M,G,D_out)

        scene_box_feat = self.extractor.forward_batch(
            batch_processed["scenes"]
        )  # (B, M, G, Q_dim)

        if scene_box_feat.dtype != torch.bfloat16:
            scene_box_feat = scene_box_feat.to(torch.bfloat16)

        scene_box_feat = self.scene_point_proj(scene_box_feat)

        scene_box_feat = scene_box_feat.view(B, -1, self.embed_dim).to(
            self.device, dtype=torch.bfloat16
        )  # (B,M,G,D_out)

        inputs_embeds = self.process_embedding(
            batch_processed=batch_processed,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            include_answer=True,
            flat_point_embs=flat_point_embs,
            scene_feat_embs=scene_box_feat,
        )

        outputs = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1]

        target_dtype = inputs_embeds.dtype
        if last_hidden_state.dtype != target_dtype:
            last_hidden_state = last_hidden_state.to(dtype=target_dtype)

        logit_loss = self.LogitLoss(last_hidden_state, labels)
        loss_res_pos, loss_size, loss_angle, position = self.PositionLoss(
            batch_processed, input_ids, outputs, prefix_len
        )

        # return

        total_loss = (
            0.25 * logit_loss
            + 0.25 * loss_res_pos
            + 0.25 * loss_size
            + 0.25 * loss_angle
        )

        if self.training:
            dummy_loss = 0.0

            if hasattr(self, "token_bias") and self.token_bias is not None:

                for key, param in self.token_bias.items():
                    if param.requires_grad:
                        dummy_loss += param.sum() * 0.0

            if hasattr(self, "point_proj") and self.point_proj is not None:
                for p in self.point_proj.parameters():
                    if p.requires_grad:
                        dummy_loss += p.sum() * 0.0

            if hasattr(self, "scene_point_proj") and self.scene_point_proj is not None:
                for p in self.scene_point_proj.parameters():
                    if p.requires_grad:
                        dummy_loss += p.sum() * 0.0

            if hasattr(self, "extractor") and self.extractor is not None:
                for p in self.extractor.parameters():
                    if p.requires_grad:
                        dummy_loss += p.sum() * 0.0

            if (
                hasattr(self, "new_lm_output_emb")
                and self.new_lm_output_emb is not None
            ):
                if self.new_lm_output_emb.requires_grad:
                    dummy_loss += self.new_lm_output_emb.sum() * 0.0

            total_loss = total_loss + dummy_loss

        if debug_mode:
            return total_loss, {
                "logit_loss": logit_loss,
                "position_loss": loss_res_pos,
                "size_loss": loss_size,
                "angle_loss": loss_angle,
                "position": position,
                "input_embeds": inputs_embeds,
            }
        else:
            return total_loss, {
                "logit_loss": logit_loss,
                "position_loss": loss_res_pos,
                "size_loss": loss_size,
                "angle_loss": loss_angle,
            }

    def process_embedding(
        self,
        batch_processed,
        input_ids,
        inputs_embeds,
        include_answer,
        flat_point_embs,
        scene_feat_embs,
    ):
        """"""
        B, L = input_ids.shape
        G = self.G
        inputs_embeds = inputs_embeds * 1.0

        if hasattr(self, "fast_bias_map_idx") and self.fast_bias_map_idx is not None:
            max_vocab = self.fast_bias_map_idx.shape[0]

            if self.fast_bias_map_idx.device != input_ids.device:
                self.fast_bias_map_idx = self.fast_bias_map_idx.to(input_ids.device)

        else:
            max_vocab = len(self.tokenizer)  # Fallback

        safe_input_ids = input_ids.clone()

        safe_input_ids[safe_input_ids >= max_vocab] = 0

        # =========================================================

        # =========================================================
        if hasattr(self, "fast_bias_map_idx") and self.fast_bias_map_idx is not None:

            # input_ids (B, L) -> bias_indices (B, L)
            bias_indices = self.fast_bias_map_idx[safe_input_ids]

            if not hasattr(self, "bias_keys"):
                self.bias_keys = sorted(list(self.token_bias.keys()))

                # (KB, D)
            stacked_biases = torch.stack([self.token_bias[k] for k in self.bias_keys])

            if stacked_biases.device != input_ids.device:
                stacked_biases = stacked_biases.to(input_ids.device)

            mask = bias_indices != -1  # (B, L) bool
            safe_indices = bias_indices.clone()
            safe_indices[~mask] = 0

            full_biases = F.embedding(safe_indices, stacked_biases)

            if full_biases.dtype != inputs_embeds.dtype:
                full_biases = full_biases.to(inputs_embeds.dtype)

            inputs_embeds = inputs_embeds + (
                full_biases * mask.unsqueeze(-1).to(inputs_embeds.dtype)
            )

            # =========================================================

            # =========================================================

        pc_feat_id = self.special_token_ids["PC_FEAT"]
        scene_feat_id = self.special_token_ids["SCENE_FEATURE"]
        item_start_id = self.special_token_ids["ITEM_START"]
        item_end_id = self.special_token_ids["ITEM_END"]

        for b in range(B):
            if include_answer:
                num_items_b = int(batch_processed["num_items"][b])
                required_num_items_b = int(batch_processed["required_item_num"][b])
                item_token_nums = num_items_b + required_num_items_b
            else:
                num_items_b = int(batch_processed["num_items"][b])
                item_token_nums = num_items_b

            scene_feat_pos = (input_ids[b] == scene_feat_id).nonzero(as_tuple=True)[0]

            if len(scene_feat_pos) > 0:
                emb_slice = scene_feat_embs[b, :]  # (D,) or (N, D)

                scene_feat_bias = self.token_bias.get("SCENE_FEATURE", None)
                if scene_feat_bias is not None:
                    pb = scene_feat_bias.to(inputs_embeds.dtype)
                    emb_slice = emb_slice.to(inputs_embeds.dtype) + pb

                inputs_embeds[b, scene_feat_pos] = emb_slice

            starts = (input_ids[b] == item_start_id).nonzero(as_tuple=True)[0]
            ends = (input_ids[b] == item_end_id).nonzero(as_tuple=True)[0]

            min_len = min(len(starts), len(ends), item_token_nums)

            emb_idx = 0
            for i in range(min_len):
                s_pos = int(starts[i])
                e_pos = int(ends[i])

                window_ids = input_ids[b, s_pos:e_pos]
                local_pc_pos = (window_ids == pc_feat_id).nonzero(as_tuple=True)[0]

                # if found enough capacity based on G
                if len(local_pc_pos) >= G:
                    global_pc_pos = local_pc_pos[:G] + s_pos

                    emb_slice = flat_point_embs[b, emb_idx : emb_idx + G, :]
                    emb_idx += G

                    pc_bias = self.token_bias.get("PC_FEAT", None)
                    if pc_bias is not None:

                        pb = pc_bias.to(inputs_embeds.dtype)
                        emb_slice = emb_slice.to(inputs_embeds.dtype) + pb.unsqueeze(0)

                    inputs_embeds[b, global_pc_pos] = emb_slice

        return inputs_embeds

    def LogitLoss(self, last_hidden_state, labels):

        logits = self.lm_head(last_hidden_state)
        if logits.dtype in [torch.float16, torch.bfloat16]:
            logits = torch.clamp(logits, min=-1000, max=1000)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous().long()  # (B, L-1)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        flat_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels != -100
        # avg_loss = flat_loss[valid_mask].mean()
        return flat_loss[valid_mask].mean()

    def PositionLoss(
        self, batch_processed, input_ids, llm_outputs, prefix_len, MAX_SCENE_SCALE=10
    ):
        """"""
        device = self.device

        if not hasattr(llm_outputs, "hidden_states"):
            print(
                "!!! Warning: llm_outputs has no hidden_states. Check config.output_hidden_states=True"
            )

            last_hidden = llm_outputs[0]
            return last_hidden.sum() * 0.0, []

        all_layers_states = llm_outputs.hidden_states

        N_layers = self.pos_decoder.num_layers_to_use  # e.g. 4

        selected_layers = all_layers_states[-N_layers:]

        if torch.isnan(selected_layers[-1]).any():
            print("!!! PANIC: last_hidden_state contains NaN/Inf !!!")
            return selected_layers[-1].sum() * 0.0, []

        if "POS_TOKEN" in self.special_token_ids:
            loc_token_id = self.special_token_ids["POS_TOKEN"]
        else:
            loc_token_id = self.tokenizer.convert_tokens_to_ids("<|LOC|>")

        req_pos = batch_processed.get("required_item_res_pos", None)
        req_mask = batch_processed.get("required_item_mask", None)

        if req_pos is None:
            return selected_layers[-1].sum() * 0.0, []

        req_pos = req_pos.to(device=device, dtype=torch.float32)
        req_pos = torch.nan_to_num(req_pos, nan=0.0)
        req_mask = req_mask.to(device)

        # ==========================================================

        # ==========================================================

        collected_hidden_lists = [[] for _ in range(N_layers)]

        target_list = []

        for b in range(len(prefix_len)):
            pl = int(prefix_len[b].item())

            loc_positions_all = (
                (input_ids[b] == loc_token_id).nonzero(as_tuple=False).squeeze(1)
            )

            if loc_positions_all.numel() == 0:
                continue

            loc_positions_answer = loc_positions_all[loc_positions_all >= pl]
            if loc_positions_answer.numel() == 0:
                continue

            valid_targets_idx = req_mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_targets_idx.numel() == 0:
                continue

            use_n = min(
                int(loc_positions_answer.numel()), int(valid_targets_idx.numel())
            )

            loc_pos_used = loc_positions_answer[:use_n]
            target_idx_used = valid_targets_idx[:use_n]

            for layer_idx in range(N_layers):
                # source: (B, Seq, Dim)
                source_layer = selected_layers[layer_idx]

                # slice: (use_n, Dim)
                vec = source_layer[b, loc_pos_used, :]
                collected_hidden_lists[layer_idx].append(vec)

                # ==========================================================

                if layer_idx == N_layers - 1 and b == 0:

                    v_debug = vec[0]
                    # print(f"\n[DEBUG CHECKSUM] TRAIN | Layer: Last | Step/Idx: {loc_pos_used[0].item()}")
                    # print(f"  > Norm : {v_debug.norm().item():.6f}")
                    # print(f"  > Top10: {v_debug[:10]}")
                    # ==========================================================

            target_vec = req_pos[b, target_idx_used, :]
            target_list.append(target_vec)

            # ==========================================================

            # ==========================================================
        if len(target_list) == 0:
            return selected_layers[-1].sum() * 0.0, []

        targets_all = torch.cat(target_list, dim=0)  # (Total_N, 9)

        final_input_tuple = []
        for layer_list in collected_hidden_lists:

            big_tensor = torch.cat(layer_list, dim=0).to(dtype=torch.float32)
            final_input_tuple.append(big_tensor)

        preds_all = self.pos_decoder(final_input_tuple)  # (B_total, 7)

        # ==========================================================
        # Phase 3: Loss (Split Dimensions)
        # ==========================================================

        # Target Split
        # targets_all shape (N, 7): [res_x, res_y, res_z, L, W, H, res_rot]

        # targets_all: [res_x, res_y, res_z, L, W, H, res_rot]
        t_pos_res = targets_all[:, :3]
        t_size = targets_all[:, 3:4]
        t_rot_res = targets_all[:, 4:5]

        p_pos_res = preds_all[:, :3]
        p_size = preds_all[:, 3:4]
        p_rot_res = preds_all[:, 4:5]

        # 1. Position Residuals Loss (XYZ)

        loss_pos_res = F.l1_loss(p_pos_res, t_pos_res) * 10.0

        # 2. Size Loss (LWH)

        loss_size = F.l1_loss(p_size, t_size / 10.0) * 10.0

        # 3. Rotation Residual Loss (Rad)

        loss_rot_res = F.l1_loss(p_rot_res, t_rot_res) * 1.0

        # total_pos_loss = loss_pos_res + loss_size

        preds_restored = torch.cat(
            [p_pos_res, p_size, p_rot_res], dim=-1  # [dx, dy, dz]  # [d_theta]
        )

        return loss_pos_res, loss_size, loss_rot_res, [preds_restored]

    def _add_token_bias_inplace(
        self, token_ids: torch.LongTensor, embeds: torch.Tensor
    ):
        if self.token_bias is None or len(self.token_bias) == 0:
            return

        device = embeds.device
        dtype = embeds.dtype

        id_to_bias_map = {}

        id_source_key = {}

        for key, bias_vec in self.token_bias.items():

            is_type = key.startswith("TYPE_TOKEN_")
            is_pos = key.startswith("POS_BIN_")
            is_rot = key.startswith("ROT_BIN_")
            is_static = key in [
                "ITEM_START",
                "ITEM_END",
                "PC_START",
                "PC_END",
                "POS_TOKEN",
            ]

            if is_type or is_static or is_pos or is_rot:
                tid = self.special_token_ids.get(key)
                if tid is None:
                    continue
                tid_int = int(tid)

                curr_bias = bias_vec.to(device=device, dtype=dtype)

                id_to_bias_map[tid_int] = curr_bias
                id_source_key[tid_int] = key

        unique_input_ids = torch.unique(token_ids)
        for uid in unique_input_ids:
            uid_val = int(uid.item())

            if uid_val in id_to_bias_map:
                mask = token_ids == uid

                # embeds[mask] shape is (Valid_Num, Hidden)
                # bias shape is (Hidden,)
                embeds[mask] = embeds[mask] + id_to_bias_map[uid_val]

    def _safe_encode_literal(self, tokenizer, text: str):

        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(ids, list):
                return ids
            return list(ids)
        except Exception:
            return []

    def _gather_type_token_ids(self):

        type_ids = []
        i = 0
        while True:
            key = f"TYPE_TOKEN_{i}"
            if key not in self.special_token_ids:
                break
            type_ids.append(int(self.special_token_ids[key]))
            i += 1
        return type_ids

    @torch.no_grad()
    def generate_with_custom_embeddings(
        self,
        prefix_ids: torch.LongTensor,
        scene_size: torch.Tensor,  # shape (B, 3)
        prefix_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        max_scene_scale: float = 10.0,
    ):

        model = self.base
        tokenizer = self.tokenizer
        lm_head = self.lm_head

        if device is None:
            device = prefix_ids.device

        B = prefix_ids.shape[0]
        if prefix_mask is not None:
            prefix_lens = prefix_mask.sum(dim=1)
        else:
            prefix_lens = torch.full(
                (B,), prefix_ids.shape[1], dtype=torch.long, device=device
            )

            # ---------------------------------------------------------------

            # ---------------------------------------------------------------
        sp = self.special_token_ids
        item_start_id = int(sp.get("ITEM_START"))
        item_end_id = int(sp.get("ITEM_END"))
        pos_token_id = int(sp.get("POS_TOKEN", sp.get("LOC_TOKEN")))

        pos_bin_ids = []
        pos_bin_map = {}  # ID -> Index
        for i in range(100):  # self.num_pos_bins
            key = f"POS_BIN_{i}"
            if key in sp:
                tid = int(sp[key])
                pos_bin_ids.append(tid)
                pos_bin_map[tid] = i  # Map ID back to 0-99

        rot_bin_ids = []
        rot_bin_map = {}
        for i in range(4):  # self.num_rot_bins
            key = f"ROT_BIN_{i}"
            if key in sp:
                tid = int(sp[key])
                rot_bin_ids.append(tid)
                rot_bin_map[tid] = i

        if len(pos_bin_ids) == 0:
            raise ValueError("POS_BIN tokens not found!")

        # ---------------------------------------------------------------

        # ---------------------------------------------------------------

        all_type_token_ids_set = set(self._gather_type_token_ids())
        sorted_ids = sorted(self._gather_type_token_ids())
        type_id_to_idx = {tid: i for i, tid in enumerate(self._gather_type_token_ids())}

        batch_allowed_types: List[set] = []

        batch_consumed_types: List[set] = [set() for _ in range(B)]

        Ks = torch.zeros(B, dtype=torch.long, device=device)

        prefix_ids_cpu = prefix_ids.cpu().tolist()

        for b in range(B):

            pl = int(prefix_lens[b].item())
            p_ids = prefix_ids_cpu[b][:pl]

            valid_set = set(p_ids) & all_type_token_ids_set
            batch_allowed_types.append(valid_set)

            Ks[b] = len(valid_set)

            # ---------------------------------------------------------------

            # ---------------------------------------------------------------
        id_seq = self._safe_encode_literal(tokenizer, " ID ")
        pos_seq = self._safe_encode_literal(tokenizer, " Position ")

        generated_tokens_list = [[] for _ in range(B)]
        is_finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_hidden_states = []

        # FSM State Vectors
        pending_seq = [list() for _ in range(B)]
        mode = torch.zeros(
            B, dtype=torch.long, device=device
        )  # 0=Free, 1=IDSEQ, 2=POSSEQ
        expect_type = torch.zeros(B, dtype=torch.bool, device=device)
        items_done = torch.zeros(B, dtype=torch.long, device=device)
        after_end = torch.zeros(B, dtype=torch.bool, device=device)
        only_eos_mask = Ks == 0

        # Bin Logic
        bin_step = torch.full((B,), -1, dtype=torch.long, device=device)  # -1, 0..3
        bin_buffer = [[] for _ in range(B)]

        current_type_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        current_type_token_id = torch.full((B,), -1, dtype=torch.long, device=device)
        pairs_per_sample: List[List[dict]] = [[] for _ in range(B)]

        debug_embeddings_list = []

        current_embeds = inputs_embeds
        current_attention_mask = prefix_mask

        # ---------------------------------------------------------------
        # 4. Prefill Forward
        # ---------------------------------------------------------------
        with torch.no_grad():
            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )

            if hasattr(outputs, "last_hidden_state"):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[-1]

            past_key_values = outputs.past_key_values
            all_hidden_states.append(hidden_states)

            # Use last token logits
            last_valid_indices = prefix_lens - 1
            batch_indices = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_indices, last_valid_indices, :]
            logits = lm_head(last_hidden.unsqueeze(1)).squeeze(1)

            masked_logits = logits.clone()
            minus_inf = torch.finfo(masked_logits.dtype).min

            for b in range(B):
                if is_finished[b]:
                    continue
                allowed = None

                if only_eos_mask[b]:
                    allowed = [eos_token_id] if eos_token_id else None
                else:

                    if items_done[b] < Ks[b]:
                        allowed = [item_start_id]
                    else:
                        allowed = [eos_token_id] if eos_token_id else None

                if allowed is not None:
                    masked_logits[b, :] = minus_inf
                    masked_logits[b, allowed] = logits[b, allowed]

            next_token_ids = torch.argmax(masked_logits, dim=-1)

            # Update State based on Prefill Decision
            for i in range(B):
                if is_finished[i]:
                    continue
                tid = int(next_token_ids[i].item())
                generated_tokens_list[i].append(tid)

                if eos_token_id is not None and tid == eos_token_id:
                    is_finished[i] = True
                    continue

                if tid == item_start_id:
                    pending_seq[i] = list(id_seq)  # Push " ID "
                    mode[i] = 1  # IDSEQ
                    after_end[i] = False

                    # ---------------------------------------------------------------
                    # 5. Generation Loop
                    # ---------------------------------------------------------------
        N_use = getattr(self.pos_decoder, "num_layers_to_use", 4)

        for step in range(1, max_new_tokens):
            if is_finished.all():
                break

            with torch.no_grad():
                # Prepare Inputs
                prev_token_ids = next_token_ids.unsqueeze(-1)

                mask_safe = is_finished
                if eos_token_id:
                    SAFE_TOK = eos_token_id
                elif pad_token_id:
                    SAFE_TOK = pad_token_id
                else:
                    SAFE_TOK = 0
                prev_token_ids[mask_safe] = SAFE_TOK

                embed_layer = model.get_input_embeddings()
                prev_embeds = embed_layer(prev_token_ids)

                # [Crucial] Add correct Bias
                self._add_token_bias_inplace(prev_token_ids, prev_embeds)
                debug_embeddings_list.append(prev_embeds[0].detach().cpu())

                if current_attention_mask is not None:
                    new_mask = torch.ones(
                        (B, 1), dtype=current_attention_mask.dtype, device=device
                    )
                    new_mask[is_finished] = 0
                    current_attention_mask = torch.cat(
                        [current_attention_mask, new_mask], dim=1
                    )

                    # Forward
                outputs = model(
                    inputs_embeds=prev_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )

                if hasattr(outputs, "last_hidden_state"):
                    step_hidden_last = outputs.last_hidden_state
                else:
                    step_hidden_last = outputs.hidden_states[-1]

                step_hidden_all = outputs.hidden_states
                past_key_values = outputs.past_key_values
                all_hidden_states.append(step_hidden_last)

                # Decoder Logic
                logits = lm_head(step_hidden_last[:, -1:, :]).squeeze(1)

                # ============ POS Decoder Trigger ============
                pos_hits = (prev_token_ids.squeeze(-1) == pos_token_id) & (~is_finished)
                if pos_hits.any():
                    idxs = pos_hits.nonzero(as_tuple=False).squeeze(1)
                    batched_inputs = []
                    for L in step_hidden_all[-N_use:]:

                        x = L[:, -1, :].index_select(0, idxs).to(dtype=torch.float32)
                        batched_inputs.append(x)

                    preds = self.pos_decoder(batched_inputs)

                    # Reconstruct BBox
                    import math

                    PI = math.pi
                    for j in range(idxs.numel()):
                        b = int(idxs[j].item())
                        t_idx = int(current_type_idx[b].item())
                        t_tid = int(current_type_token_id[b].item())

                        bins = bin_buffer[b]
                        if len(bins) < 4:
                            bins = [10, 10, 10, 0]  # Fallback

                        idx_x, idx_y, idx_z, idx_rot = bins[-4:]  # Use last 4
                        bin_buffer[b] = []  # Reset

                        preds_vec = preds[j]
                        res_pos = preds_vec[0:3]
                        raw_size = preds_vec[3:4]
                        res_rot = preds_vec[4:5]

                        # Pos Calculation
                        scene_len = scene_size[b]  # (3,)
                        # 0..99 -> Center [0.005 .. 0.995]
                        bin_centers_norm = (
                            torch.tensor([idx_x, idx_y, idx_z], device=device) + 0.5
                        ) / self.num_pos_bins
                        # Map to Scene absolute coords (-L/2 to L/2)
                        bin_centers_abs = bin_centers_norm * scene_len - (
                            scene_len / 2.0
                        )

                        final_pos = bin_centers_abs + res_pos

                        # Size Calculation (Softplus ensures positive)
                        final_size = raw_size * max_scene_scale
                        # final_size = final_size

                        centers_rad = [0.0, PI / 2, PI, 3 * PI / 2]
                        center_rot_val = centers_rad[idx_rot] if idx_rot < 4 else 0.0
                        final_rot = center_rot_val + res_rot

                        if t_idx >= 0:
                            # Concatenate [x,y,z, l,w,h, rot]
                            bbox_tensor = torch.cat([final_pos, final_size, final_rot])

                            bbox = bbox_tensor.detach().cpu().tolist()

                            pairs_per_sample[b].append(
                                {
                                    "type_idx": t_idx,
                                    "type_token_id": t_tid,
                                    "bbox": bbox,
                                }
                            )

                            # ============ FSM + Masking Logic (Optimized) ============
                masked_logits = logits.clone()
                minus_inf = torch.finfo(masked_logits.dtype).min

                for b in range(B):
                    if is_finished[b]:
                        if eos_token_id:
                            masked_logits[b, :] = minus_inf
                            masked_logits[b, eos_token_id] = logits[b, eos_token_id]
                        continue

                    allowed = None

                    # 1. Pending Fixed Strings
                    if len(pending_seq[b]) > 0:
                        allowed = [pending_seq[b][0]]

                        # 2. Bin Generation
                    elif bin_step[b] != -1:
                        step_idx = bin_step[b].item()
                        if 0 <= step_idx <= 2:
                            allowed = pos_bin_ids
                        elif step_idx == 3:
                            allowed = rot_bin_ids

                            # 3. Select Type (Non-repetitive Constraint)
                    elif expect_type[b]:

                        remaining = list(
                            batch_allowed_types[b] - batch_consumed_types[b]
                        )

                        if len(remaining) > 0:
                            allowed = remaining
                        else:

                            allowed = [eos_token_id] if eos_token_id else None

                            # 4. After Item -> Next or End
                    elif after_end[b]:

                        remaining_count = len(
                            batch_allowed_types[b] - batch_consumed_types[b]
                        )

                        if remaining_count > 0:
                            allowed = [item_start_id]
                        else:
                            allowed = [eos_token_id] if eos_token_id else None

                            # 5. Global Start (Case where loop starts inside here)
                    elif items_done[b] == 0 and mode[b] == 0 and bin_step[b] == -1:

                        allowed = [item_start_id]

                        # Apply Mask
                    if allowed is not None:
                        masked_logits[b, :] = minus_inf
                        idx_t = torch.tensor(allowed, device=device, dtype=torch.long)
                        masked_logits[b, idx_t] = logits[b, idx_t]
                    elif eos_token_id:
                        # Fallback
                        masked_logits[b, :] = minus_inf
                        masked_logits[b, eos_token_id] = logits[b, eos_token_id]

                next_token_ids = torch.argmax(masked_logits, dim=-1)

                # ============ Update State ============
                for i in range(B):
                    if is_finished[i]:
                        continue
                    tid = int(next_token_ids[i].item())
                    generated_tokens_list[i].append(tid)

                    if eos_token_id and tid == eos_token_id:
                        is_finished[i] = True
                        continue

                        # A. Pending Seq
                    if len(pending_seq[i]) > 0 and tid == pending_seq[i][0]:
                        pending_seq[i].pop(0)

                        if (
                            mode[i] == 1 and len(pending_seq[i]) == 0
                        ):  # IDSEQ done -> Type
                            expect_type[i] = True
                            mode[i] = 0

                        elif (
                            mode[i] == 2 and len(pending_seq[i]) == 0
                        ):  # POSSEQ done -> X
                            bin_step[i] = 0
                            mode[i] = 0

                            # B. Bin Logic
                    elif bin_step[i] != -1:
                        if bin_step[i] <= 2:
                            if tid in pos_bin_map:
                                bin_buffer[i].append(pos_bin_map[tid])
                        elif bin_step[i] == 3:
                            if tid in rot_bin_map:
                                bin_buffer[i].append(rot_bin_map[tid])

                        if bin_step[i] < 3:
                            bin_step[i] += 1
                        else:
                            bin_step[i] = -1
                            pending_seq[i] = [pos_token_id, item_end_id]

                            # C. Main Logic
                    else:
                        if tid == item_start_id:
                            pending_seq[i] = list(id_seq)
                            mode[i] = 1
                            after_end[i] = False
                            current_type_idx[i] = -1
                            bin_step[i] = -1
                            bin_buffer[i] = []

                        elif expect_type[i]:
                            expect_type[i] = False

                            batch_consumed_types[i].add(tid)

                            current_type_idx[i] = int(type_id_to_idx.get(tid, -1))
                            current_type_token_id[i] = tid

                            pending_seq[i] = list(pos_seq)
                            mode[i] = 2

                        elif tid == item_end_id:
                            items_done[i] += 1
                            after_end[i] = True

                            # Finish up...
                            # ... (rest of the function same as before)
        full_hidden_states = torch.cat(all_hidden_states, dim=1)
        max_gen_len = max(len(t) for t in generated_tokens_list) if B > 0 else 0
        max_total_len = prefix_ids.shape[1] + max_gen_len
        generated_ids = torch.full(
            (B, max_total_len),
            pad_token_id if pad_token_id else 0,
            dtype=torch.long,
            device=device,
        )

        for i in range(B):
            pl = int(prefix_lens[i].item())
            generated_ids[i, :pl] = prefix_ids[i, :pl]
            gen = generated_tokens_list[i]
            if len(gen) > 0:
                generated_ids[i, pl : pl + len(gen)] = torch.tensor(
                    gen, dtype=torch.long, device=device
                )

        generated_tokens = []
        for i in range(B):
            pl = int(prefix_lens[i].item())
            gen_len = len(generated_tokens_list[i])
            full_seq = generated_ids[i, : pl + gen_len]
            valid = [t for t in full_seq.tolist() if 0 <= t < len(tokenizer)]
            generated_tokens.append(tokenizer.convert_ids_to_tokens(valid))

        full_debug_embeddings = torch.cat(debug_embeddings_list, dim=0)

        return (
            generated_ids,
            generated_tokens,
            full_hidden_states,
            prefix_lens,
            pairs_per_sample,
            full_debug_embeddings,
        )

    def generate_with_custom_embeddings_free(
        self,
        prefix_ids: torch.LongTensor,
        scene_size: torch.Tensor,
        prefix_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        max_scene_scale: float = 10.0,
    ):
        """"""
        debug_embeddings_list = []
        model = self.base
        lm_head = self.lm_head

        if device is None:
            device = prefix_ids.device

        B = prefix_ids.shape[0]

        # scene_size = scene_size.to(device)

        if prefix_mask is not None:
            prefix_lens = prefix_mask.sum(dim=1)
        else:
            prefix_lens = torch.full(
                (B,), prefix_ids.shape[1], dtype=torch.long, device=device
            )

        sp = self.special_token_ids
        pos_token_id = int(sp.get("POS_TOKEN", sp.get("LOC_TOKEN")))

        type_token_ids_set = set(self._gather_type_token_ids())
        type_id_to_idx = {tid: i for i, tid in enumerate(self._gather_type_token_ids())}

        pos_bin_map = {}  # Token ID -> 0~99
        for i in range(self.num_pos_bins):
            key = f"POS_BIN_{i}"
            if key in self.special_token_ids:
                tid = int(self.special_token_ids[key])
                pos_bin_map[tid] = i

        rot_bin_map = {}  # Token ID -> 0~3
        for i in range(self.num_rot_bins):
            key = f"ROT_BIN_{i}"
            if key in self.special_token_ids:
                tid = int(self.special_token_ids[key])
                rot_bin_map[tid] = i

        generated_tokens_list = [[] for _ in range(B)]
        is_finished = torch.zeros(B, dtype=torch.bool, device=device)
        all_hidden_states = []

        last_seen_type_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        last_seen_type_token = torch.full((B,), -1, dtype=torch.long, device=device)

        # [Modification 3] Bin Buffer

        bin_buffer = [[] for _ in range(B)]

        pairs_per_sample: List[List[dict]] = [[] for _ in range(B)]

        current_embeds = inputs_embeds
        current_attention_mask = prefix_mask

        if current_attention_mask is not None:
            current_attention_mask = current_attention_mask.long()

        with torch.no_grad():
            outputs = model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention_mask,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            if hasattr(outputs, "last_hidden_state"):

                hidden_states = outputs.last_hidden_state
            else:

                hidden_states = outputs.hidden_states[-1]
                # -----------------------------------------------

            batch_indices = torch.arange(B, device=device)
            last_valid_indices = prefix_lens - 1
            last_hidden = hidden_states[batch_indices, last_valid_indices, :]
            logits = lm_head(last_hidden.unsqueeze(1)).squeeze(1)
            next_token_ids = torch.argmax(logits, dim=-1).view(B)

            for i in range(B):
                tid = int(next_token_ids[i].item())
                generated_tokens_list[i].append(tid)
                if eos_token_id is not None and tid == eos_token_id:
                    is_finished[i] = True

        debug_embeddings_list.append(current_embeds[0].detach().cpu())

        # ==========================================
        # Step 2: Decoding Loop
        # ==========================================
        N_use = getattr(self.pos_decoder, "num_layers_to_use", 4)

        for step in range(1, max_new_tokens):
            if is_finished.all():
                break

            with torch.no_grad():
                prev_token_ids = next_token_ids.unsqueeze(-1)
                if eos_token_id is not None:
                    prev_token_ids[is_finished] = eos_token_id

                embed_layer = model.get_input_embeddings()
                prev_embeds = embed_layer(prev_token_ids)
                # prev_embeds = model.embed_tokens(prev_token_ids)

                self._add_token_bias_inplace(prev_token_ids, prev_embeds)
                debug_embeddings_list.append(prev_embeds[0].detach().cpu())

                if current_attention_mask is not None:
                    new_mask = torch.ones(
                        (B, 1), dtype=current_attention_mask.dtype, device=device
                    )
                    new_mask[is_finished] = 0
                    current_attention_mask = torch.cat(
                        [current_attention_mask, new_mask], dim=1
                    )

                    # Attention Forward
                outputs = model(
                    inputs_embeds=prev_embeds,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )

                if hasattr(outputs, "last_hidden_state"):
                    step_hidden_last = outputs.last_hidden_state
                else:
                    step_hidden_last = outputs.hidden_states[-1]

                step_hidden_all = outputs.hidden_states
                # -----------------------------
                past_key_values = outputs.past_key_values
                all_hidden_states.append(step_hidden_last)

                prev_ids_cpu = prev_token_ids.squeeze(-1).cpu().tolist()

                for b, tid in enumerate(prev_ids_cpu):
                    if is_finished[b]:
                        continue

                    if tid in type_token_ids_set:
                        last_seen_type_idx[b] = type_id_to_idx[tid]
                        last_seen_type_token[b] = tid
                        bin_buffer[b] = []

                    elif tid in pos_bin_map:
                        bin_buffer[b].append(pos_bin_map[tid])

                    elif tid in rot_bin_map:

                        bin_buffer[b].append(rot_bin_map[tid])

                is_pos_input = (prev_token_ids.squeeze(-1) == pos_token_id) & (
                    ~is_finished
                )

                if is_pos_input.any():

                    idxs = is_pos_input.nonzero(as_tuple=False).squeeze(1)
                    batched_inputs = []
                    num_layers = len(step_hidden_all)
                    start_layer = num_layers - N_use

                    for i in range(start_layer, num_layers):
                        L = step_hidden_all[i]
                        x = L[:, -1, :].index_select(0, idxs).to(dtype=torch.float32)
                        batched_inputs.append(x)

                        # 2. Decoder Forward

                    raw_preds = self.pos_decoder(batched_inputs)

                    import math

                    PI = math.pi

                    for j in range(idxs.numel()):
                        b = int(idxs[j].item())

                        bins = bin_buffer[b]
                        # print(bins)

                        if len(bins) < 4:

                            bins = [10, 10, 10, 0]  # default

                        bin_buffer[b] = []

                        idx_x, idx_y, idx_z, idx_rot = bins[-4:]

                        preds_vec = raw_preds[j]  # (7,)
                        res_pos = preds_vec[0:3]
                        # print("res_pos: ",res_pos)
                        raw_size = preds_vec[3:4]
                        res_rot = preds_vec[4:5]

                        # Center = (idx + 0.5) / 100 * Size - Size/2
                        # Abs = Center + Residual
                        scene_len = scene_size[b]  # (3,)

                        bin_centers = (
                            torch.tensor([idx_x, idx_y, idx_z], device=device) + 0.5
                        ) / self.num_pos_bins

                        # print([idx_x, idx_y, idx_z],bin_centers)

                        bin_centers = bin_centers * scene_len - (scene_len / 2.0)

                        final_pos = bin_centers + res_pos

                        final_size = raw_size * max_scene_scale

                        # Center Rotations (Rad): 0, PI/2, PI, 3PI/2
                        centers_rad = [0.0, PI / 2, PI, 3 * PI / 2]
                        center_rot_val = centers_rad[idx_rot] if idx_rot < 4 else 0.0

                        final_rot = center_rot_val + res_rot

                        # Format: [x, y, z, l, w, h, rot]

                        bbox = (
                            torch.cat([final_pos, final_size, final_rot])
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        t_idx = int(last_seen_type_idx[b].item())
                        t_tid = int(last_seen_type_token[b].item())

                        if t_idx >= 0:
                            pairs_per_sample[b].append(
                                {
                                    "type_idx": t_idx,
                                    "type_token_id": t_tid,
                                    "bbox": bbox,
                                }
                            )

                logits = lm_head(step_hidden_last[:, -1:, :])
                next_token_ids = torch.argmax(logits, dim=-1).view(B)

                for i in range(B):
                    if is_finished[i]:
                        continue
                    tid = int(next_token_ids[i].item())
                    generated_tokens_list[i].append(tid)
                    if eos_token_id is not None and tid == eos_token_id:
                        is_finished[i] = True

                        # ==========================================

                        # ==========================================
        full_hidden_states = (
            torch.cat(all_hidden_states, dim=1) if all_hidden_states else None
        )

        max_gen_len = 0
        if len(generated_tokens_list) > 0:
            max_gen_len = max(len(t) for t in generated_tokens_list)

        total_len = prefix_ids.shape[1] + max_gen_len
        generated_ids = torch.full(
            (B, total_len),
            pad_token_id if pad_token_id else 0,
            dtype=torch.long,
            device=device,
        )

        for i in range(B):
            pl = int(prefix_lens[i].item())
            generated_ids[i, :pl] = prefix_ids[i, :pl]
            gen = generated_tokens_list[i]
            if len(gen) > 0:
                generated_ids[i, pl : pl + len(gen)] = torch.tensor(gen, device=device)

        generated_tokens = []
        for i in range(B):
            pl = int(prefix_lens[i].item())
            gen_ids = generated_ids[i, pl : pl + len(generated_tokens_list[i])].tolist()
            try:
                toks = self.tokenizer.convert_ids_to_tokens(gen_ids)
            except:
                toks = []
            generated_tokens.append(toks)

        full_debug_embeddings = torch.cat(
            debug_embeddings_list, dim=0
        )  # (Total_Seq_Len, H)

        return (
            generated_ids,
            generated_tokens,
            full_hidden_states,
            prefix_lens,
            pairs_per_sample,
            full_debug_embeddings,
        )

    def generate(
        self,
        input_ids,  # (B, L) directly from DataLoader
        attention_mask,  # (B, L)
        batch_processed,
        prefix_len,
        max_new_tokens=128,
        use_FSM=False,
    ):
        B, L = input_ids.shape
        inputs_embeds = self.base.get_input_embeddings()(input_ids)  # (B, L, E)
        G = self.G

        flat_point_embs, asset_mask = self.extractor.forward_assets_batch(
            batch_processed, batch_processed["num_items"].tolist()
        )  # (B, M, G, Q_dim)

        flat_point_embs = self.point_proj(flat_point_embs)
        flat_point_embs = flat_point_embs.view(B, -1, self.embed_dim).to(
            self.device, dtype=torch.float32
        )  # (B,M,G,D_out)

        scene_box_feat = self.extractor.forward_batch(
            batch_processed["scenes"]
        )  # (B, M, G, Q_dim)
        scene_box_feat = self.point_proj(scene_box_feat)
        scene_box_feat = scene_box_feat.view(B, -1, self.embed_dim).to(
            self.device, dtype=torch.float32
        )  # (B,M,G,D_out)

        inputs_embeds = self.process_embedding(
            batch_processed=batch_processed,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            include_answer=False,
            flat_point_embs=flat_point_embs,
            scene_feat_embs=scene_box_feat,
        )

        # 3) prepare to monkey-patch the model's output head
        base_model = self.base

        self.lm_head = self.lm_head.to(self.device)
        try:
            emb_dtype = base_model.get_input_embeddings().weight.dtype
            if self.lm_head.weight.dtype != emb_dtype:
                self.lm_head = self.lm_head.to(dtype=emb_dtype)
        except Exception:
            pass

        if not use_FSM:
            (
                generated_ids,
                generated_tokens,
                last_hidden_state,
                prefix_lens,
                pairs_per_sample,
                full_debug_embeddings,
            ) = self.generate_with_custom_embeddings_free(  #
                prefix_ids=input_ids,
                scene_size=batch_processed["scene_sizes"],
                prefix_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_id,
                pad_token_id=self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.pad_token
                ),
                device=self.device,
            )
        else:
            (
                generated_ids,
                generated_tokens,
                last_hidden_state,
                prefix_lens,
                pairs_per_sample,
                full_debug_embeddings,
            ) = self.generate_with_custom_embeddings(  #
                prefix_ids=input_ids,
                scene_size=batch_processed["scene_sizes"],
                prefix_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_id,
                pad_token_id=self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.pad_token
                ),
                device=self.device,
            )

        """
        print("FSM generated_tokens")
        print(generated_tokens[0])
        print("FSM generated_ids")
        print(generated_ids[0][prefix_len[0]:])

        #print("pairs per sample")
        k=0
        print("PAIRS PER SAMPLE")
        print(pairs_per_sample[0][0]["bbox"][0:3])
        print(pairs_per_sample[0][1]["bbox"][0:3])
        print(pairs_per_sample[0][2]["bbox"][0:3])
        """

        # print(pairs_per_sample)
        # return generated_ids, generated_tokens, last_hidden_state, prefix_lens

        # id_list, pred_position_list=self.convert_tokens_to_type_and_pos(batch_processed=batch_processed,
        #                                                                input_ids=generated_ids,
        #                                                                last_hidden_state=last_hidden_state,
        #                                                                prefix_len=prefix_lens
        #                                                                )
        # return generated_ids,generated_tokens, id_list, pred_position_list

        return generated_tokens, generated_ids, pairs_per_sample, full_debug_embeddings

    def compose_scene(
        self,
        batch_processed,
        pairs_per_sample,
        include_scene,
        output_dir,
        layout_root,
        asset_root,
        empty_scene_dir,
        SCENE_DATA_ROOT,
        scene_is_m3dlayout=False,
    ):
        scene_id_list = batch_processed["scene_id_list"]
        batch_size = len(scene_id_list)

        scene_rot_list = batch_processed.get("scene_rot_rad_list", [0.0] * batch_size)
        asset_rot_list_batch = batch_processed.get("asset_rot_rad_list", None)
        prefix_mapping_batch = batch_processed.get("_prefix_type_ids", None)

        scene_and_asset_meshes_list = []

        for i in range(batch_size):
            scene_id = scene_id_list[i]

            start_scene_rot = scene_rot_list[i]
            if isinstance(start_scene_rot, torch.Tensor):
                start_scene_rot = start_scene_rot.item()

            collected_asset_meshes = {}

            def get_normalized_asset_path(uid):
                path = os.path.join(asset_root, uid, "raw_model.obj")
                if os.path.exists(path):
                    return path
                    # M3DLayout fallback: assets/<scene_id>/<uid>/raw_model.glb
                if "_" in uid:
                    scene_id = uid.split("_")[0]
                    alt = os.path.join(asset_root, scene_id, uid, "raw_model.glb")
                    if os.path.exists(alt):
                        return alt
                return path

            empty_scene_id_clean = scene_id.split("/")[-1]
            empty_scene_path = os.path.join(
                empty_scene_dir, empty_scene_id_clean + ".glb"
            )
            if not os.path.exists(empty_scene_path):
                empty_scene_path = os.path.join(
                    empty_scene_dir, empty_scene_id_clean + ".obj"
                )

            final_scene = trimesh.Scene()

            if os.path.exists(empty_scene_path):
                try:
                    room_mesh = trimesh.load(empty_scene_path, force="mesh")
                    R_scene_mat = trimesh.transformations.rotation_matrix(
                        start_scene_rot, [0, 0, 1]
                    )
                    room_mesh.apply_transform(R_scene_mat)
                except Exception as e:
                    print(f"[Warn] Load scene failed: {e}")
                    room_mesh = trimesh.Trimesh()
            else:
                room_mesh = trimesh.Trimesh()

            temp_bbox = pairs_per_sample[i]
            temp_asset_list = batch_processed["asset_id_list"][i]
            temp_zy_preprocess = batch_processed["zy_preprocess"][i]

            if asset_rot_list_batch is not None:
                temp_asset_rot_aug = asset_rot_list_batch[i]
            else:
                temp_asset_rot_aug = [0.0] * len(temp_asset_list)

            current_prefix_mapping = (
                prefix_mapping_batch[i] if prefix_mapping_batch else None
            )

            data_list = []
            valid_asset_count = len(temp_asset_list)

            for j in range(len(temp_bbox)):
                pred_item = temp_bbox[j]

                # --- [FIX START] ---

                lookup_id = pred_item["type_idx"]

                if current_prefix_mapping is not None:
                    try:

                        item_idx = current_prefix_mapping.index(lookup_id)
                    except ValueError:
                        print(
                            f"Warning: Model generated Type {lookup_id}, but it is not in prefix mapping {current_prefix_mapping}. Skipping."
                        )
                        continue
                else:
                    # Identity Fallback
                    item_idx = lookup_id
                    # --- [FIX END] ---

                if item_idx < 0 or item_idx >= valid_asset_count:
                    continue

                rot_val = temp_asset_rot_aug[item_idx]
                if isinstance(rot_val, torch.Tensor):
                    rot_val = rot_val.item()

                data_list.append(
                    {
                        "model_uid": temp_asset_list[item_idx]["asset_uid"],
                        "bbox": pred_item["bbox"],
                        "zy_preprocess": temp_zy_preprocess[item_idx],
                        "rot_aug": rot_val,
                    }
                )

            for idx, d in enumerate(data_list):
                uid = d.get("model_uid")
                if not uid:
                    continue

                asset_path = get_normalized_asset_path(uid)
                if not os.path.exists(asset_path):
                    continue

                try:
                    mesh = trimesh.load(asset_path, force="mesh")
                except:
                    continue

                center = mesh.bounding_box.centroid
                mesh.vertices -= center

                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)

                bbox = d["bbox"]
                zy_dim = d["zy_preprocess"][:2]
                scale_factor = d["zy_preprocess"][2:]
                rot_aug = d["rot_aug"]
                pred_yaw = bbox[-1]
                real_target_yaw = pred_yaw + rot_aug
                model_pred_scale = bbox[-2]

                M_pred_yaw = np.eye(4)
                M_pred_yaw[:3, :3] = np.array(
                    [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
                )

                if not scene_is_m3dlayout:
                    mesh.apply_transform(M_pred_yaw)
                S = np.diag([scale_factor[0], scale_factor[1], scale_factor[2], 1])
                mesh.apply_transform(S)

                base_scale = scale_factor[-1] if scale_factor[-1] > 1e-6 else 1.0
                ratio = model_pred_scale / base_scale
                S_model_pred = np.diag([ratio, ratio, ratio, 1])
                mesh.apply_transform(S_model_pred)

                R_1 = trimesh.transformations.euler_matrix(
                    zy_dim[0], zy_dim[1], 0, axes="rxyz"
                )
                mesh.apply_transform(R_1)

                R = trimesh.transformations.euler_matrix(
                    0, 0, real_target_yaw, axes="rxyz"
                )
                T = trimesh.transformations.translation_matrix(bbox[0:3])
                M = T @ R
                mesh.apply_transform(M)

                final_scene.add_geometry(mesh, geom_name=f"gen_{idx}_{uid}")
                collected_asset_meshes[f"gen_{idx}_{uid}"] = mesh.copy()

                # SCENE_DATA_ROOT = "data/custom_scenes"
            with open(os.path.join(SCENE_DATA_ROOT, f"{scene_id}.json"), "r") as f:
                scene_info = json.load(f)

            physics_info = {
                "scene_id": scene_id,
                "scene_height": scene_info["scene_height"],
                "boundary": scene_info["boundary"],
                "objects": [],
            }

            for name, mesh_obj in collected_asset_meshes.items():

                bounds = mesh_obj.bounding_box
                extents = bounds.extents.tolist()
                centroid = bounds.centroid.tolist()

                physics_info["objects"].append(
                    {
                        "node_name": name,
                        "initial_pos": centroid,
                        "extents": extents,
                        "initial_quat": [0, 0, 0, 1],
                    }
                )

            scene_and_asset_meshes_list.append(
                {
                    "room_mesh": room_mesh,
                    "collected_asset_meshes": collected_asset_meshes,
                }
            )

            output_sub_dir = scene_id
            os.makedirs(os.path.join(output_dir, output_sub_dir), exist_ok=True)
            output_path_wt_scene = os.path.join(
                output_dir, output_sub_dir, "layout_wt_scene.glb"
            )
            output_path_full = os.path.join(
                output_dir, output_sub_dir, "layout_full.glb"
            )

            physics_json_path = os.path.join(
                output_dir, output_sub_dir, "layout_physics_info.json"
            )
            with open(physics_json_path, "w") as f:
                json.dump(physics_info, f, indent=4)

            os.makedirs(os.path.dirname(output_path_wt_scene), exist_ok=True)
            try:
                final_scene.export(output_path_wt_scene)
                final_scene.add_geometry(room_mesh, geom_name="room")
                final_scene.export(output_path_full)
                print("Export successful: ", output_path_full)
            except Exception as e:
                print(f"Export failed: {e}")

        return scene_and_asset_meshes_list


def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    else:
        return batch
