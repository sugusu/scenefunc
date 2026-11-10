#!/usr/bin/env python3
"""Run layout inference on one or more scenes with optional post-processing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nala.eval_defaults import (
    ASSET_ENCODER_TYPE,
    DATA_ROOT,
    DEFAULT_BASE_MODEL,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_TOKENIZER_PATH,
    IMG_CATEGORY_LIST,
    IMG_SAMPLED_ORDERS,
    IMG_SWAP_DICT,
    IMG_USAGE_TABLE,
    LORA_ALPHA,
    LORA_R,
    M3D_CATEGORY_LIST,
    M3D_SAMPLED_ORDERS,
    M3D_SWAP_DICT,
    M3D_USAGE_TABLE,
    MAX_ITEMS,
    MAX_LENGTH,
    OUTPUT_INFER_ROOT,
    SEED,
    THREED_CATEGORY_LIST,
    THREED_SAMPLED_ORDERS,
    THREED_SWAP_DICT,
    THREED_USAGE_TABLE,
)
from nala.models.model7B import LayoutLLM, batch_to_device, model_config, point_backbone_config
from nala.data.data_preprocess_new import LayoutLLMEvalDataCollator, ThreeDFrontDataset
from nala.post_processing import process_scene_physics


def dataset_paths(dataset_type: str):
    root = DATA_ROOT
    if dataset_type == "imaginarium":
        return {
            "layout_dir": os.path.join(root, "Imaginarium/centralized_scene"),
            "asset_dir": os.path.join(root, "Imaginarium/imaginarium_assets_obj"),
            "empty_scene_dir": os.path.join(root, "Imaginarium/empty_scene_glbs"),
            "swap_dict": IMG_SWAP_DICT,
            "category_list": IMG_CATEGORY_LIST,
            "sampled_orders": IMG_SAMPLED_ORDERS,
            "usage_table": IMG_USAGE_TABLE,
            "scene_is_m3dlayout": False,
        }
    if dataset_type == "m3dlayout":
        base = os.path.join(root, "M3DLayout/scene_dataset/infinigen_converted")
        return {
            "layout_dir": os.path.join(base, "centralized_scene"),
            "asset_dir": os.path.join(base, "assets"),
            "empty_scene_dir": os.path.join(base, "empty_scene_glbs"),
            "swap_dict": M3D_SWAP_DICT,
            "category_list": M3D_CATEGORY_LIST,
            "sampled_orders": M3D_SAMPLED_ORDERS,
            "usage_table": M3D_USAGE_TABLE,
            "scene_is_m3dlayout": True,
        }
    return {
        "layout_dir": os.path.join(root, "3Dfront_cleaned/centralized_scene"),
        "asset_dir": os.path.join(root, "3Dfront_cleaned/3D-FUTURE-model"),
        "empty_scene_dir": os.path.join(root, "3Dfront_cleaned/empty_scene_glbs"),
        "swap_dict": THREED_SWAP_DICT,
        "category_list": THREED_CATEGORY_LIST,
        "sampled_orders": THREED_SAMPLED_ORDERS,
        "usage_table": THREED_USAGE_TABLE,
        "scene_is_m3dlayout": False,
    }


def build_model(
    checkpoint_path: str,
    tokenizer_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
) -> LayoutLLM:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side="right", trust_remote_code=False
    )
    config = model_config(
        device="cuda",
        token_3D_start_end=True,
        token_item_start_end=True,
        hidden_size=3584,
        item_position_dim=5,
        pos_decoder_layer=3,
        pos_decoder_dim=[128, 64, 32],
        asset_encoder_type=ASSET_ENCODER_TYPE,
        asset_encoder_ckpt_path=None,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )
    point_config = point_backbone_config(
        model_name="", load_model=True, model_dir="",
        trans_dim=384, depth=12, drop_path_rate=0.1, cls_dim=40,
        num_heads=6, group_size=32, num_group=512,
        point_dims=3, encoder_dims=256, scene_dim=6,
        box_dim=3, box_hidden_layer=2, box_hidden_dim=[128, 128],
        projection_hidden_layer=2, projection_hidden_dim=[128, 128],
        use_max_pool=True, use_color=True,
    )
    model = LayoutLLM(
        config=config,
        point_backbone_config=point_config,
        tokenizer=tokenizer,
        base_model_name_or_path=base_model,
    )
    model.add_lora()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to("cuda")
    return model


def run_inference(
    model: LayoutLLM,
    scene_list: list,
    layout_dir: str,
    asset_dir: str,
    empty_scene_dir: str,
    output_dir: str,
    *,
    swap_dict: str,
    category_list: str,
    sampled_orders: str | None = None,
    usage_table: str | None = None,
    scene_is_m3dlayout: bool = False,
    postprocess: str = "off",
):
    ds = ThreeDFrontDataset(
        scene_list,
        layout_dir,
        asset_dir,
        empty_scene_dir,
        use_asset_desc=True,
        use_scene_desc=True,
        use_asset_swap=False,
        augment_scene_rot=False,
        augment_asset_rot=False,
        asset_swap_dict_dir=swap_dict,
        asset_category_list=category_list,
        pristine_prob=0.0,
        sampled_orders_path=sampled_orders,
        asset_usage_table_path=usage_table,
        scene_is_m3dlayout=scene_is_m3dlayout,
    )
    collator = LayoutLLMEvalDataCollator(
        model.tokenizer,
        model.special_token_map,
        max_length=MAX_LENGTH,
        fixed_order=False,
        max_items=MAX_ITEMS,
        asset_desc_dropout=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
    device = torch.device("cuda")
    os.makedirs(output_dir, exist_ok=True)

    for idx, batch in enumerate(loader):
        scene_id = batch["batch_processed"]["scene_id_list"][0]
        print(f"[{idx + 1}/{len(ds)}] {scene_id}")
        t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_processed = batch_to_device(batch["batch_processed"], device)
        prefix_lens = batch["prefix_lens"].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, pairs_per_sample, _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                batch_processed=batch_processed,
                prefix_len=prefix_lens,
                max_new_tokens=1000,
                use_FSM=True,
            )
        model.compose_scene(
            batch["batch_processed"],
            pairs_per_sample,
            False,
            output_dir,
            layout_dir,
            asset_dir,
            empty_scene_dir,
            layout_dir,
            scene_is_m3dlayout=scene_is_m3dlayout,
        )
        scene_out = os.path.join(output_dir, scene_id.split("/")[-1][:16])
        if postprocess != "off" and os.path.isdir(scene_out):
            process_scene_physics(scene_out)
        print(f"  done in {time.time() - t0:.1f}s -> {scene_out}")


def main():
    parser = argparse.ArgumentParser(description="NaLA layout inference")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to layout checkpoint (.pth). Or set NALA_CHECKPOINT.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to saved tokenizer directory. Or set NALA_TOKENIZER.",
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--dataset_type",
        choices=["imaginarium", "3dfront", "m3dlayout"],
        default="imaginarium",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(OUTPUT_INFER_ROOT, "runs"),
    )
    parser.add_argument(
        "--postprocess",
        choices=["off", "full"],
        default="off",
        help="off=raw model output; full=run physics post-processing",
    )
    parser.add_argument(
        "--scene_list",
        required=True,
        help="JSON file with list of scene ids, or inline JSON array",
    )
    parser.add_argument("--layout_dir", default=None)
    parser.add_argument("--asset_dir", default=None)
    parser.add_argument("--empty_scene_dir", default=None)
    parser.add_argument(
        "--scene_is_m3dlayout",
        action="store_true",
        help="override dataset_type m3dlayout flag",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("Please provide --checkpoint or set NALA_CHECKPOINT")
    if not args.tokenizer:
        parser.error("Please provide --tokenizer or set NALA_TOKENIZER")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.tokenizer):
        parser.error(f"Tokenizer directory not found: {args.tokenizer}")

    torch.manual_seed(args.seed)
    if os.path.isfile(args.scene_list):
        with open(args.scene_list, "r") as f:
            scenes = json.load(f)
    else:
        scenes = json.loads(args.scene_list)

    paths = dataset_paths(args.dataset_type)
    layout_dir = args.layout_dir or paths["layout_dir"]
    asset_dir = args.asset_dir or paths["asset_dir"]
    empty_scene_dir = args.empty_scene_dir or paths["empty_scene_dir"]
    scene_is_m3dlayout = args.scene_is_m3dlayout or paths["scene_is_m3dlayout"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tokenizer:  {args.tokenizer}")
    print(f"Base model: {args.base_model}")
    print(f"Data root:  {DATA_ROOT}")
    print(f"Layout dir: {layout_dir}")

    model = build_model(args.checkpoint, args.tokenizer, args.base_model)
    run_inference(
        model,
        scenes,
        layout_dir,
        asset_dir,
        empty_scene_dir,
        args.output_dir,
        swap_dict=paths["swap_dict"],
        category_list=paths["category_list"],
        sampled_orders=paths["sampled_orders"],
        usage_table=paths["usage_table"],
        scene_is_m3dlayout=scene_is_m3dlayout,
        postprocess=args.postprocess,
    )


if __name__ == "__main__":
    main()
