"""Shared defaults for inference and training helpers."""

from __future__ import annotations

import os

from nala.paths import DEFAULT_CHECKPOINT, DEFAULT_DATA_ROOT, DEFAULT_TOKENIZER, OUTPUTS_DIR, data_path

DATA_ROOT = str(DEFAULT_DATA_ROOT)
OUTPUT_INFER_ROOT = str(OUTPUTS_DIR / "infer")

THREED_SAMPLED_ORDERS = data_path("3Dfront_cleaned/sampled_orders.json")
THREED_USAGE_TABLE = data_path("3Dfront_cleaned/usage_table.json")
IMG_SAMPLED_ORDERS = data_path("Imaginarium/sampled_orders.json")
IMG_USAGE_TABLE = data_path("Imaginarium/usage_table.json")

IMG_SWAP_DICT = os.path.join(DATA_ROOT, "Imaginarium/category_to_model_ids.json")
IMG_CATEGORY_LIST = os.path.join(DATA_ROOT, "Imaginarium/model_ids_to_category.json")
THREED_SWAP_DICT = os.path.join(DATA_ROOT, "3Dfront_cleaned/category_to_model_ids.json")
THREED_CATEGORY_LIST = os.path.join(DATA_ROOT, "3Dfront_cleaned/model_ids_to_category.json")
M3D_SWAP_DICT = os.path.join(
    DATA_ROOT, "M3DLayout/scene_dataset/infinigen_converted/category_to_model_ids.json"
)
M3D_CATEGORY_LIST = os.path.join(
    DATA_ROOT, "M3DLayout/scene_dataset/infinigen_converted/model_ids_to_category.json"
)
M3D_SAMPLED_ORDERS = os.path.join(
    DATA_ROOT, "M3DLayout/scene_dataset/infinigen_converted/sampled_orders.json"
)
M3D_USAGE_TABLE = os.path.join(
    DATA_ROOT, "M3DLayout/scene_dataset/infinigen_converted/usage_table.json"
)

DEFAULT_BASE_MODEL = os.environ.get("NALA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT
DEFAULT_TOKENIZER_PATH = DEFAULT_TOKENIZER

ASSET_ENCODER_TYPE = "pointmae"
LORA_R = 64
LORA_ALPHA = 128
MAX_ITEMS = 20
MAX_LENGTH = 4096
SEED = 42
