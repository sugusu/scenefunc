# NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation (ECCV 2026)


<p align="center">
  <a href="https://adamcwan.github.io/NaLA/"><strong>Project Page</strong></a> |
  <a href="http://arxiv.org/abs/2606.29395"><strong>arXiv</strong></a> |
  <a href="https://github.com/adamcwan/NaLA-code"><strong>Code</strong></a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" alt="NaLA teaser" width="90%">
</p>

---

## Introduction

This is the official implementation of **NaLA**, a 3D-native LLM layout agent for high-quality 3D scene generation. NaLA encodes point clouds of scenes and assets as native 3D tokens and generates placements through a coarse-to-fine pose decoding strategy, enabling end-to-end reasoning over 3D geometry for physically and semantically plausible layouts.

**Implementation notes.** Default backbone: `Qwen/Qwen2.5-7B-Instruct`. Default asset encoder: Point-MAE (PointBERT supported). Training mix: **3D-FRONT**, **Imaginarium**, and **M3DLayout** via `--dataset_to_use mixed_all`.

---

## Project structure

```
NaLA-code/
├── assets/                 # Teaser figure
├── checkpoints/encoders/   # Point-MAE / PointBERT / SPFormer weights
├── configs/                # DeepSpeed & model configs
├── docs/                   # Dataset preparation notes
├── scripts/                # train_ds.py, train.py, infer.py
├── shells/                 # Multi-GPU launch scripts
├── tools/                  # Point-cloud & metadata preprocessing
└── src/nala/
    ├── data/               # Dataset & collators
    ├── models/             # LayoutLLM + point encoders
    ├── post_processing/    # Optional physics cleanup
    └── utils/
```

---

## Environment

Install the package and dependencies with:

```bash
git clone https://github.com/adamcwan/NaLA-code.git
cd NaLA-code
pip install -e .
```

Dependencies: PyTorch (≥2.1), Transformers, PEFT, DeepSpeed, trimesh, spconv (Linux). LLM weights: `Qwen/Qwen2.5-7B-Instruct` from Hugging Face.

### Encoder weights

Download the pretrained encoders below and place them under `checkpoints/encoders/` (rename to the local filenames if needed):

| File | Role | Source |
|------|------|--------|
| `pointMAE_pretrain.pth` | Asset encoder (Point-MAE) | [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) (`pretrain.pth`) |
| `point_bert_v1.2.pt` | Asset encoder (PointBERT, optional) | [Hugging Face](https://huggingface.co/RunsenXu/PointLLM_7B_v1.1_init/blob/main/point_bert_v1.2.pt) |
| `spformer_encoder_uniform_superpoints.pth` | Scene encoder (SPFormer) | [SPFormer](https://github.com/sunjiahao1999/SPFormer) |

---

## Datasets

Prepare **3D-FRONT**, **Imaginarium**, and **M3DLayout**, then convert them to the expected layout. See **[docs/dataset.md](docs/dataset.md)**.

---


## Training

Start multi-GPU training with:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NALA_DATA_ROOT=data
bash shells/run_ds_mixed_all.sh
```

Trained layout checkpoints are saved under `outputs/ckpt/`.

---

## Inference

Run layout inference on selected scenes with:

```bash
python scripts/infer.py \
  --checkpoint /path/to/checkpoint.pth \
  --tokenizer /path/to/tokenizer \
  --dataset_type imaginarium \
  --scene_list '["your/scene/id"]' \
  --output_dir outputs/infer/runs
```

Results (composed scene `.glb` and layout metadata) are written under `outputs/infer/runs/`.

---

## Checklist

- ✅ Release NaLA training and inference code
- ⏳ Release NaLA checkpoints *(planned for September)*
- ⏳ Release NaLA training data *(planned for September)*

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wan2026nala,
  title={NaLA: A 3D Native LLM Layout Agent for High-quality 3D Scene Generation},
  author={Wan, Cheng and Mao, Yongsen and Wu, Wenzheng and Xie, Yuxuan and Xiang, Chucheng and Wang, Runze and Zhang, Xiang and Liu, Zhongyuan and Dai, Rushi and Liu, Yuan},
  booktitle={ECCV},
  year={2026}
}
```
