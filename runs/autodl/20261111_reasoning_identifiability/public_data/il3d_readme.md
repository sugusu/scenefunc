---
license: apache-2.0
task_categories:
- text-to-3d
---
# IL3D: A Large-scale Indoor Layout Dataset for LLM-Driven 3D Scene Generation
## 🏠 [Project Page](https://wenxuzhou.github.io/project/IL3D/)

# IL3D Dataset Usage Guide
This document provides an overview of the IL3D dataset.

## Environment

We bulid this dataset with Ubuntu 22.04, Python 3.11, CUDA 11.8

## Dataset Contents
The IL3D dataset consists of three main components:

3D-FRONT.zip: A 3D asset library containing 3D models for furniture and objects.
HSSD.zip: A supplementary 3D asset library with additional high-quality 3D models.
layout.zip: A scene layout library containing configuration files that define the arrangement of 3D assets in scenes.

## Setup Instructions
Follow these steps to set up the environment and process the dataset:
Unzip the Dataset FilesExtract the three compressed files using the following commands:
```
unzip 3D-FRONT.zip
unzip HSSD.zip
unzip layout.zip
```

This will create directories containing the 3D assets and scene layout configurations.

## Set Up the Conda Environment
Create a new Conda environment named IL3D with Python 3.11:
```
conda create -n IL3D python=3.11
conda activate IL3D
pip install -r requirements.txt
```

## Generate USDA Scene Models
Run the provided Python script to generate 3D scene models in USDA format:
```
python get_usda_rooms.py
```

This script processes the configuration files in the layout.zip archive and references the 3D assets from 3D-FRONT.zip and HSSD.zip. The USDA format requires absolute paths to the 3D assets, which is why this script generates the USDA files separately.