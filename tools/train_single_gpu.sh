#!/bin/bash

# Optional: activate your environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate mobileseed

# Run training on single GPU
python tools/train.py ./configs/Mobile_Seed/MS_tiny_cityscapes.py --gpu-id 0
