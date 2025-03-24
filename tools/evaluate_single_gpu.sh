#!/bin/bash

python tools/test.py ./configs/Mobile_Seed/MS_tiny_cityscapes.py ./ckpt/MS_tiny_cityscapes.pth --gpu-id 0 --eval mIoU
