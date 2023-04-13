#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main.py -cfg configs/ucf/32_5.yaml --batch-size 2 --accumulation-steps 8 --output output/mil --pretrained k400_32_8.pth