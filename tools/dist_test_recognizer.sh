#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/ucf/32_5.yaml --output output/test --only_test --pretrained output/test/best.pth
