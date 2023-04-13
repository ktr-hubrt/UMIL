 #!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 main_umil.py -cfg configs/ucf/32_5.yaml --batch-size 1 --batch-size-umil 16 --accumulation-steps 8 --output output/umil --pretrained k400_32_8.pth