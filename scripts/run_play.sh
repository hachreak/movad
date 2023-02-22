#!/bin/bash

VRS=$1
EPOCH=$2
HOW_MANY=$3

CONFIG_FILE="cfgs/v${VRS}.yml"
OUT_DIR="output/hpc/v${VRS}"

python src/main.py --config $CONFIG_FILE --num_workers 0 --output ${OUT_DIR} --phase play --epoch ${EPOCH} --make_video --num_videos ${HOW_MANY}
