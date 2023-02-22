#!/bin/bash

VRS=$1

EPS=`ls -1 output/hpc/v${VRS}/eval| awk -F"-" '{print $2}' | awk -F. '{print $1}'|sort -n`

echo "eval dota v$VRS"
for EPOCH in $EPS; do
    echo "epoch $EPOCH"
    CONFIG_FILE="cfgs/v${VRS}.yml"
    OUT_DIR="output/hpc/v${VRS}"
    python src/main.py --config $CONFIG_FILE --num_workers 0 --output ${OUT_DIR} --phase test --epoch ${EPOCH}
done
