#!/bin/bash

VRS=$1
EPOCH_INIT=${2:-0}

HPC=${HPC:-hpc}

EPS=`ls -1 output/${HPC}/v${VRS}/eval| awk -F"-" '{print $2}' | awk -F. '{print $1}'|sort -n`

echo "eval dota v$VRS"
for EPOCH in $EPS; do
    echo "epoch $EPOCH"
    if [ $EPOCH -lt $EPOCH_INIT ]; then
      echo "skip epoch $EPOCH"
      continue
    fi
    CONFIG_FILE="cfgs/v${VRS}.yml"
    OUT_DIR="output/${HPC}/v${VRS}"
    RESULTS=`python src/main.py --config $CONFIG_FILE --num_workers 0 --output ${OUT_DIR} --phase test --epoch ${EPOCH} --machine_reading |grep ^results`
    R_EPOCHS="$R_EPOCHS\n`echo $RESULTS | awk '{print $2}'`"
    R_AUC="$R_AUC\n`echo $RESULTS | awk '{print $3}'`"
    R_F1="$R_F1\n`echo $RESULTS | awk '{print $5}'`"
    R_ACC="$R_ACC\n`echo $RESULTS | awk '{print $6}'`"
    R_F1MEAN="$R_F1MEAN\n`echo $RESULTS | awk '{print $7}'`"
done

echo "########"
echo ""
echo "epochs"
echo -e $R_EPOCHS
echo ""
echo "F1-mean"
echo -e $R_F1MEAN
echo ""
echo "F1-score"
echo -e $R_F1
echo ""
echo "f-AUC"
echo -e $R_AUC
echo ""
echo "Accuracy"
echo -e $R_ACC
