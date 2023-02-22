#!/bin/bash

VRS=$1
EPOCH=$2

HPC=${HPC:-hpc}
EXTENDED=${EXTENDED:-no}

EPS=`ls -1 output/${HPC}/v${VRS}/eval| awk -F"-" '{print $2}' | awk -F. '{print $1}'|sort -n`

echo "eval dota v$VRS epoch $EPOCH"
echo "epoch $EPOCH"

function print_results {
  VRS=$1
  EPOCH=$2

  CONFIG_FILE="cfgs/v${VRS}.yml"
  OUT_DIR="output/${HPC}/v${VRS}"

  RESULTS=`python src/main.py --config $CONFIG_FILE --num_workers 0 --output ${OUT_DIR} --phase test --epoch ${EPOCH} --machine_reading |grep ^results`

  # ST
  echo $RESULTS | awk '{print $67}'
  # AH
  echo $RESULTS | awk '{print $68}'
  # LA
  echo $RESULTS | awk '{print $69}'
  # OC
  echo $RESULTS | awk '{print $70}'
  # TC
  echo $RESULTS | awk '{print $71}'
  # VP
  echo $RESULTS | awk '{print $72}'

  if [ $EXTENDED = 'yes' ]; then
    # VO
    echo $RESULTS | awk '{print $73}'
    # OO = OO-r + OO-l
    echo $RESULTS | awk '{print $77}'
    # UK
    echo $RESULTS | awk '{print $76}'
  fi

  # ST*
  echo $RESULTS | awk '{print $43}'
  # AH*
  echo $RESULTS | awk '{print $44}'
  # LA*
  echo $RESULTS | awk '{print $45}'
  # OC*
  echo $RESULTS | awk '{print $46}'
  # TC*
  echo $RESULTS | awk '{print $47}'
  # VP*
  echo $RESULTS | awk '{print $48}'
  # VO*
  echo $RESULTS | awk '{print $49}'
  # OO* = OO-r + OO-l
  echo $RESULTS | awk '{print $53}'

  if [ $EXTENDED = 'yes' ]; then
    # UK*
    echo $RESULTS | awk '{print $52}'
  fi
}

# print_results $VRS $EPOCH
for i in `print_results $VRS $EPOCH`; do
  printf "%.1f & " `echo $i*100 | bc`
done
echo -e "\n"
