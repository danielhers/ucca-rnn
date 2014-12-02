#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 models/<filename> [cosine|euclidean]"
  exit
fi

# verbose
set -x

export PYTHONPATH=${PYTHONPATH}:../ucca

infile=$1
metric=$2
data=dev

python3 run_net.py --in_file $infile --distance --metric $metric
