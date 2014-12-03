#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 models/<filename> [cosine|euclidean]"
  exit
fi

# verbose
set -x

export PYTHONPATH=${PYTHONPATH}:../ucca

infile=$1
if [ $# -gt 1 ]; then
    metric=$2
else
    metric="cosine"
fi
data=dev

python3 run_net.py --in_file $infile --distance --metric $metric
