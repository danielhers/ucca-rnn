#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 models/<filename>"
  exit
fi

# verbose
set -x

export PYTHONPATH=${PYTHONPATH}:../ucca

infile=$1
data=dev

if [ `basename $infile` == "baseline.bin" ]; then
  python3 run_baseline.py --in_file $infile --test
else
  python3 run_net.py --in_file $infile --test --data $data
fi
