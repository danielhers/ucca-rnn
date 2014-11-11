#!/bin/bash

# verbose
set -x

export PYTHONPATH=${PYTHONPATH}:../ucca

infile=$1
data=dev

echo $infile
python3 run_net.py --in_file $infile --test --data $data
