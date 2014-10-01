#!/bin/bash

# verbose
set -x

export PYTHONPATH=${PYTHONPATH}:../ucca

infile=$1
data=dev

echo $infile
python3 runNNet.py --inFile $infile --test --data $data
