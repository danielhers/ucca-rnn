#!/bin/bash

# verbose
set -x

infile=$1
data=test

echo $infile
python runNNet.py --inFile $infile --test --data $data
