#!/bin/bash

# verbose
set -x

if [ $# -lt 1 ]; then
  echo "Usage: $0 models/<filename>"
  exit
fi

infile=$1
data=test

echo $infile
python runNNet.py --inFile $infile --test --data $data
