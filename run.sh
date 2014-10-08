#!/bin/bash

# verbose
set -x

DIR=$(dirname $0)
cd $DIR
export PYTHONPATH=$DIR/../ucca

epochs=40
step=1e-1
wvecDim=50
wvecFile="/cs/cgrad/danielh/nlp/danielh/workspace/glove/glove.6B.50d.txt.gz"

outfile="models/rntn_wvecDim_${wvecDim}_step_${step}_2.bin"

echo $outfile
python3 runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --wvecDim $wvecDim --wvecFile $wvecFile

