#!/bin/bash

# verbose
set -x

DIR=$(dirname $0)
cd $DIR
export PYTHONPATH=$DIR/../ucca

epochs=40
step=1e-1
wvec_dim=50
wvec_file="/cs/cgrad/danielh/nlp/danielh/workspace/glove/glove.6B.50d.txt.gz"

outfile="models/rntn_wvec_dim_${wvec_dim}_step_${step}_2.bin"

echo $outfile
python3 run_net.py --step $step --epochs $epochs --out_file $outfile \
                --wvec_dim $wvec_dim --wvec_file $wvec_file

