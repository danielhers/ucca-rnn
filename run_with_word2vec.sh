#!/bin/bash

# verbose
set -x

DIR=$(dirname $0)
cd $DIR
export PYTHONPATH=$DIR/../ucca

epochs=40
step=1e-1
wvec_dim=300
#wvec_file="../word2vec/GoogleNews-vectors-negative300.bin.gz"
wvec_file="../word2vec/GoogleNews-vectors-negative300.txt.gz"

outfile="models/rnn_wvec_dim_${wvec_dim}_step_${step}.bin"

echo $outfile
python3 run_net.py --step $step --epochs $epochs --out_file $outfile \
                --wvec_dim $wvec_dim --wvec_file $wvec_file

