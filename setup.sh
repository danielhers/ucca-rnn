#!/bin/bash

mkdir -p passages trees models

export PYTHONPATH=${PYTHONPATH}:..

# Get passages
data=ucca_corpus_xmls.tgz
curl -O http://homepages.inf.ed.ac.uk/oabend/ucca/$data
tar xvzf $data -C passages
rm -f $data

# Split to train, dev and test
python3 split.py

# Convert to trees
python3 uccatree.py
