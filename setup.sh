#!/bin/bash

# Get passages
data=ucca_corpus_xmls.tgz
curl -O http://homepages.inf.ed.ac.uk/oabend/ucca/$data
mkdir passages
tar xvzf $data -C passages
rm -f $data

# Convert to trees
python uccatree.py

# Create directory for saved models
mkdir models
