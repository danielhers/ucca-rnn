#!/bin/bash

[ -d passages_backup ] || mv passages passages_backup
[ -d trees_backup ] || mv trees trees_backup
mkdir -p passages/{dev,test,train} trees models

export PYTHONPATH=${PYTHONPATH}:../ucca

# Get passages
cp toy.xml passages/train/
cp toy.xml passages/dev/
cp toy.xml passages/test/

# Convert to trees
python3 uccatree.py
