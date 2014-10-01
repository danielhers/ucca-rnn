#!/bin/bash

if [ -d passages_backup ]; then
    rm -rf passages
    mv passages_backup passages
fi
if [ -d trees_backup ]; then
    rm -rf trees
    mv trees_backup trees
fi