#!/bin/bash
CURR_PATH=$(pwd)
cd ..
cd ..
rm chroma.log
rm -r data/vectordb/*
cd $CURR_PATH

