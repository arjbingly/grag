#!/bin/bash
CURR_PATH=$(pwd)
cd ..
conda init bash
conda activate Capstone5
rm chroma.log
rm -r data/vectordb/*
rm data/doc_store/*
chroma run --path data/vectordb --port 8001
