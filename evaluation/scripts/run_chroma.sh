#!/bin/bash
CURR_PATH=$(pwd)
cd ..
conda init bash
conda activate Capstone5
chroma run --path data/vectordb --port 8001
