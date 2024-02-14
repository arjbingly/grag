#!/bin/bash
conda init bash
conda activate Capstone5
cd ..
cd ..
chroma run --path data/vectordb
