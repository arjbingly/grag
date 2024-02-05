# Capstone AWS Setup 
- OS: Ubuntu
- Instance type: g5.2xlarge

# Python Environment Setup - Conda
Run the following commands to install miniconda:

	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm -rf ~/miniconda3/miniconda.sh

After installing, initialise Miniconda:

	~/miniconda3/bin/conda init bash
	~/miniconda3/bin/conda init zsh
	
And the reboot: `sudo reboot`

Update conda: `conda update conda`

## Requirements (Ignore, to be completed)
python
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install conda-forge::chromadb
conda install conda-forge::langchain

Deps for unstructured 
conda install libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc tesseract-lang

# Setup Remote Interpreter - PyCharm
Since PyCharm does not support Remote Conda envionments out of the box. 
We first need to create a bash file on the AWS instance, I have created the below file at `~/activate_Capstone5`

	#!/usr/bin/env bash
	source ~/miniconda3/bin/activate Capstone5
	python "$@"
	
Give executable permission: `chmod +x ~/activate_Capstone5`

Then point to this file while creating a remote interpreter on PyCharm.

To verify this, try running the below python script.

	import os
	import torch
	
	print(os.environ.get('CONDA_PREFIX'))
	print(torch.cuda.is_available())
	
You should expect the following output:

	/home/ubuntu/miniconda3/envs/Capstone5
	True

If you are not running a GPU instance, the second line will return `False`.	