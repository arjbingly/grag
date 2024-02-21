## Instruction for code folder

- All classes and functions need to be in component folder
- Main loop code needs to be in the main loop code folder.
- All codes need to have docstrings and hints.

To make the environment and install requirements:  
`conda env create -f <path/to/file>`

To download the Llama models from official repository:
- Request access (fast response).
- Clone the repository and `cd` to it.
- Run `source ./download.sh`. It will prompt for URL paste that from email from Meta.
- Then select the model to download.

Findings
- Llama-2-70b-chat is going OOM for Q5_K_M quantization.
- 
