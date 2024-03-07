import os
import sys
from pathlib import Path

# add BasicRAG folder to sys path
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from BasicRAG_v2 import call_rag

if __name__ == "__main__":
    query = 'What types of dependencies does dependence analysis identify in loop programs?'
    responses, sources = call_rag(query)
