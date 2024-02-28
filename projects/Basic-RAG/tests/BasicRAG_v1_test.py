import os
from pathlib import Path
import sys

# add Basic-RAG folder to sys path
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from BasicRAG_v1 import call_rag

if __name__ == "__main__":
    query = 'What types of dependencies does dependence analysis identify in loop programs?'
    response, sources = call_rag(query)
