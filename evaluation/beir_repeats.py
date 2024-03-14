from datasets import load_dataset
import pandas as pd

HF_DATASET_NAME = "BeIR/hotpotqa-generated-queries"

dataset = load_dataset(HF_DATASET_NAME)['train']
df = pd.DataFrame(dataset)
