from datasets import load_dataset
import pandas as pd
import numpy as np
#%%

HF_DATASET_NAME = "BeIR/hotpotqa-generated-queries"

dataset = load_dataset(HF_DATASET_NAME)['train']
# df = pd.DataFrame(dataset)
#%%
text_hash = [hash(row['text']) for row in dataset]
#%%
df = pd.DataFrame(dataset, index='_id')
df['text_hash'] = text_hash
#%%
num_samples = len(dataset)
num_unique_text = len(np.unique(text_hash))
print(num_samples == num_unique_text)
