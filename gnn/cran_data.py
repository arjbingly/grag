import ir_datasets

from gnn.gen_json import gen_json

dataset = ir_datasets.load("cranfield")
print(f'{dataset=}')

save_path = 'Data/cranfield.json'

docs = dataset.docs_iter()
data = [doc.text for doc in docs]
gen_json(data=data, save_path=save_path)
