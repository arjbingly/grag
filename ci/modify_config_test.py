from grag.components.utils import get_config

config = get_config()
print(f"{config['root']['root_path']=}")
print(f"{config['data']['data_path'] = }")
print(f"{config['llm']['base_dir'] = }")
