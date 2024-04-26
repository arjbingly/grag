import configparser
import os

from grag.components.utils import get_config

config = configparser.ConfigParser()

workspace = os.getenv('WORKSPACE')
jenkins_home = os.getenv('JENKINS_HOME')

config = get_config()
config['root']['root_path'] = f'{workspace}'
config['data']['data_path'] = f'{jenkins_home}/ci_test_data/data'
config['llm']['base_dir'] = f'{jenkins_home}/ci_test_models/models'
config['env']['env_path'] = f'{jenkins_home}/env_file/.env'

with open(f'{workspace}/config.ini', 'w') as configfile:
    config.write(configfile)
