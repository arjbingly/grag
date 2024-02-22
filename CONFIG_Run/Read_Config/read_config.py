import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation


# Read config.ini fileconfig_file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
# config_file = "config.ini"
# os.environ['CONFIG_FILE'] = config_file
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

print(config['instance']['user'])
# print(config['instance']['user'])
# print(config.instance.user)
