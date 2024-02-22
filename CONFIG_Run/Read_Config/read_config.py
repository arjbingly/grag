import os
from dotenv import load_dotenv
from configparser import ConfigParser, ExtendedInterpolation


# Read config.ini file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

print(config['instance']['user'])
