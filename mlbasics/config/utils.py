import pkg_resources
import logging
import yaml
import os

def get_mlbasics_path():
    
    path = os.environ.get("ML_BASICS_PATH")
    if not path:
        path = pkg_resources.resource_filename("ML_BASICS_PATH", '')
        os.environ["ML_BASICS_PATH"] = path

    return path

def get_config():
    global config_yaml
    return config_yaml

# --- LOADING CONFIG GLOBAL ---
config_path = os.path.join(get_mlbasics_path(), "config", "config.yaml")
with open(config_path, 'r') as stream:
    config_yaml = yaml.safe_load(stream)

# --- LOADING LOGGING ---
logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s",
    level=config_yaml['general']['logging_level']
)

