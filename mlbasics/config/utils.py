import pkg_resources
import logging
import yaml
import os

def get_mlbasics_path():
    """
    Retrieves the path for MLBasics configurations.

    :return: The path to the MLBasics configuration directory.
    .. note:: You can set your own ML_BASICS_PATH in your IDE/Terminal for
        working in a different directory
    """
    
    path = os.environ.get("ML_BASICS_PATH")
    if not path:
        path = pkg_resources.resource_filename("mlbasics", '')
        os.environ["ML_BASICS_PATH"] = path

    return path

def get_config():
    """
    Retrieves the MLBasics configuration.

    :return: A dictionary containing the configuration settings.
    .. note:: The configuration is setled as a Singleton so one modification of the settings
    affects all files.
    """

    global config_yaml
    return config_yaml

# ***************************************** #
# Initializing MLBASICS Configuration       #
# ***************************************** #

# --- Loading Config ---
config_path = os.path.join(get_mlbasics_path(), "config", "config.yaml")
with open(config_path, 'r') as stream:
    config_yaml = yaml.safe_load(stream)

# --- Loading Logging ---
logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s",
    level=config_yaml['general']['logging_level']
)