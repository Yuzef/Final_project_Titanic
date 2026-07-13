from omegaconf import OmegaConf
from configs.config import config as default_config

def load_config(config_path=None):
    """
    Загружает config.

    Если config_path не передан, используется configs/config.py.
    Если config_path передан, config загружается из .yaml файла.
    """
    if config_path is None:
        return default_config
    
    return OmegaConf.load(config_path)

