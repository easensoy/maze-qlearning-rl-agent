import yaml
import os
from types import SimpleNamespace

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    
    def dict_to_namespace(d):
        return SimpleNamespace(**{k: dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})
    
    return dict_to_namespace(data)

CONFIG = load_config()