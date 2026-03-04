import yaml

# Load YAML configuration file
def load_config(path='_config.yml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config