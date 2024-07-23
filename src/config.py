import yaml
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_file):
    with open(config_file, 'w') as file:
        yaml.dump(config, file)