import yaml


def load_config(path: str) -> dict:
    with open(path) as stream:
        config: dict = yaml.safe_load(stream)

    return config
