import yaml


def load_config(path: str) -> dict:
    with open(path) as stream:
        config: dict = yaml.safe_load(stream)

    if not config["cluster"].get("min_border_length", False):
        config["cluster"]["min_border_length"] = (
            config["cluster"]["border_dist_tol"] * 4
        )

    if not config["cluster"].get("translation_tol", False):
        config["cluster"]["translation_tol"] = config["cluster"]["border_dist_tol"] * 5
    return config
