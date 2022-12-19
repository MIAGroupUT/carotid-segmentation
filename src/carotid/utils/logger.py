import json
import toml
from typing import Dict, Any, Union, Optional


def write_json(parameters: Dict[str, Any], json_path: str) -> None:
    """Writes parameters dictionary at json_path."""
    json_data = json.dumps(parameters, skipkeys=True, indent=4)
    with open(json_path, "w") as f:
        f.write(json_data)


def read_json(json_path: str) -> Dict[str, Any]:
    """Reads JSON file at json_path and returns corresponding dictionary."""
    with open(json_path, "r") as f:
        parameters = json.load(f)

    return parameters


def read_and_fill_default_toml(
    config_path: Optional[Union[str, Dict[str, Any]]], default_path: str
) -> Dict[str, Any]:
    default_parameters = toml.load(default_path)

    if config_path is None:
        return default_parameters
    elif isinstance(config_path, str):
        config_parameters = toml.load(config_path)
    elif isinstance(config_path, dict):
        config_parameters = config_path.copy()
    else:
        raise ValueError(
            f"First argument should be a path to a TOML file or a dictionary of parameters."
            f"Current value is {config_path}."
        )

    for task_key, task_dict in default_parameters.items():
        # Initialize task if not in config
        if task_key not in config_parameters:
            config_parameters[task_key] = dict()

        # Replace parameters
        for param, value in task_dict.items():
            if param not in config_parameters[task_key]:
                config_parameters[task_key][param] = value

    return config_parameters
