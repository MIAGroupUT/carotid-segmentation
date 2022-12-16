import json
from typing import Dict, Any


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
