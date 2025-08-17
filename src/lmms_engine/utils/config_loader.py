import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """Load configuration from either JSON or YAML file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        List of configuration dictionaries

    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If config file does not exist
    """
    config_path = Path(config_path)

    # If path doesn't exist and it's a relative path, try from the project root
    if not config_path.exists() and not config_path.is_absolute():
        # Try to find the project root (where pyproject.toml is)
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / "pyproject.toml").exists():
                alternative_path = current_dir / config_path
                if alternative_path.exists():
                    config_path = alternative_path
                    break
            current_dir = current_dir.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, "r") as f:
        if suffix == ".json":
            return json.load(f)
        elif suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {suffix}. Use .json, .yaml, or .yml"
            )


def save_config(config: List[Dict[str, Any]], config_path: str) -> None:
    """Save configuration to either JSON or YAML file.

    Args:
        config: List of configuration dictionaries
        config_path: Path to save configuration file

    Raises:
        ValueError: If file extension is not supported
    """
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()

    with open(config_path, "w") as f:
        if suffix == ".json":
            json.dump(config, f, indent=2)
        elif suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {suffix}. Use .json, .yaml, or .yml"
            )
