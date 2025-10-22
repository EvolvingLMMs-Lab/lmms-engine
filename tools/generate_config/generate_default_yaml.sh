#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


# Run the Python script with all passed arguments
python "${SCRIPT_DIR}/generate_default_yaml.py" "$@"
