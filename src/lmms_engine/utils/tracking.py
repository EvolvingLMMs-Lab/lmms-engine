# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import os
from enum import Enum
from functools import partial
from pathlib import Path
from re import L
from typing import Any, List, Union

import wandb

from .logging_utils import Logging


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
    ]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: Union[str, List[str]] = "console",
        config=None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]

        self.logger = {}

        if "wandb" in default_backend:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
            )
            self.logger["wandb"] = wandb

    def log(self, data):
        if "wandb" in self.logger:
            self.logger["wandb"].log(data)

        # format console log
        print(f"{data}")

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
