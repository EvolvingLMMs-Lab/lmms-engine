# Most of the code copied from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py
# Modified to work on patch our models

import collections
import inspect

from loguru import logger
from transformers import PreTrainedModel


class MonkeyPatcher:
    def __init__(self, *args, **kwargs):
        # In format
        # {"model_type": {"liger": apply_liger_kernel_to_xxx, "custom": apply_custom_kernel_to_xxx}}
        self._dict = collections.defaultdict(dict)

    def register(self, model_type, patch_type):
        def decorator(func):
            if not callable(func):
                raise TypeError(f"Error: {func} must be callable!")
            if patch_type in self._dict[model_type]:
                logger.warning(
                    f"Monkey patch for model_type='{model_type}', patch_type='{patch_type}' already exists and will be overwritten by {getattr(func, '__name__', repr(func))}."
                )
            self._dict[model_type][patch_type] = func
            return func

        return decorator

    def apply_monkey_patch(self, model_type, patch_type, **kwargs):
        if isinstance(patch_type, list):
            for patch in patch_type:
                self.apply_monkey_patch(model_type, patch, **kwargs)
            return
        if not model_type:
            logger.info("Model type was not provided. No patches will be applied.")
            return
        if model_type not in self._dict.keys():
            logger.info(
                f"There are currently no patches supported for model type: {model_type} with patch type: {patch_type}. Available model types: {self._dict.keys()}"
            )
            return

        apply_fn = self._dict[model_type][patch_type]
        apply_fn_signature = inspect.signature(apply_fn)

        # Filter out the keyword arguments that are not supported by the apply function
        applicable_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in apply_fn_signature.parameters
        }

        logger.info(
            f"Applying patches for model type: {model_type} with patch type: {patch_type} with kwargs: {applicable_kwargs}"
        )

        apply_fn(**applicable_kwargs)

    def apply_monkey_patch_to_instance(
        self, model: PreTrainedModel, patch_type, **kwargs
    ):
        if isinstance(patch_type, list):
            for patch in patch_type:
                self.apply_monkey_patch_to_instance(model, patch, **kwargs)
            return

        model_type = getattr(model, "config", None) and getattr(
            model.config, "model_type", None
        )
        if not model_type:
            logger.info(
                "Model type could not be determined from model config. No patches will be applied."
            )
            return
        if model_type not in self._dict.keys():
            logger.info(
                f"There are currently no patches supported for model type: {model_type} with patch type: {patch_type}. Available model types: {self._dict.keys()}"
            )
            return

        apply_fn = self._dict[model_type][patch_type]

        apply_fn_signature = inspect.signature(apply_fn)

        # Filter out the keyword arguments that are not supported by the apply function
        applicable_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in apply_fn_signature.parameters
        }
        logger.info(
            f"Applying patches to model instance with model type: {model_type} with patch type: {patch_type} with kwargs: {applicable_kwargs}"
        )

        apply_fn(model=model, **applicable_kwargs)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


MONKEY_PATCHER = MonkeyPatcher()
