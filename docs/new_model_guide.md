### Guide: Adding a new model (and performance monkey patches)

This guide explains how to add a new model to the engine and how to provide an optional monkey patch for performance (e.g., removing padding or swapping kernels).

### Overview

- Register your model with Hugging Face auto classes so it can be created from config or checkpoints.
- Organize your model package and ensure registration runs at import time.
- Optionally, register a performance patch (e.g., rmpad/liger) and apply it pre- or post-init.

### 1) File layout

- Create a new package under `src/lmms_engine/models/<your_model>/`:

```
src/lmms_engine/models/<your_model>/
  __init__.py
  configuration_<your_model>.py       # if you have a custom config class
  modeling_<your_model>.py            # your model implementation or wrappers
  patchers/
    __init__.py
    liger.py                          # example: performance patch module (optional)
```

### 2) Register the model

Use the helper in `lmms_engine.mapping_func.register_model` to bind your config and model class to the auto classes.

```python
# src/lmms_engine/models/<your_model>/__init__.py
from transformers import PretrainedConfig, PreTrainedModel
from lmms_engine.mapping_func import register_model

class YourModelConfig(PretrainedConfig):
    model_type = "your_model"
    # ... your fields ...

class YourModel(PreTrainedModel):
    config_class = YourModelConfig
    # ... implement forward, init weights, etc. ...

register_model(
    model_type=YourModelConfig.model_type,
    model_config=YourModelConfig,
    model_class=YourModel,
    model_general_type="causal_lm",  # or "image_text_to_text", "masked_lm", "general"
)

# Optional: import patchers so their decorators run
try:
    from .patchers import liger  # noqa: F401
except Exception:
    pass
```

After registration, you can create the model via:
- From pretrained: `AutoConfig.from_pretrained` + appropriate `AutoModel*` loader
- From config: using `create_model_from_config(model_type, config_dict)`

### Monkey Patcher: Registering and Applying Patches

After you register your model, you are recommended to add your own monkey patch to achieve the best performance and allows you to use operations such as `remove padding` and `sequence parallel`. As a rule of thumb, we recommend you to check the implementation of `qwen2` and `qwen2.5 vl` to see how these ops are being applied.

During training, you can either specified the `use_rmpad` or `use_liger_kernel` to apply monkey patch with liger strategy or use the `monkey_patch_kwargs` in model config to control the extra kwargs.

### Concepts

- **Model type**: The `transformers` model family name (e.g., `aero`, `qwen2_5_vl`).
- **Patch type**: The strategy name (currently `liger` is supported by the public API).
- **Pre-init vs Post-init**:
  - Pre-init: call before model instantiation to patch `transformers` modules.
  - Post-init: call on an already-created model instance to replace modules/forwards.

### Quick start

#### 1) Register a patch function

Create a module (e.g., `src/lmms_engine/models/aero/patchers/liger.py`) and register a callable:

```python
from lmms_engine.models.monkey_patch import MONKEY_PATCHER

@MONKEY_PATCHER.register("aero", "liger")
def apply_liger_to_aero(*, rope=True, rms_norm=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=True, use_rmpad=False, model=None, **kwargs):
    """
    Implement your patch logic here.
    - If called for pre-init (no `model` passed), patch HF symbols.
    - If called for post-init (a `model` is passed), mutate the model instance.
    """
    # Example (pseudocode):
    # if model is None:
    #     ... pre-init patching ...
    # else:
    #     ... post-init patching ...
    pass
```

Notes:
- The decorator stores the function at `("aero", "liger")`. Re-registering the same pair overwrites the previous function and logs a warning.
- Only kwargs that appear in your function signature will be forwarded when applying the patch; extra kwargs are ignored automatically.

#### 2) Ensure the registration module is imported

Registration happens at import time. Make sure the module containing your `@MONKEY_PATCHER.register(...)` is imported before you call `apply_monkey_patch(...)`.

Common options:
- Import in the model package `__init__.py` (e.g., `src/lmms_engine/models/aero/__init__.py`).
- Or import explicitly in your runner/setup code before applying patches.

```python
# Example: force import so the decorator runs and registers your function
import lmms_engine.models.aero.patchers.liger  # noqa: F401
```

#### 3) Apply the patch (pre-init)

Call before creating the model to patch `transformers` implementations:

```python
from lmms_engine.models.monkey_patch import MONKEY_PATCHER

MONKEY_PATCHER.apply_monkey_patch(
    model_type="aero",
    patch_type="liger",
    rope=True,
    use_rmpad=True,
)
```

#### 4) Apply the patch to an instance (post-init)

Call on a created model instance to replace modules or `forward` functions:

```python
from lmms_engine.models.monkey_patch import MONKEY_PATCHER

model = ...  # create/load your model

MONKEY_PATCHER.apply_monkey_patch_to_instance(
    model=model,
    patch_type="liger",
    rope=True,
    use_rmpad=True,
)
```

### Signature filtering

`apply_monkey_patch(...)` and `apply_monkey_patch_to_instance(...)` automatically filter `**kwargs` based on the registered function’s signature, so you can pass a shared kwargs dict and each strategy only receives supported arguments.

### Overwrite behavior

If a `(model_type, patch_type)` is registered more than once, the new function overwrites the previous one. A warning is logged to make this explicit.

### Current limitations

- The public API currently routes `patch_type == "liger"`. Other patch types will raise `ValueError` unless the implementation is extended accordingly.
- If you see a message like "There are currently no Liger kernels supported for model type: <model_type>", ensure that the registration module was imported before applying, and that you used the correct `model_type` string.

### Recommendations for adding new patches

- Place patchers per-model under `src/lmms_engine/models/<model>/patchers/<patch_type>.py`.
- Separate pre-init and post-init logic inside the same function by checking whether `model` is provided.
- Perform optional dependency checks (e.g., Liger) lazily inside the function and log a helpful warning if unavailable.
- Keep the function idempotent: if a patch might be called multiple times, guard against double-application.


