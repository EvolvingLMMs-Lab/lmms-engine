# Transformers 5.0 Migration Guide

This guide helps you migrate to transformers 5.0 while maintaining backward compatibility with older models.

## Overview

LMMs Engine now supports `transformers >= 5.0` while maintaining backward compatibility with `transformers 4.x`. This enables training with the latest models like Qwen3.5 while preserving support for existing models.

## Compatibility Matrix

| Model Family | transformers < 5.0 | transformers >= 5.0 | Minimum Version |
|-------------|-------------------|---------------------|-----------------|
| Qwen2.5-VL | ✅ | ✅ | - |
| Qwen3-VL | ✅ | ✅ | - |
| Qwen3 | ✅ | ✅ | - |
| **Qwen3.5** | ❌ | ✅ | **>= 5.3.0** |
| LLaVA-OneVision1.5 | ✅ | ❌ | < 5.0.0 |
| DLLM models (DreamDLLM, Qwen3DLLM, LLaDADLLM) | ✅ | ❌ | < 5.0.0 |

## Installation

### For Qwen3.5 Training (New Feature)

Qwen3.5 requires transformers 5.3.0 or higher:

```bash
pip install "transformers>=5.3.0"
```

Or with uv:

```bash
uv pip install "transformers>=5.3.0"
```

### For Legacy Models (LLaVA-OneVision1.5, DLLM)

If you need to use LLaVA-OneVision1.5 or DLLM models, install transformers 4.x:

```bash
pip install "transformers<5.0.0"
```

Or with uv:

```bash
uv pip install "transformers<5.0.0"
```

## Verified Compatibilities

The following models have been tested and verified:

### Tested with transformers >= 5.0
- ✅ **Qwen2.5-VL** - Fully compatible
- ✅ **Qwen3-VL** - Fully compatible  
- ✅ **Qwen3** - Fully compatible

### Tested with transformers < 5.0
- ✅ **Qwen2.5-VL** - Fully compatible
- ✅ **Qwen3-VL** - Fully compatible
- ✅ **Qwen3** - Fully compatible
- ✅ **LLaVA-OneVision1.5** - Only compatible with < 5.0
- ✅ **DLLM models** - Only compatible with < 5.0

## How It Works

LMMs Engine automatically detects your transformers version and:

1. **With transformers >= 5.0**: Loads Qwen3.5 and all compatible models. Legacy models (LLaVA-OneVision1.5, DLLM) are excluded from imports.

2. **With transformers < 5.0**: Loads all legacy models. Qwen3.5 is not available.

The version check is performed at import time using `is_transformers_version_greater_or_equal_to()`.

## Troubleshooting

### Error: "Module not found" for Qwen3.5

**Symptom**: Trying to use Qwen3.5 but getting import errors.

**Solution**: Qwen3.5 requires transformers >= 5.3.0. Install the correct version:

```bash
pip install "transformers>=5.3.0"
```

### Error: "Module not found" for LLaVA-OneVision1.5 or DLLM

**Symptom**: Trying to use LLaVA-OneVision1.5 or DLLM models but they're not available.

**Solution**: These models are incompatible with transformers >= 5.0. Downgrade to transformers 4.x:

```bash
pip install "transformers<5.0.0"
```

### Error: ImportError when importing models

**Symptom**: `ImportError` or `ModuleNotFoundError` when importing specific models.

**Solution**: Check your transformers version and consult the compatibility matrix above. Ensure you're using the correct transformers version for your target model.

## Implementation Details

The compatibility is implemented through conditional imports in `src/lmms_engine/models/__init__.py`:

```python
from lmms_engine.utils.import_utils import is_transformers_version_greater_or_equal_to

is_transformers_5 = is_transformers_version_greater_or_equal_to("5.0.0")

# Models that work with both versions are always imported
from .qwen2_5_vl import apply_liger_kernel_to_qwen2_5_vl
from .qwen3_vl import apply_liger_kernel_to_qwen3_vl
from .qwen3 import apply_liger_kernel_to_qwen3

# Models only compatible with transformers < 5.0
if not is_transformers_5:
    from .llava_onevision1_5 import LLaVAOneVision1_5_ForConditionalGeneration
    from .dream_dllm import DreamDLLMForMaskedLM
    # ... other legacy models
```

## Related Resources

- [Qwen-VL Training Guide](../models/qwenvl.md)
- [Data Preparation Guide](../user_guide/data_prep.md)
- [Training Configuration](../getting_started/train.md)
