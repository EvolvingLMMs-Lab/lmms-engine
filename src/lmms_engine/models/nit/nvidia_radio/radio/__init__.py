# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Enable support for other model types via the timm register_model mechanism
from . import (
    extra_models,
    extra_timm_models,
    open_clip_adaptor,
    vision_transformer_xpos,
)
from .adaptor_base import AdaptorBase, AdaptorInput, RadioOutput

# Register the adaptors
from .adaptor_registry import adaptor_registry
