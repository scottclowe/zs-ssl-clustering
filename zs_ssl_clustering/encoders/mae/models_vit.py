"""
Adapted from
- https://github.com/facebookresearch/mae/blob/efb2a80/models_vit.py
"""

# This source code is
# Copyright (c) Meta Platforms, Inc. and affiliates.
# distributed under the terms of the license
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# https://github.com/facebookresearch/mae/blob/efb2a80/LICENSE
#
# The original source code can be found at
# https://github.com/facebookresearch/mae/blob/efb2a80/models_vit.py
#
# The implementation was modified to work with timm==0.9.12, preserving the
# behaviour of the original code that was designed with timm==0.3.2 and timm==0.4.5.

# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch.nn as nn


def vit_base_patch16(global_pool=False, **kwargs):
    timm_global_pool = "avg" if global_pool else "token"
    model = timm.models.vision_transformer.VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=timm_global_pool,
        **kwargs,
    )
    return model


def vit_large_patch16(global_pool=False, **kwargs):
    timm_global_pool = "avg" if global_pool else "token"
    model = timm.models.vision_transformer.VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=timm_global_pool,
        **kwargs,
    )
    return model


def vit_huge_patch14(global_pool=False, **kwargs):
    timm_global_pool = "avg" if global_pool else "token"
    model = timm.models.vision_transformer.VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=timm_global_pool,
        **kwargs,
    )
    return model
