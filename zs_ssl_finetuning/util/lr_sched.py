"""
Adapted from https://github.com/facebookresearch/mae/blob/efb2a80/util/lr_sched.py
"""

# This source code is
# Copyright (c) Meta Platforms, Inc. and affiliates.
# distributed under the terms of the license
# Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# https://github.com/facebookresearch/mae/blob/efb2a80/LICENSE
#
# Aside from whitespace and formatting changes, this version is the same as the
# original, which can be found at
# https://github.com/facebookresearch/mae/blob/efb2a80/util/lr_sched.py

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
