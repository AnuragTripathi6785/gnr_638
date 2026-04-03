"""
Load pretrained Darknet extraction.conv.weights into PyTorch YOLOv1 model.

The extraction.conv.weights file contains the first 20 conv layers of YOLOv1
pretrained on ImageNet classification. Loading these gives a much better
starting point than random initialization.

Darknet binary weight format (no batch norm):
  Header : 4 x int32  (major, minor, revision, seen)
  Per conv layer:
    biases  : float32 array of shape (out_channels,)
    weights : float32 array of shape (out_channels, in_channels, kH, kW)
"""

import numpy as np
import torch
import torch.nn as nn


def load_darknet_conv_weights(model, weights_path, num_conv=20):
    """
    Load pretrained Darknet conv weights into the first `num_conv` layers.

    Args:
        model       : YOLOv1 instance
        weights_path: path to extraction.conv.weights
        num_conv    : number of conv layers to load (default 20 = full backbone)

    Returns:
        model with pretrained backbone weights loaded
    """
    # Collect all Conv2d layers in forward order
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"Model has {len(conv_layers)} conv layers total")
    print(f"Loading up to {num_conv} from pretrained weights...\n")
    loaded = 0

    with open(weights_path, 'rb') as f:
        # Read header
        header = np.fromfile(f, dtype=np.int32, count=4)
        print(f"Weights header (major, minor, revision, seen): {header}")

        for i, conv in enumerate(conv_layers[:num_conv]):
            out_ch = conv.weight.shape[0]
            in_ch  = conv.weight.shape[1]
            kH     = conv.weight.shape[2]
            kW     = conv.weight.shape[3]

            # Load biases — stop if file is exhausted
            bias = np.fromfile(f, dtype=np.float32, count=out_ch)
            if len(bias) < out_ch:
                print(f"  [{i+1:2d}] File exhausted — stopped at layer {i+1}")
                break
            conv.bias.data.copy_(torch.from_numpy(bias))

            # Load weights
            wts = np.fromfile(f, dtype=np.float32, count=out_ch * in_ch * kH * kW)
            if len(wts) < out_ch * in_ch * kH * kW:
                print(f"  [{i+1:2d}] File exhausted during weights — stopped at layer {i+1}")
                break
            wts = wts.reshape(out_ch, in_ch, kH, kW)
            conv.weight.data.copy_(torch.from_numpy(wts))

            print(f"  [{i+1:2d}] Conv {kH}x{kW}  {in_ch:4d} -> {out_ch:4d}  loaded ✓")
            loaded += 1

        # Check how many bytes remain (detection head weights — not loaded)
        remaining = np.fromfile(f, dtype=np.float32)
        print(f"\nRemaining in file: {len(remaining)} floats (detection head — skipped)")

    print(f"\nPretrained backbone loaded successfully!")
    return model
