import numpy as np
import torch
import torch.nn as nn


def load_darknet_conv_weights(model, weights_path, num_conv=20):
    """
    Load pretrained Darknet weights into the first num_conv conv layers.
    Darknet binary format: 4x int32 header, then per layer: biases then weights.
    """
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"Model has {len(conv_layers)} conv layers, loading up to {num_conv}...")
    loaded = 0

    with open(weights_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        print(f"Header: {header}")

        for i, conv in enumerate(conv_layers[:num_conv]):
            out_ch = conv.weight.shape[0]
            in_ch  = conv.weight.shape[1]
            kH     = conv.weight.shape[2]
            kW     = conv.weight.shape[3]

            bias = np.fromfile(f, dtype=np.float32, count=out_ch)
            if len(bias) < out_ch:
                print(f"  [{i+1}] File exhausted at layer {i+1}")
                break
            conv.bias.data.copy_(torch.from_numpy(bias))

            wts = np.fromfile(f, dtype=np.float32, count=out_ch * in_ch * kH * kW)
            if len(wts) < out_ch * in_ch * kH * kW:
                print(f"  [{i+1}] File exhausted during weights at layer {i+1}")
                break
            wts = wts.reshape(out_ch, in_ch, kH, kW)
            conv.weight.data.copy_(torch.from_numpy(wts))

            print(f"  [{i+1:2d}] Conv {kH}x{kW}  {in_ch} -> {out_ch}  loaded")
            loaded += 1

        remaining = np.fromfile(f, dtype=np.float32)
        print(f"\nRemaining in file: {len(remaining)} floats (skipped)")

    print(f"Loaded {loaded} layers from pretrained weights")
    return model
