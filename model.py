"""
YOLOv1 Network Architecture — exact replica of Figure 3 from the CVPR 2016 paper.

24 convolutional layers + 2 fully connected layers.
No Batch Normalization.  Leaky ReLU (slope 0.1) everywhere except the final
layer which uses a linear activation.

Architecture (from Figure 3):
  Conv 7x7x64/s2 → MaxPool 2x2/s2
  Conv 3x3x192   → MaxPool 2x2/s2
  Conv 1x1x128, 3x3x256, 1x1x256, 3x3x512 → MaxPool 2x2/s2
  4x (Conv 1x1x256, 3x3x512)  + Conv 1x1x512, 3x3x1024 → MaxPool 2x2/s2
  2x (Conv 1x1x512, 3x3x1024) + Conv 3x3x1024, 3x3x1024/s2
  Conv 3x3x1024, 3x3x1024
  FC 4096 → FC S*S*(B*5+C)
"""

import torch
import torch.nn as nn

from config import S, B, NUM_CLASSES, DROPOUT_RATE


class LeakyReLU(nn.Module):
    """Leaky ReLU with slope 0.1 as defined in Eq. (2) of the paper."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x > 0, x, 0.1 * x)


class ConvBlock(nn.Module):
    """Conv → LeakyReLU  (no batch norm, faithful to the 2016 paper)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.activation = LeakyReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class YOLOv1(nn.Module):
    """
    Full YOLO detection network.
    Input:  (N, 3, 448, 448)
    Output: (N, S, S, B*5 + C)   i.e. (N, 7, 7, 14) for our 4-class dataset
    """

    def __init__(self, num_classes=NUM_CLASSES, num_boxes=B, grid_size=S):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

        # ── 24 Convolutional Layers ──────────────────────────────────────────

        # Block 1 : Conv 7x7x64 stride 2  → MaxPool 2x2 stride 2
        #   Input : 448x448x3  → 224x224x64 → 112x112x64
        self.block1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2 : Conv 3x3x192 → MaxPool 2x2 stride 2
        #   112x112x64  → 112x112x192 → 56x56x192
        self.block2 = nn.Sequential(
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3 : 1x1x128, 3x3x256, 1x1x256, 3x3x512 → MaxPool
        #   56x56x192 → 56x56x512 → 28x28x512
        self.block3 = nn.Sequential(
            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4 : 4 x (1x1x256, 3x3x512) + 1x1x512 + 3x3x1024 → MaxPool
        #   28x28x512  → 28x28x1024 → 14x14x1024
        block4_layers = []
        for _ in range(4):
            block4_layers.append(ConvBlock(512, 256, kernel_size=1))
            block4_layers.append(ConvBlock(256, 512, kernel_size=3, padding=1))
        block4_layers.append(ConvBlock(512, 512, kernel_size=1))
        block4_layers.append(ConvBlock(512, 1024, kernel_size=3, padding=1))
        block4_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block4 = nn.Sequential(*block4_layers)

        # Block 5 : 2 x (1x1x512, 3x3x1024) + 3x3x1024 + 3x3x1024 stride 2
        #   14x14x1024 → 7x7x1024
        self.block5 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
        )

        # Block 6 : Conv 3x3x1024, Conv 3x3x1024
        #   7x7x1024 → 7x7x1024
        self.block6 = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
        )

        # ── Fully Connected Layers ───────────────────────────────────────────
        # Flatten 7*7*1024 = 50176 → 4096 → S*S*(B*5+C)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.S * self.S * 1024, 4096),
            LeakyReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
            # Final layer: linear activation (no nonlinearity)
        )

        # Weight initialization — use Kaiming uniform for conv, Xavier for FC
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (N, 3, 448, 448)
        Returns:
            (N, S, S, B*5 + C)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x


if __name__ == "__main__":
    # Quick sanity check
    model = YOLOv1()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"YOLOv1 — Total parameters: {total_params:,}")
    dummy = torch.randn(1, 3, 448, 448)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 7, 7, 14), f"Expected (1,7,7,14) got {out.shape}"
    print("Architecture OK ✓")
