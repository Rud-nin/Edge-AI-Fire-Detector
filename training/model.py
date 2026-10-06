from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Custom block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
    ) -> None:
        super().__init__()

        self.block: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                # bias is redundant before BatchNorm; BN supplies the shift.
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock(nn.Module):
    """Custom block: (Conv2d -> BatchNorm2d -> ReLU) x 2 -> MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_kernel_size: int = 2,
    ) -> None:
        super().__init__()

        self.block: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                # bias is redundant before BatchNorm; BN supplies the shift.
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Model1(nn.Module):
    """Original 3-conv model."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=3,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d(
                output_size=1,
            ),
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=32,
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # (B)atch
        # (C)hannels: 3 [R, G, B] per pixel
        # (H)eight: ex. 1080 pixels
        # (W)idth: ex. 1920 pixels

        x = self.features(x)
        # x: (B, 32, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # x: (B, 32)

        x = self.classifier(x)
        # x: (B, 2)

        return x


class Model2(nn.Module):
    """5 stacked ConvBlock stages."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # 5 stacked custom blocks; each MaxPool2d halves the spatial size.
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        block_channels: Tuple[Tuple[int, int], ...] = (
            (in_channels, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
        )

        self.features: nn.Sequential = nn.Sequential(
            *[
                ConvBlock(in_channels=in_ch, out_channels=out_ch)
                for in_ch, out_ch in block_channels
            ],
            nn.AdaptiveAvgPool2d(
                output_size=1,
            ),
            nn.ReLU()
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=block_channels[-1][1],
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        x = self.features(x)
        # x: (B, 128, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # x: (B, 128)

        x = self.classifier(x)
        # x: (B, num_classes)

        return x


class Model3(nn.Module):
    """FireNet (Dunnings/Breckon, 2018).

    Conv 5x5 stride 4 -> 64, max pool 3 stride 2, LRN,
    Conv 4x4 stride 1 -> 128, max pool 3 stride 2, LRN,
    Conv 1x1 stride 1 -> 256, max pool 3 stride 2, LRN,
    Dense 4096 (tanh) -> dropout, Dense 4096 (tanh) -> dropout, Dense num_classes.

    Expects a 224x224 input. The reference network ends in softmax; we return
    raw logits so it works with nn.CrossEntropyLoss in train.py.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),

            # 'same' padding for the even 4x4 kernel is 1 before / 2 after
            # (PyTorch's symmetric padding would give 29 instead of 28).
            # nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=1.0),
        )

        # 224 -> 55 -> 27 -> 24 -> 11 -> 11 -> 5 (valid padding).
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(
                in_features=256 * 5 * 5,
                out_features=4096,
            ),
            nn.Tanh(),
            nn.Dropout(p=0.5),

            nn.Linear(
                in_features=4096,
                out_features=4096,
            ),
            nn.Tanh(),
            nn.Dropout(p=0.5),

            nn.Linear(
                in_features=4096,
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        x = self.features(x)
        # x: (B, 256, 5, 5)

        x = torch.flatten(x, start_dim=1)
        # x: (B, 256 * 5 * 5)

        x = self.classifier(x)
        # x: (B, num_classes)

        return x


class Model4(nn.Module):
    """3 stacked DoubleConvBlock stages."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        # 3 stacked double-conv blocks -> the 6 convs are, in order,
        # (3,8) (8,8) (8,16) (16,16) (16,32) (32,32).
        # Each MaxPool2d halves the spatial size: 224 -> 112 -> 56 -> 28.
        self.features: nn.Sequential = nn.Sequential(
            DoubleConvBlock(in_channels=in_channels, out_channels=8),
            DoubleConvBlock(in_channels=8, out_channels=16),
            DoubleConvBlock(in_channels=16, out_channels=32),
            nn.AdaptiveAvgPool2d(
                output_size=1,
            ),
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=32,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=32,
                out_features=num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        x = self.features(x)
        # x: (B, 32, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # x: (B, 32)

        x = self.classifier(x)
        # x: (B, num_classes)

        return x
