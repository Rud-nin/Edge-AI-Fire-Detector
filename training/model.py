from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from const import FLAG_IMAGENET_NORMALIZATION


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


class Model5(nn.Module):
    """MobileNetV2 with ImageNet weights, retargeted to the binary task.

    Transfer-learning variant of the project's model family. Every section that
    differs from the from-scratch models is marked "CHANGED for MobileNetV2".

    - torchvision's MobileNetV2 with pretrained ImageNet weights.
    - the 1000-class ImageNet head is replaced by
      ``Linear(last_channel, num_classes)`` (kept behind MobileNetV2's own
      ``Dropout(0.2)``), so the output is plain logits (B, num_classes) and
      stays compatible with nn.CrossEntropyLoss.
    - the convolutional backbone (``features``) starts frozen, so only the new
      classifier is trained; call ``set_backbone_frozen(False)`` later to
      fine-tune the whole network.
    - while the backbone is frozen it is also kept in eval mode, so its
      BatchNorm layers reuse the pretrained running statistics instead of
      drifting towards the fire dataset (``train()`` re-applies this every
      epoch); unfreezing restores normal training behaviour.
    - the ImageNet mean/std shift the pretrained weights expect is NOT part of
      this model. ``DATA_FLAGS`` asks the load_* helpers in utils.py for
      already-normalised input, so the exported graph holds no Sub/Mul ops and
      firmware.cpp's ENABLE_IMAGENET_NORMALIZATION applies the same values on
      device.
    """

    # Input-preparation requirements handed to the load_* helpers in utils.py:
    # bit 0 = apply the ImageNet mean/std shift (const.py) to the samples.
    DATA_FLAGS: int = FLAG_IMAGENET_NORMALIZATION

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # CHANGED for MobileNetV2: MobileNetV2 is defined for 3-channel input.
        if in_channels != 3:
            raise ValueError(f"Model5 expects 3 input channels, got {in_channels}")

        # CHANGED for MobileNetV2: build the torchvision backbone. torchvision
        # >= 0.13 uses the weights= API (pretrained=True is deprecated).
        weights: Optional[MobileNet_V2_Weights] = (
            MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        mobilenet: nn.Module = mobilenet_v2(weights=weights)

        self.features: nn.Sequential = mobilenet.features
        # MobileNetV2's own forward calls F.adaptive_avg_pool2d(x, (1, 1));
        # keep the same operation as an explicit module.
        self.avgpool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.last_channel: int = mobilenet.last_channel

        # CHANGED for MobileNetV2: replace the ImageNet classifier with a binary
        # head; MobileNetV2's classifier is Dropout(0.2) -> Linear.
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=self.last_channel,
                out_features=num_classes,
            ),
        )

        # Only the new classifier is trained initially.
        self.backbone_frozen: bool = False
        self.set_backbone_frozen(True)

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Freeze/unfreeze the MobileNetV2 backbone for later fine-tuning.

        While frozen, the backbone is kept in eval mode so its BatchNorm layers
        keep the pretrained running statistics (they would otherwise update from
        batch statistics even though their weight/bias are frozen).
        """
        self.backbone_frozen = frozen
        for parameter in self.features.parameters():
            parameter.requires_grad = not frozen
        self._sync_backbone_mode()

    def _sync_backbone_mode(self) -> None:
        """Backbone runs in eval mode whenever it is frozen (or the model is)."""
        self.features.train(self.training and not self.backbone_frozen)

    def train(self, mode: bool = True) -> "Model5":
        # CHANGED for MobileNetV2: train.py calls model.train() each epoch, which
        # would put the frozen backbone back into train mode; undo that here so
        # its BatchNorm statistics stay fixed.
        super().train(mode)
        self._sync_backbone_mode()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W), already ImageNet-normalised (see DATA_FLAGS)

        x = self.features(x)
        # x: (B, last_channel, H', W')

        x = self.avgpool(x)
        # x: (B, last_channel, 1, 1)

        x = torch.flatten(x, start_dim=1)
        # x: (B, last_channel)

        x = self.classifier(x)
        # x: (B, num_classes) - two logits, compatible with CrossEntropyLoss

        return x


# Registry of the ModelN classes, keyed by N. train.py's and quantize.py's
# --model-type (and the test scripts') look their model up here. Discovery is by
# name, so adding a ModelN class makes --model-type N available without touching
# the callers.
MODEL_CLASSES: Dict[int, Type[nn.Module]] = {
    int(name[len("Model"):]): member
    for name, member in globals().items()
    if name.startswith("Model") and name[len("Model"):].isdigit()
    and isinstance(member, type) and issubclass(member, nn.Module)
}
