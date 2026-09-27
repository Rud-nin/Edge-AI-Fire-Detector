import torch
import torch.nn as nn


class Model(nn.Module):
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

        self.classifier: nn.Linear = nn.Linear(
            in_features=32,
            out_features=num_classes,
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