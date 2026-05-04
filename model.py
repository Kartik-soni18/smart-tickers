"""
model.py ─ CNN Architectures for RGB Market Fingerprint Classification
=======================================================================
Two backbone options, both adapted for 64×64 3-channel input:

  • ResNet-18    – lightweight, fast to train, strong baseline
  • EfficientNet-B0 – better accuracy/compute tradeoff

Output:  3-class logits  →  {0: Sell, 1: Hold, 2: Buy}
         (CrossEntropyLoss handles softmax internally during training;
          use `return_logits=False` for inference probabilities)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)
from typing import Literal


class MarketFingerprintCNN(nn.Module):
    """
    CNN backbone for RGB Market Fingerprint 3-class classification.

    Architecture changes vs. vanilla ImageNet models:
      ResNet-18:
        • First conv: 7×7 stride-2  →  3×3 stride-1  (preserves 64×64 spatial info)
        • MaxPool removed (replaced with Identity) to avoid over-downsampling
        • FC head: Linear → Dropout → Linear(num_classes)

      EfficientNet-B0:
        • Works natively at 64×64 without stride changes
        • Classifier head replaced with Dropout → Linear(num_classes)

    Args:
        backbone    : 'resnet18' | 'efficientnet_b0'
        num_classes : number of output classes (default 3)
        pretrained  : use ImageNet-pretrained weights (recommended)
        dropout     : dropout rate in classifier head
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "efficientnet_b0"] = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes   = num_classes

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base    = resnet18(weights=weights)

            # ── Adapt for 64×64 input ──────────────────────────────────
            # Original conv1: kernel=7, stride=2, padding=3  (designed for 224×224)
            # Replacement   : kernel=3, stride=1, padding=1  (preserves 64×64 features)
            base.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
            base.maxpool = nn.Identity()   # skip aggressive spatial reduction

            in_feat  = base.fc.in_features
            base.fc  = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_feat, num_classes),
            )
            self.model = base

        elif backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base    = efficientnet_b0(weights=weights)

            in_feat          = base.classifier[1].in_features
            base.classifier  = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_feat, num_classes),
            )
            self.model = base

        else:
            raise ValueError(
                f"Unknown backbone {backbone!r}. Choose 'resnet18' or 'efficientnet_b0'."
            )

        self._softmax = nn.Softmax(dim=1)

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, return_logits: bool = True) -> torch.Tensor:
        """
        Args:
            x             : (B, 3, 64, 64) float32 tensor
            return_logits : True  → raw logits  (use with CrossEntropyLoss)
                            False → softmax probs (use for inference)
        Returns:
            (B, num_classes) tensor
        """
        logits = self.model(x)
        return logits if return_logits else self._softmax(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices {0=Sell, 1=Hold, 2=Buy}."""
        with torch.no_grad():
            return self(x, return_logits=False).argmax(dim=1)

    # ──────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print a short architecture summary."""
        print(f"\n{'─'*50}")
        print(f"  Backbone      : {self.backbone_name}")
        print(f"  Input shape   : (B, 3, 64, 64)")
        print(f"  Output classes: {self.num_classes}  →  Sell / Hold / Buy")
        print(f"  Parameters    : {self.count_parameters():,}")
        print(f"{'─'*50}\n")


# ─── Convenience factory ──────────────────────────────────────────────────────

def build_model(
    backbone: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.3,
    device: torch.device = None,
) -> MarketFingerprintCNN:
    """Construct and move the model to `device`."""
    model = MarketFingerprintCNN(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
