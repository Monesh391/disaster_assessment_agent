"""
DamageNet v2: Pretrained Siamese Backbone for Disaster Damage Classification
============================================================================
Addresses Problem 4 & 7:
- Pretrained Siamese backbone (ResNet18 / ResNet34) with shared weights on pre and post.
- Concatenates pre features, post features, and feature difference before head.
- Input size 128x128.
- Supports 7th channel polygon mask (4 channels per branch: RGB + Mask).
- Supports standard 4-class head and Ordinal / CORAL head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


class SiameseDamageNetV2(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        in_channels_per_branch: int = 4,  # 3 (RGB) or 4 (RGB + Mask)
        num_classes: int = 4,
        head_type: str = "classification",  # "classification" or "ordinal"
        dropout: float = 0.25,
        use_product_diff: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels_per_branch = in_channels_per_branch
        self.num_classes = num_classes
        self.head_type = head_type
        self.use_product_diff = use_product_diff
        
        # 1. Instantiate torchvision backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # 2. Modify first conv layer if in_channels_per_branch != 3
        if "resnet" in backbone_name and in_channels_per_branch != 3:
            orig_conv = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels_per_branch,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            if pretrained:
                # Copy RGB weights and initialize additional channels
                with torch.no_grad():
                    new_conv.weight[:, :3, :, :] = orig_conv.weight.data
                    for ch in range(3, in_channels_per_branch):
                        new_conv.weight[:, ch:ch+1, :, :] = orig_conv.weight.data.mean(dim=1, keepdim=True) * 0.5
            base_model.conv1 = new_conv
            
        # Extract convolutional feature extractor (everything before fc)
        if "resnet" in backbone_name:
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            )
        else:  # efficientnet
            self.backbone = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
            
        self.feature_dim = feature_dim
        
        # 3. Combined representation dimension
        # [feat_pre, feat_post, abs(feat_post - feat_pre), (optional: feat_post * feat_pre)]
        multiplier = 4 if use_product_diff else 3
        fused_dim = feature_dim * multiplier
        
        # 4. Classification / Ordinal Head
        if head_type == "ordinal":
            # CORAL / Ordinal binary thresholds (K - 1 binary tasks for K=4 classes: >=1, >=2, >=3)
            out_dim = num_classes - 1  # 3
        else:
            out_dim = num_classes  # 4
            
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, out_dim)
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1D feature vector from image branch."""
        feat = self.backbone(x)
        return torch.flatten(feat, 1)
        
    def forward(
        self,
        pre: torch.Tensor,
        post: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for paired pre/post building crops.
        Args:
            pre: Tensor of shape (B, 3, H, W) or (B, 4, H, W)
            post: Tensor of shape (B, 3, H, W) or (B, 4, H, W)
            mask: Optional Tensor of shape (B, 1, H, W) if pre/post are 3 channels
        Returns:
            logits: (B, 4) for classification or (B, 3) for ordinal
        """
        # If mask is provided and branches are 3 channels, concatenate mask
        if mask is not None and pre.shape[1] == 3 and self.in_channels_per_branch == 4:
            x_pre = torch.cat([pre, mask], dim=1)
            x_post = torch.cat([post, mask], dim=1)
        elif pre.shape[1] == self.in_channels_per_branch:
            x_pre = pre
            x_post = post
        elif pre.shape[1] == 7:  # 7-channel single tensor
            # Channels 0..2: Pre RGB, Channels 3..5: Post RGB, Channel 6: Mask
            m = pre[:, 6:7, :, :]
            x_pre = torch.cat([pre[:, 0:3, :, :], m], dim=1)
            x_post = torch.cat([pre[:, 3:6, :, :], m], dim=1)
        else:
            # Fallback zero-mask if 4 channels expected but not supplied
            if self.in_channels_per_branch == 4 and pre.shape[1] == 3:
                m = torch.zeros((pre.shape[0], 1, pre.shape[2], pre.shape[3]), dtype=pre.dtype, device=pre.device)
                x_pre = torch.cat([pre, m], dim=1)
                x_post = torch.cat([post, m], dim=1)
            else:
                x_pre = pre
                x_post = post

        # Shared Siamese feature extraction
        f_pre = self.extract_features(x_pre)
        f_post = self.extract_features(x_post)
        
        # Difference features
        f_diff = torch.abs(f_post - f_pre)
        
        if self.use_product_diff:
            f_prod = f_pre * f_post
            fused = torch.cat([f_pre, f_post, f_diff, f_prod], dim=1)
        else:
            fused = torch.cat([f_pre, f_post, f_diff], dim=1)
            
        logits = self.head(fused)
        return logits

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts raw network output into 4-class probabilities [P(0), P(1), P(2), P(3)]."""
        if self.head_type == "ordinal":
            # Ordinal probabilities from 3 binary logits
            # p_geq_1, p_geq_2, p_geq_3
            p_geq = torch.sigmoid(logits)  # (B, 3)
            p1 = p_geq[:, 0]
            p2 = p_geq[:, 1]
            p3 = p_geq[:, 2]
            
            p_cls0 = 1.0 - p1
            p_cls1 = torch.clamp(p1 - p2, min=0.0)
            p_cls2 = torch.clamp(p2 - p3, min=0.0)
            p_cls3 = p3
            
            probs = torch.stack([p_cls0, p_cls1, p_cls2, p_cls3], dim=1)
            # Normalize to sum to 1.0
            probs = probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-7)
            return probs
        else:
            return F.softmax(logits, dim=1)

    def predict_classes(
        self,
        logits: torch.Tensor,
        calibration_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Converts logits to class predictions (0..3) with optional validation-fitted probability scaling.
        """
        probs = self.logits_to_probs(logits)
        if calibration_weights is not None:
            cal_w = calibration_weights.to(probs.device)
            scaled = probs * cal_w
            return torch.argmax(scaled, dim=1)
        return torch.argmax(probs, dim=1)
