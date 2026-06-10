import torch
import torch.nn as nn
import torch.nn.functional as F

class RedMaskSpatialFuser(nn.Module):
    def __init__(self, embed_dim=256):
        """
        Dedicated convolutional module for processing the coarse red mask (BiomedCLIP CAM output).
        Acts as an architectural alternative to DPO by using spatial gating semantics 
        to suppress false positives in non-pathological (normal) tissues.
        """
        super().__init__()
        
        # 1. Extract local structures and edge features from the 1-channel coarse mask
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 2. Project feature channels to match SAM's image embedding dimension (typically 256)
        self.proj = nn.Conv2d(64, embed_dim, kernel_size=1, stride=1)
        
        # 3. Final gating convolution to calibrate spatial attention weights
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, red_mask, target_size):
        """
        Args:
            red_mask (torch.Tensor): Coarse visual prompt mask with shape [B, 1, H, W].
            target_size (tuple): Target spatial resolution of SAM image embeddings (e.g., 64x64).
            
        Returns:
            feat (torch.Tensor): Extracted mask feature map with shape [B, 256, H_t, W_t].
            gate (torch.Tensor): Sigmoid-activated spatial attention gate with shape [B, 256, H_t, W_t].
        """
        # Dynamically match mask spatial dimensions with SAM's image embedding size
        if red_mask.shape[-2:] != target_size:
            red_mask = F.interpolate(red_mask, size=target_size, mode='bilinear', align_corners=False)
            
        # Forward pass through the feature extraction pipeline
        x = self.stem(red_mask)
        feat = self.proj(x)
        
        # Compute the element-wise gating map bounded between 0 and 1
        gate = torch.sigmoid(self.gate_conv(feat))
        
        return feat, gate