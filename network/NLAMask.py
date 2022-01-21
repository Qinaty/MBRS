import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from .blocks.AttentionNet import ResBlock, Non_local_Block


class NLAMask(nn.Module):
    def __init__(self, H, W, input_features=3, output_features=192, down_features=96):
        # input_features = 3, output_features = 192 ??? (paper)
        super(NLAMask, self).__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.d_features = down_features
        self.H = H
        self.W = W

        # downsample
        self.conv = nn.Conv2d(self.in_features, self.d_features, 5, 1, 2)
        self.down = nn.Conv2d(2 * self.d_features, self.out_features, 5, 2, 2)
        self.up = nn.UpsamplingBilinear2d()

        self.trunk1 = nn.Sequential(ResBlock(self.d_features, self.d_features, 3, 1, 1),
                                    ResBlock(self.d_features, self.d_features, 3, 1, 1),
                                    nn.Conv2d(self.d_features, 2*self.d_features, 5, 2, 2))
        self.trunk2 = nn.Sequential(ResBlock(2*self.d_features, 2*self.d_features, 3, 1, 1),
                                    ResBlock(2*self.d_features, 2*self.d_features, 3, 1, 1),
                                    ResBlock(2*self.d_features, 2*self.d_features, 3, 1, 1))
        self.trunk3 = nn.Sequential(ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                    ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                    ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                    nn.Conv2d(self.out_features, self.out_features, 5, 2, 2))

        self.mask = nn.Sequential(Non_local_Block(self.out_features, self.out_features//2),
                                  ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                  ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                  ResBlock(self.out_features, self.out_features, 3, 1, 1),
                                  nn.Conv2d(self.out_features, self.out_features, 1, 1, 0))

    def forward(self, x):
        # (b, c, h, w)
        x1 = self.conv(x)
        x2 = self.trunk1(x1)
        x3 = self.trunk2(x2) + x2
        x3 = self.down(x3)
        x4 = self.trunk3(x3)
        x5 = self.trunk3(x4)
        mask = f.sigmoid(self.mask(x5))  # channel = out_features(192)
        mask = torch.sum(mask, 1, true)  # 对各个channel求和得到mask
        # Upsample ???
        mask = self.up(mask, scale_factor=8)

        return mask
