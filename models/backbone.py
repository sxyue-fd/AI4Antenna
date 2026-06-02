import torch.nn as nn
from .blocks import ConvBNAct, ResBlock


class DeepCNNBackbone(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()

        self.net = nn.Sequential(
            # 大核起步：更快看到整体几何关系
            ConvBNAct(in_ch, 64, k=3, s=1, p=1),
            ConvBNAct(64, 64, k=3, s=1, p=1),
            ResBlock(64),

            # 转入小核
            ConvBNAct(64, 128, k=3, s=1, p=1),
            ResBlock(128),

            # 第一次下采样
            ConvBNAct(128, 128, k=3, s=2, p=1),
            ResBlock(128),

            ConvBNAct(128, 256, k=3, s=1, p=1),
            ResBlock(256),

            # 第二次下采样
            ConvBNAct(256, 256, k=3, s=2, p=1),
            ResBlock(256),

            ConvBNAct(256, 512, k=3, s=1, p=1),
            ResBlock(512),


            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)