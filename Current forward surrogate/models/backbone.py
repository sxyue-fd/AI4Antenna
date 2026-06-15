import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity

        return self.act(out)


class DeepCNNBackbone(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNAct(in_ch, 64, k=3, s=1, p=1),
            ConvBNAct(64, 64, k=3, s=1, p=1),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)
