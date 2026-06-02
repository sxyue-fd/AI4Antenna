import torch.nn as nn
from .backbone import DeepCNNBackbone
from .heads import S11Head, PatternHead


class ForwardProxyNet(nn.Module):
    def __init__(self, in_ch, y_dim, fp, p, t):
        super().__init__()

        self.backbone = DeepCNNBackbone(in_ch)

        self.s11_head = S11Head(512, y_dim)
        self.pattern_head = PatternHead(512, fp, p, t)

    def forward(self, x):
        feat = self.backbone(x)

        y = self.s11_head(feat)
        p = self.pattern_head(feat)

        return y, p


def build_forward_proxy_model(cfg, dataset_info):
    x_shape = dataset_info["x_shape"]
    y_shape = dataset_info["y_shape"]
    p_shape = dataset_info["pattern_shape"]

    return ForwardProxyNet(
        in_ch=x_shape[0],
        y_dim=y_shape[0],
        fp=p_shape[0],
        p=p_shape[1],
        t=p_shape[2],
    )