import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, y_weight=1.0, p_weight=1.0):
        super().__init__()
        self.y_weight = y_weight
        self.p_weight = p_weight
        self.loss_y = nn.MSELoss()
        self.loss_p = nn.MSELoss()
        #self.loss_y = nn.L1Loss()
        #self.loss_p = nn.L1Loss()

    def forward(self, y_pred, y_true, p_pred, p_true, return_components=False):
        ly = self.loss_y(y_pred, y_true)
        lp = self.loss_p(p_pred, p_true)
        total = self.y_weight * ly + self.p_weight * lp

        if return_components:
            return total, ly.detach(), lp.detach()

        return total


def build_loss_function(cfg):
    return MultiTaskLoss(
        cfg["loss"]["y_weight"],
        cfg["loss"]["p_weight"]
    )