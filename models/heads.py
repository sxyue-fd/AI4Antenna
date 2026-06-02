import torch.nn as nn
import torch.nn.functional as F


class S11Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PatternHead(nn.Module):
    def __init__(self, in_dim, fp, p, t):
        super().__init__()
        self.fp = fp
        self.p = p
        self.t = t

        self.fc = nn.Linear(in_dim, 128 * 30)

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

        self.out = nn.Conv1d(128, fp * p, 1)

    def forward(self, x):
        x = self.fc(x)                  # [B, 128*30]
        x = x.view(-1, 128, 30)

        x = F.interpolate(x, size=self.t, mode="linear", align_corners=False)

        x = self.decoder(x)
        x = self.out(x)

        return x.view(-1, self.fp, self.p, self.t) 
    