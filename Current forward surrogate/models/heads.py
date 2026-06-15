import torch.nn as nn


class S11Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

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
        self.out_channels = fp * p
        self.init_seq_len = 15

        if self.init_seq_len * 8 != t:
            raise ValueError(f"Expected t={self.init_seq_len * 8} for three 2x upsampling stages, got {t}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.out_channels * self.init_seq_len),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.out_channels, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GELU(),

            nn.Conv1d(128, self.out_channels, kernel_size=3, padding=1, padding_mode="circular"),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = x.view(batch_size, self.out_channels, self.init_seq_len)
        x = self.decoder(x)

        return x.view(batch_size, self.fp, self.p, self.t)
    
