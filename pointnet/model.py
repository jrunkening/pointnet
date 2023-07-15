import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self, activate) -> None:
        super().__init__()

        self.activate = activate()

        self.up = nn.Sequential(
            nn.Linear(3, 64),
            self.activate,
            nn.Linear(64, 128),
            self.activate,
            nn.Linear(128, 1024),
            self.activate,
        )
        self.down = nn.Sequential(
            nn.Linear(1024, 512),
            self.activate,
            nn.Linear(512, 256),
            self.activate,
        )

        self.matmul = nn.Linear(256, 9)

    def forward(self, xs):
        h = self.up(xs)
        h = nn.MaxPool1d(h.size(1))(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.down(h)
        h = self.matmul(h)

        return h.reshape(h.size(0), 3, 3)
