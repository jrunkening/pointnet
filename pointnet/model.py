import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, n) -> None:
        super().__init__()

        self.bn = nn.BatchNorm1d(n)

    def forward(self, xs):
        return self.bn(xs.permute(0, 2, 1)).permute(0, 2, 1)


class TNet(nn.Module):
    def __init__(self, activate, in_dim, out_dim) -> None:
        super().__init__()

        self.activate = activate
        self.out_dim = out_dim

        self.up = nn.Sequential(
            nn.Linear(in_dim, 64),
            Norm(64),
            self.activate,
            # nn.Linear(64, 64),
            # Norm(64),
            # self.activate,
            # nn.Linear(64, 64),
            # Norm(64),
            # self.activate,
            # nn.Linear(64, 64),
            # Norm(64),
            # self.activate,

            nn.Linear(64, 128),
            Norm(128),
            self.activate,
            # nn.Linear(128, 128),
            # Norm(128),
            # self.activate,
            # nn.Linear(128, 128),
            # Norm(128),
            # self.activate,
            # nn.Linear(128, 128),
            # Norm(128),
            # self.activate,

            nn.Linear(128, 1024),
            Norm(1024),
            self.activate,
            # nn.Linear(1024, 1024),
            # Norm(1024),
            # self.activate,
            # nn.Linear(1024, 1024),
            # Norm(1024),
            # self.activate,
            # nn.Linear(1024, 1024),
            # Norm(1024),
            # self.activate,
        )
        self.down = nn.Sequential(
            nn.Linear(1024, 512),
            Norm(512),
            self.activate,
            nn.Linear(512, 256),
            Norm(256),
            self.activate,
        )

        self.matmul = nn.Linear(256, self.out_dim*self.out_dim)

    def forward(self, xs):
        h = self.up(xs)
        h = torch.max(h, dim=1, keepdim=True)[0]
        h = self.down(h)
        h = self.matmul(h)

        return h.reshape(h.size(0), self.out_dim, self.out_dim)


class PointNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.activate = nn.ReLU()

        self.input_transform = TNet(self.activate, 3, 3)
        self.mlp0 = nn.Sequential(
            nn.Linear(3, 64),
            Norm(64),
            self.activate,
            nn.Linear(64, 64),
            Norm(64),
            self.activate,
        )
        self.feature_transform = TNet(self.activate, 64, 64)
        self.mlp1 = nn.Sequential(
            nn.Linear(64, 64),
            Norm(64),
            self.activate,
            nn.Linear(64, 128),
            Norm(128),
            self.activate,
            nn.Linear(128, 1024),
            Norm(1024),
            self.activate,
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            Norm(512),
            self.activate,
            nn.Linear(512, 256),
            Norm(256),
            self.activate,
            nn.Linear(256, num_classes),
            Norm(num_classes),
            self.activate,
        )

    def forward(self, xs):
        h = xs.bmm(self.input_transform(xs))
        h = self.mlp0(h)
        h = h.bmm(self.feature_transform(h))
        h = self.mlp1(h)
        h = torch.max(h, dim=1, keepdim=True)[0]
        h = self.mlp2(h)

        return h.squeeze()
