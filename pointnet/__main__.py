from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from pointnet.data import ModelNet40H5, stack_collate_fn
from pointnet.model import TNet


if __name__ == "__main__":
    train_loader = DataLoader(
        ModelNet40H5(phase="train", data_root=Path(__file__).parent.parent.joinpath("data")),
        collate_fn=stack_collate_fn,
        batch_size=16,
    )
    for data in train_loader:
        xs, ys = data["coordinates"], data["labels"]
        break
    print(xs.shape)
    print(TNet(nn.ReLU)(xs).shape)
