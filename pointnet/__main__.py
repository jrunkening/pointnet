from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from pointnet.data import ModelNet40H5, stack_collate_fn
from pointnet.model import PointNet


if __name__ == "__main__":
    train_loader = DataLoader(
        ModelNet40H5(phase="train", data_root=Path(__file__).parent.parent.joinpath("data")),
        collate_fn=stack_collate_fn,
        batch_size=16,
    )
    for data in train_loader:
        xs, ys = data["coordinates"], data["labels"]
        break
    print(nn.functional.one_hot(ys, num_classes=40).shape)
    print(PointNet(40)(xs).shape)
