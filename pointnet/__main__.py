import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from pointnet.data import ModelNet40H5, stack_collate_fn
from pointnet.model import PointNet


DATA_PATH = Path(__file__).parent.parent.joinpath("data")
MODEL_PATH = Path(__file__).parent.parent.joinpath("model")


def train_loop(dataloader, model: PointNet, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, data in enumerate(dataloader):
        # get data
        xs, ys = data["coordinates"].to(device), data["labels"].to(device)

        # calculate loss
        loss = loss_fn(model(xs), ys)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(xs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model: PointNet, loss_fn, device):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            xs, ys = data["coordinates"].to(device), data["labels"].to(device)

            pred = model(xs).detach()
            test_loss += loss_fn(pred, ys).item()
            correct += (pred.argmax(1) == ys).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss


def main(on_gpu=True):
    device = "cuda" if on_gpu and torch.cuda.is_available() else "cpu"
    batch_size = 16

    train_dataloader = DataLoader(
        ModelNet40H5(phase="train", data_root=DATA_PATH),
        collate_fn=stack_collate_fn,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ModelNet40H5(phase="test", data_root=DATA_PATH),
        collate_fn=stack_collate_fn,
        batch_size=batch_size,
        shuffle=False,
    )

    model = PointNet(40)
    model.load_state_dict(torch.load(MODEL_PATH.joinpath(os.listdir(MODEL_PATH, )[-1])))
    learning_rate = 5e-4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 500

    for t in range(epochs):
        model = model.to(device)

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

        model = model.to("cpu")
        torch.save(model.state_dict(), MODEL_PATH.joinpath("model.pth"))

    print("Done!")


if __name__ == "__main__":
    main()
