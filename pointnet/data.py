from pathlib import Path
import os
import subprocess
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import h5py


class ModelNet40H5(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str,
        transform=None,
        num_points=2048,
    ):
        Dataset.__init__(self)

        self.data_root = Path(data_root)
        self.download_dataset()

        self.data_root = self.data_root.joinpath("modelnet40_ply_hdf5_2048")
        phase = "test" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(phase)

        self.transform = transform
        self.phase = phase
        self.num_points = num_points

    def download_dataset(self):
        if os.path.exists(self.data_root.joinpath("modelnet40_ply_hdf5_2048.zip")):
            print("Use downloaded dataset")
            return

        print("Downloading the 2k down-sampled ModelNet40 dataset...")
        subprocess.run([
            "wget",
            f"--directory-prefix={self.data_root}",
            "--no-check-certificate",
            "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
        ])
        subprocess.run([
            "unzip",
            f"-d{self.data_root}",
            self.data_root.joinpath("modelnet40_ply_hdf5_2048.zip")
        ])

    def load_data(self, phase):
        data, labels = [], []
        assert os.path.exists(self.data_root), f"{self.data_root} does not exist"
        files = glob.glob(self.data_root.joinpath(f"ply_data_{phase}*.h5").as_posix())
        assert len(files) > 0, "No files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        data = np.stack(data, axis=0)
        labels = np.stack(labels, axis=0)
        return data, labels

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = torch.from_numpy(xyz)
        label = torch.from_numpy(label)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"ModelNet40H5(phase={self.phase}, length={len(self)}, transform={self.transform})"
