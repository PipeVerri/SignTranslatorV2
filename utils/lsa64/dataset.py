import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LSA64Dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [f for f in os.listdir(root) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        label = int(filename.split("_")[0]) - 1 # -1 para que los labels empiezen en 0
        arr = np.load(os.path.join(self.root, filename))

        if arr.shape[0] == 0:
            raise ValueError(f"Empty array at {filename}")

        tensor = torch.from_numpy(arr).float() # Numpy trabaja con float64 pero torch con float32
        return tensor, label