import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CassavaDataset(Dataset):
    """PyTorch Dataset for Cassava Leaf Disease Classification."""
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label
