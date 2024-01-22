import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


class CUB200Datatset(Dataset):
    
    def __init__(
        self,
        image_dir,
        annotations_file_path,
        transform=None,
        target_transform=None
    ):
        super().__init__()

        self.img_labels = pd.read_csv(annotations_file_path)
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        rgb_image = image.convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            rgb_image = self.transform(rgb_image)
        if self.target_transform:
            label = self.target_transform(label)
        return rgb_image, int(label)