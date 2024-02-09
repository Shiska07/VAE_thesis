import os
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

NUM_WORKERS = 4

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
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


def get_dataloaders(train_dir, train_ann_path, test_dir, test_ann_path, batch_size, val_ratio=0.15):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # train data object
    ddsm_train = CustomDataset(train_dir, train_ann_path,
                                      transform=transform)

    # get indices for validation
    labels = [label for _, label in ddsm_train]
    train_idxs, val_idxs = train_test_split(np.arange(len(ddsm_train)),
                                            test_size=val_ratio,
                                            random_state=42,
                                            shuffle=True,
                                            stratify=labels)

    # test data object
    ddsm_test = CustomDataset(test_dir, test_ann_path, transform=transform)

    # get dataloaders
    dl_train = DataLoader(Subset(ddsm_train, train_idxs), batch_size, shuffle=True, num_workers=NUM_WORKERS)
    dl_val = DataLoader(Subset(ddsm_train, val_idxs), batch_size, shuffle=False, num_workers=NUM_WORKERS)
    dl_test = DataLoader(ddsm_test, batch_size, num_workers=NUM_WORKERS)

    return dl_train, dl_val, dl_test