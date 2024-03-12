import os
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from utils.preprocess import mean, std
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

class CUBDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 train_batch_size,
                 test_batch_size,
                 push_batch_size,
                 num_workers=4):

        super().__init__()

        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir,
                                      "cub200_cropped", "train_cropped_augmented")
        self.test_dir = os.path.join(self.data_dir, "cub200_cropped",
                                     "test_cropped")
        self.train_push_dir = os.path.join(self.data_dir, "cub200_cropped", "train_cropped")
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.push_batch_size = push_batch_size
        self.num_workers = num_workers

        self.val_size = 0.2
        self.n_classes = 200
        self.input_channels = 1
        self.input_height = 224
        self.transform = transforms.Compose([transforms.Resize(size=(
            self.input_height, self.input_height)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                 std=std)])

        # load the dataset
        self.train_dataset = datasets.ImageFolder(self.train_dir,
                                                  transform=self.transform)
        self.train_push_dataset = datasets.ImageFolder(self.train_push_dir,
                                                       transform=self.transform)
        self.val_dataset = datasets.ImageFolder(self.train_dir,
                                                transform=self.transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir,
                                                 transform=self.transform)

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.val_size * num_train))

        # np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          sampler=self.train_sampler,
                                               num_workers=self.num_workers)

    def train_push_dataloader(self):
        return DataLoader(self.train_push_dataset, batch_size=self.push_batch_size,
                          sampler=self.train_sampler,
                                               num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size,
                          sampler=self.val_sampler,
                                             num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                          shuffle=False, num_workers=self.num_workers)

