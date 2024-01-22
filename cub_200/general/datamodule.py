import argparse
from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import train_test_split
import numpy as np

from general.dataset import CUB200Dataset
from utils.stats_tools import DataLoaderStats
from utils.os_tools import load_np_array, save_transformation


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        val_ratio,
        resizing_factor,
        batch_size,
        logging_dir,
        num_dataloader_workers

    ):

        super().__init__()

        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.resizing_factor = resizing_factor
        self.batch_size = batch_size
        self.logging_dir = logging_dir
        self.num_dataloader_workers = num_dataloader_workers

        # normalization statistics(need to change)
        self.mean = [0.42956483, 0.42956483, 0.42956483]
        self.std = [0.09959752, 0.09959752, 0.09959752]

        self.transform = transforms.Compose([
            transforms.Resize((self.resizing_factor, self.resizing_factor)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        
    def setup(self, stage=None):
        
        # from the main data directory get paths to the test and train folders
        train_dir = join(self.data_dir, "train_cropped_augmented/")
        train_annotations_path = join(self.data_dir, "train_cropped_augmented_annotations.csv")
        
        test_dir = join(self.data_dir, "test_cropped/")
        test_annotations_path = join(self.data_dir, "test_cropped_annotations.csv")

        '''
                transforms_list = []
        inverse_transforms_list = []
        final_size = self.patch_size

        if self.normalize_transform:
            std = load_np_array(join(self.data_dir, "std.gz"))
            mean = load_np_array(join(self.data_dir, "mean.gz"))
            print("\n")
            print(f"mean of training set used for normalization: {mean}")
            print(f"std of training set used for normalization: {std}")
            print("\n")
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
            inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1, 1, 1])))
            inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0, 0, 0]), std=1/std))

        if self.resize_transform_size is not None:
            transforms_list.append(transforms.Resize(size=self.resize_transform_size, interpolation=InterpolationMode.BILINEAR))
            inverse_transforms_list.insert(0, transforms.Resize(size=self.patch_size, interpolation=InterpolationMode.BILINEAR))
            final_size = self.resize_transform_size
        transforms_list.append(transforms.CenterCrop(final_size))

        # Composing and saving transformations and inverse transformations to file
        transformations = transforms.Compose(transforms_list)
        inverse_transformations = transforms.Compose(inverse_transforms_list)

        save_transformation(transformations, join(self.logging_dir, "trans.obj"))
        save_transformation(inverse_transformations, join(self.logging_dir, "inv_trans.obj"))
        '''
        # the train and validation set will be created from the complete training dataset
        self.complete_train_data = CUB200Dataset(train_dir, train_annotations_path, self.transform)

        # get indices for validation
        labels = [label for _, label in self.complete_train_data]
        train_idxs, val_idxs = train_test_split(np.arange(len(self.complete_train_data)),
                                            test_size=self.val_ratio,
                                            random_state=42,
                                            shuffle=True,
                                            stratify=labels)
        
        # Creating corresponding datasets
        if stage in (None, "fit"):
            self.train_dataset = Subset(self.complete_train_data, train_idxs)
            self.val_dataset = Subset(self.complete_train_data, val_idxs)
        elif stage in (None, "validate"):
            self.val_dataset = DataLoader(Subset(self.complete_train_data, val_idxs), self.batch_size, shuffle=True, num_workers=self.      num_dataloader_workers)
        elif stage in (None, "test"):
            self.test_dataset = CUB200Dataset(test_dir, test_annotations_path, self.transform)

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_dataloader_workers, drop_last=True)