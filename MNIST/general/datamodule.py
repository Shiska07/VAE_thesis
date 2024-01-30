import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./", random_seed = 42,
                 val_size=0.2, num_workers=4, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_size = val_size
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor()
                                             ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # load the dataset
            self.train_dataset = MNIST(root=self.data_dir, train=True,
                                            transform=self.transform)
            self.val_dataset = MNIST(root=self.data_dir, train=True,
                                         transform=self.transform)

            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.val_size * num_train))


            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            self.train_sampler = SubsetRandomSampler(train_idx)
            self.val_sampler = SubsetRandomSampler(val_idx)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                                               num_workers=self.num_workers, persistent_workers=True, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler,
                                             num_workers=self.num_workers, persistent_workers=True, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers, persistent_workers=True, pin_memory=False)

