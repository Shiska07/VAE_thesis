import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.datasets as datasets
from preprocess import mean, std, preprocess_input_function
from settings import img_size, train_dir, test_dir, train_push_dir


class CUBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./",
                 val_size=0.2, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.val_size = val_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                 std=std)])


    def setup(self, stage: str):

        # load the dataset
        self.train_dataset = datasets.ImageFolder(train_dir,transform=self.transform)
        self.train_push_dataset = datasets.ImageFolder(train_push_dir,
                                                       transform=self.transform)
        self.val_dataset = datasets.ImageFolder(train_dir,transform=self.transform)
        self.test_dataset =  datasets.ImageFolder(test_dir,transform=self.transform)

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.val_size * num_train))


        # np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler,
                                               num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler,
                                             num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

