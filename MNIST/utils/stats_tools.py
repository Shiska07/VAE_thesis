from os.path import join

import torch
import numpy as np

from utils.os_tools import save_np_array

class DataLoaderStats:
    def __init__(self, loader, stats_dir):
        # Accumulating means and stds
        self.std = torch.tensor([0, 0, 0])
        self.mean = torch.tensor([0, 0, 0])

        for image_batch, _ in loader:
            # Last batches may be smaller than self.batch_size
            batch_size = image_batch.shape[0]
            for batch_id in range(batch_size):
                std, mean = torch.std_mean(image_batch[batch_id], dim=(1,2), unbiased=False)

                # Not the correct calculation, but a good enough estimate.
                # TODO->Find an efficient implementation of Welford's algorithm
                self.std = self.std + std
                self.mean = self.mean + mean
        
        self.std = self.std / len(loader.dataset)
        self.mean = self.mean / len(loader.dataset)

        save_np_array(self.std.numpy(), join(stats_dir, "std.gz"))
        save_np_array(self.mean.numpy(), join(stats_dir, "mean.gz"))

        print(f"Stats are calculated and saved for the dataloader: {stats_dir}")

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean
