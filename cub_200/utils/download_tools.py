from os.path import join, exists
from urllib import request
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# Ref: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, save_dir):
    filename = url.split('/')[-1] + ".svs"
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=10, miniters=1, desc=filename) as t:
        request.urlretrieve(url, filename=join(save_dir, filename), reporthook=t.update_to)
        
def download_gdc_data(ID, num_samples, csv_path, save_dir):
    rng = np.random.default_rng(seed=ID)

    df = pd.read_csv(csv_path)
    df["size"] = df["size"] / 1e9   # Converting Bytes to Gigabytes
    df = df[df["size"].between(1.25, 1.5, inclusive="both")].reset_index(drop=True)    # Only selecting from slides that are between 1.25 GB and 1.5 GB
    ids = df.sample(n=num_samples, random_state=rng, axis=0)["id"].tolist()

    base_url = "https://api.gdc.cancer.gov/data/"
    view_url = "https://portal.gdc.cancer.gov/files/"
    for id in ids:
        file_url = base_url + id
        file_path = join(save_dir, id) + ".svs"

        if not exists(file_path):
            print(f"Downloading {file_url}...")
            print(f"You can view the image at {view_url + id}...")
            try:
                download(file_url, save_dir)
            except HTTPError as e:
                print("Something went wrong downloading slides from GDC!\n", e)
        else:
            print(f"{file_path} already exists! Download skipped...")
            print(f"You can view the image at {view_url + id}...")
            