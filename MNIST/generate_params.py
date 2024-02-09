import os
import json
import random
import numpy as np
from itertools import combinations

params_dict = {
    "data_dir": "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\data",
    "logging_dir": "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\general\\logs",
    "logging_name": "params1",
    "plots_dir": "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\general\\plots",
    "ckpt_path": "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\general\\checkpoints",
    "random_seed": 42,
    "val_ratio": 0.2,
    "num_classes": 10,
    "general_dir": "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\general",
    "num_dl_workers": 4,
    "input_height": 28,
    "input_channels": 1,
    "kl_coeff": 0.1,
    "latent_channels": 16,
    "lr": 0.002,
    "accelerator": "cpu",
    "strategy": "auto",
    "max_epochs": 15,
    "batch_size": 64,
    "sigma": 4,
    "alpha": 20,
    "input_width": 28,
    "n_prototypes": 20
}

ranges_dict ={
    "kl_coeff": (np.linspace(0.1, 0.9, 10)).tolist(),
    "recon_coeff": 1,
    "clst_coeff": 1,
    "sep_coeff": 1,
    "latent_channels": list(range(6, 18, 2)),
    "lr": [0.1, 0.01, 0.001, 0.0001, 0.00001],
    "min_delta": (np.linspace(0.1, 2, 10)).tolist(),
    "max_epochs": list(range(2, 15, 2)),
    "batch_size": [32, 64, 128],
}

curr_dir = os.getcwd()
params_dir_name = "params"
save_dir = os.path.join(curr_dir, params_dir_name)

from general.helpers import create_dir

def create_n_dictionaries(n_):

    for i in range(n_):
        new_dict = params_dict.copy()
        for key, values in ranges_dict.items():
            new_dict[key] = random.choice(values)

        # logging name based on bathc_size, channel_numbers and file number
        new_dict["logging_name"] = "params_" + str(new_dict["batch_size"]) + "_" + \
                                   str(new_dict["latent_channels"]) + "_" + str(i) + \
                                    str(new_dict["exp_run"])



        # make dir based on batchsize and
        create_dir(os.path.join(save_dir, f'batchsz_{str(new_dict["batch_size"])}'))

        # destination path to json file
        json_file_path = os.path.join(save_dir, f'batchsz_{str(new_dict["batch_size"])}', f'{new_dict["logging_name"]}.json')

        # save .json file
        with open(json_file_path, 'w') as json_file:
            json.dump(new_dict, json_file, indent=4)


n = 10
create_n_dictionaries(n)

