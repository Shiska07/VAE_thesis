import os
import json
import random
import numpy as np
from itertools import combinations

params_dict = {
    "data_dir": "/MNIST/data",
    "logging_dir": "/MNIST/logs",
    "general_dir": "/MNIST/general",
    "prototype_saving_dir": "/MNIST/prototypes",
    "exp_run": "vae_convergence_testing5",
    "logging_name": "params1",
    "random_seed": 42,
    "recon_coeff": 1,
    "kl_coeff": 0.1,
    "ce_coeff": 0,
    "clst_coeff": 0,
    "sep_coeff": 0,
    "l1_coeff": 0,
    "num_dl_workers": 4,
    "input_channels": 1,
    "lr": 0.001,
    "accelerator": "gpu",
    "strategy": "auto",
    "max_epochs": 30,
    "batch_size": 64
}

ranges_dict ={
    "kl_coeff": (np.linspace(0.01, 0.2, 100)).tolist(),
    "lr": [0.001, 0.0001],
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
        new_dict["logging_name"] = "params_" + str(new_dict["batch_size"]) + "_" \
                                   + str(i) + \
                                    str(new_dict["exp_run"])



        # make dir based on batchsize and
        create_dir(os.path.join(save_dir, f'batchsz_{str(new_dict["batch_size"])}'))

        # destination path to json file
        json_file_path = os.path.join(save_dir, f'batchsz_{str(new_dict["batch_size"])}', f'{new_dict["logging_name"]}.json')

        # save .json file
        with open(json_file_path, 'w') as json_file:
            json.dump(new_dict, json_file, indent=4)


n = 15
create_n_dictionaries(n)

