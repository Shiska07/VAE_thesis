import os
import pdb
import sys
import torch
import zipfile
from argparse import ArgumentParser
from datamodule import CUBDataModule
from configs import epoch_configs, loss_configs
from train_model import TrainingPipiline
from model import PartProtoVAE
from utils.helpers import load_parameters, create_dir, save_model_specs_txt


def save_entire_model(model, model_dest):
    # save the entire model
    model_architecture_path = os.path.join(model_dest, 'arc.pth')
    model_weights_path = os.path.join(model_dest, 'weights.pth')
    torch.save(model, model_architecture_path)
    torch.save(model.state_dict(), model_weights_path)
    print(f'Model saved at {model_dest}')


def main_func(params_dir_path):
    for filename in os.listdir(params_dir_path):

        if filename.endswith('.json'):
            file_path = os.path.join(params_dir_path, filename)

            params = load_parameters(file_path)
            params['filename'] = filename

            data_module = CUBDataModule(params['data_dir'],
                                        params['train_batch_size'],
                                        params['test_batch_size'],
                                        params['push_batch_size'],
                                        params['num_dl_workers'])

            # set configuration files
            loss_configs.set_loss_coeffs(params)
            epoch_configs.set_epoch_configs(params)

            model = PartProtoVAE(params['base_architecture'],
                                 params['layers_to_exclude'],
                                 params['prototype_saving_dir'],
                                 params['logging_dir'],
                                 params['default_lr'])


            # initializes model with base architecture
            prev_model_ckpt = None
            training_pipeline = TrainingPipiline(params, data_module, prev_model_ckpt)

            # train added vae component
            # training_pipeline.fit_warm_vae()
            # training_pipeline.test_model()


    return 0


if __name__ == '__main__':

    # zip_file = "data.zip"
    #
    # with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    #     zip_ref.extractall(os.getcwd())
    # parser = ArgumentParser(description="Load JSON files from a directory")
    # parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    # args = parser.parse_args()
    # folder_name = args.folder_name
    # directory_path = os.path.abspath(folder_name)

    directory_path = ".\\params"
    # directory_path = ".\\params"

    # Calling the main function
    ret_val = main_func(directory_path)