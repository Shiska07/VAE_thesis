import os
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import PartProtoVAE
from datamodule import MNISTDataModule
from settings import img_size, val_ratio
from helpers import load_parameters, create_dir, save_plots
from custom_callbacks import early_stopping_callback, getmodel_ckpt_callback, \
    clear_prev_logs, PrototypeProjectionCallback

def save_entire_model(model, model_dest):
    # save the entire model
    model_architecture_path = os.path.join(model_dest, 'arc.pth')
    model_weights_path = os.path.join(model_dest, 'weights.pth')
    torch.save(model, model_architecture_path)
    torch.save(model.state_dict(), model_weights_path)
    print(f'Model saved at {model_dest}')


def main_func(params_dir_path):
    for filename in os.listdir(directory_path):

        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)

            params = load_parameters(file_path)
            params['filename'] = filename

            data_module = MNISTDataModule(params['batch_size'], params['data_dir'], params['random_seed'],
                                          val_ratio, params['num_dl_workers'])

            # Iniatiating the model
            protov_model = PartProtoVAE(img_size,
                                        params['input_channels'],
                                        params['ce_coeff'],
                                        params['kl_coeff'],
                                        params['recon_coeff'],
                                        params['clst_coeff'],
                                        params['sep_coeff'],
                                        params['l1_coeff'],
                                        params['prototype_saving_dir'],
                                        params['lr']
                                        )

            # clear previous logs
            logging_dir = os.path.join(params['logging_dir'], params['logging_name'])
            clear_prev_logs(logging_dir)

            # Creating Logging Directory, callbacks and checkpoint
            best_model_ckpt_path = os.path.join(logging_dir, 'checkpoints')
            create_dir(best_model_ckpt_path)

            checkpoint_callback = getmodel_ckpt_callback(best_model_ckpt_path,
                                                         params)

            # initialize trainer
            tb_logger = TensorBoardLogger(params['logging_dir'], name=params['logging_name'], log_graph=False)
            trainer = pl.Trainer(
                accelerator=params['accelerator'],
                max_epochs=params['max_epochs'],
                strategy=params['strategy'],
                callbacks=[early_stopping_callback, checkpoint_callback,
                           PrototypeProjectionCallback()],
                enable_progress_bar=True,
                check_val_every_n_epoch=1,
                enable_checkpointing=True,
                log_every_n_steps=1,
                logger=tb_logger,
                limit_train_batches=10
                # limit_val_batches=1,
                # limit_test_batches=1
            )

            protov_model.set_mode("vae_only")
            trainer.fit(protov_model, data_module)
            trainer.test(protov_model, data_module)

            # save model
            model_dest = os.path.join(best_model_ckpt_path, 'saved_model')

            create_dir(model_dest)
            save_entire_model(protov_model, model_dest)

            return 0


if __name__ == '__main__':
    # parser = ArgumentParser(description="Load JSON files from a directory")
    # parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    # args = parser.parse_args()
    # folder_name = args.folder_name

    # directory_path = os.path.abspath(folder_name)
    directory_path = "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\params"

    # Calling the main function
    ret_val = main_func(directory_path)