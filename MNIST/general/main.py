import os
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model import PartProtoVAE
from datamodule import MNISTDataModule
from utils import load_parameters, create_dir, save_plots

from settings import img_size, val_ratio


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
            model = PartProtoVAE(img_size,
                                 params['input_channels'],
                                 params['ce_coeff'],
                                 params['kl_coeff'],
                                 params['recon_coeff'],
                                 params['clst_coeff'],
                                 params['sep_coeff'],
                                 params['lr']
                                 )

            # Creating Logging Directory, callbacks and checkpoint
            best_model_ckpt = os.path.join(params['ckpt_path'], params['logging_name'])
            create_dir(best_model_ckpt)

            early_stopping_callback = EarlyStopping(
                monitor='val_loss', min_delta=0.01,
                patience=5,
                verbose=True,
                mode='min'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=best_model_ckpt,  # Directory to save checkpoints
                filename=f'best_model',  # Prefix for the checkpoint filenames
                save_top_k=1,  # Save the best model only
                mode='min',
                every_n_epochs=1
            )

            # initialize trainer
            tb_logger = TensorBoardLogger(params['logging_dir'], name=params['logging_name'], log_graph=False)
            trainer = pl.Trainer(
                accelerator=params['accelerator'],
                max_epochs=params['max_epochs'],
                strategy=params['strategy'],
                callbacks=[early_stopping_callback, checkpoint_callback],
                enable_progress_bar=True, check_val_every_n_epoch=1, enable_checkpointing=True,
                logger=tb_logger
            )

            trainer.fit(model, data_module)
            trainer.test(model, data_module)

            # save test loss
            model.save_test_loss_data(best_model_ckpt)

            # save plots
            train_epoch_hist, val_epoch_hist = model.get_history()
            hist_path = os.path.join(params['logging_dir'], params['logging_name'], 'plots')
            create_dir(hist_path)
            save_plots(train_epoch_hist, val_epoch_hist, hist_path)

            # save model
            model_dest = os.path.join(params['ckpt_path'], params['logging_name'], 'saved_model')
            create_dir(model_dest)
            save_entire_model(model, model_dest)


if __name__ == '__main__':
    # parser = ArgumentParser(description="Load JSON files from a directory")
    # parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    # args = parser.parse_args()
    # folder_name = args.folder_name

    # directory_path = os.path.abspath(folder_name)
    directory_path = "C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\MNIST\\params"

    # Calling the main function
    main_func(directory_path)