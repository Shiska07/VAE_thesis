import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from task import PartProtoVAE
from datamodule import MNISTDataModule
from utils import load_parameters, create_dir

def main_func(params_dir_path):
    for filename in os.listdir(directory_path):

        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)

            params = load_parameters(file_path)
            params['filename'] = filename

            data_module = MNISTDataModule(params['batch_size'], params['data_dir'], params['random_seed'],
                                          params['val_ratio'], params['num_dl_workers'])


            # Iniatiating the model
            model = PartProtoVAE(params['input_height'],
                                 params['input_channels'],
                                 params['kl_coeff'],
                                 params['latent_channels'],
                                 params['lr']
            )


            # Creating Logging Directory, callbacks and checkpoint
            create_dir(params['logging_dir']+params['logging_name'])
            create_dir(params['plots_dir'])
            create_dir(params['ckpt_dir'])

            early_stopping_callback = EarlyStopping(
                monitor='val_loss', min_delta=0.01,
                patience=5,
                verbose=True,
                mode='min'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=params['ckpt_path'],  # Directory to save checkpoints
                filename=f'best_model',  # Prefix for the checkpoint filenames
                save_top_k=1,  # Save the best model only
                mode='min',
                every_n_epochs=1
            )

            # initialize trainer
            tb_logger = TensorBoardLogger(params['logging_dir'], name=params['logging_name'], log_graph=False)
            trainer = pl.Trainer(
                accelerator = params['accelerator'],
                max_epochs = params['max_epochs'],
                strategy = params['strategy'],
                callbacks=[early_stopping_callback, checkpoint_callback],
                enable_progress_bar=True, check_val_every_n_epoch=1, enable_checkpointing=True,
                logger = tb_logger
            )

            trainer.fit(model, data_module)
            # trainer.test(model, data_module)
    


if __name__ == '__main__':
    parser = ArgumentParser(description="Load JSON files from a directory")
    parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    args = parser.parse_args()
    folder_name = args.folder_name

    directory_path = os.path.abspath(folder_name)

    # Calling the main function
    main_func(directory_path)
    