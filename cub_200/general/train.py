import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from general.task import VAE
from general.datamodule import DataModule
from utils.os_tools import create_dir, load_parameters

def main_func(params_dir_path):
    for filename in os.listdir(directory_path):

        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)

            params = load_parameters(file_path)
            params['filename'] = filename

            data_module = DataModule(params['data_dir'], params['val_ratio'], params['resizing_factor'],
                                     params['batch_size'], params['logging_dir'], params['num_dl_workers'])

            # Iniatiating the model
            model = VAE()

            # Creating Logging Directory
            create_dir(args.logging_dir)

            tb_logger = TensorBoardLogger(args.logging_dir, name=args.logging_name, log_graph=True)
            trainer = pl.Trainer(
                accelerator = args.accelerator,
                devices = args.devices,
                max_epochs = args.max_epochs,
                strategy = args.strategy,
                logger = tb_logger,
                gradient_clip_val=0.5,
            )

            trainer.fit(model, data_module)
            trainer.test(model, data_module)
    
if __name__ == '__main__':
    parser = ArgumentParser(description="Load JSON files from a directory")
    parser.add_argument("folder_name", help="Name of the folder containing JSON files")
    args = parser.parse_args()
    folder_name = args.folder_name

    directory_path = os.path.abspath(folder_name)

    # Calling the main function
    main_func(directory_path)
    